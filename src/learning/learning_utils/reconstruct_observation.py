import os
import time
import torch
import random
import imageio
import numpy as np
import torch.optim as optim

from utils.cuda import cuda_var
from skimage.transform import resize
from utils.beautify_time import elapsed_from_str
from model.decoder.decoder_wrapper import DecoderModelWrapper


class ReconstructObservation:
    def __init__(self, exp_setup):
        self.logger = exp_setup.logger

        self.height, self.width, self.channel = exp_setup.config["obs_dim"]
        self.out_dim = exp_setup.constants[
            "vq_dim"
        ]  # exp_setup.constants["hidden_dim"]
        self.experiment = exp_setup.experiment

        self.max_epoch = exp_setup.constants["encoder_training_epoch"]
        self.learning_rate = exp_setup.constants["encoder_training_lr"]
        self.batch_size = exp_setup.constants["encoder_training_batch_size"]
        self.dev_pct = exp_setup.constants["validation_data_percent"]
        self.patience = exp_setup.constants["patience"]
        self.grad_clip = exp_setup.constants["grad_clip"]
        self.model_type = exp_setup.constants["decoder_type"]

    @staticmethod
    def _calc_loss(decoder, batch, encoder=None):
        obs = cuda_var(torch.FloatTensor(np.array([dp[0] for dp in batch])))
        encoding = cuda_var(torch.vstack([dp[1] for dp in batch]))

        if obs.dim() == 4:  # Image
            obs = obs.transpose(2, 3).transpose(
                1, 2
            )  # batch x channel x height x width

        # encoding = encoder.encode(obs)
        x_pred = decoder.decode(encoding)
        diff = x_pred - obs
        loss = diff * diff  # batch x  channel x height x width
        loss = loss.mean()

        return loss, dict()

    def _encode_dataset(self, encoder, dataset):
        batches = [
            dataset[i : i + self.batch_size]
            for i in range(0, len(dataset), self.batch_size)
        ]
        processed_dataset = []

        for batch in batches:
            obs = cuda_var(
                torch.FloatTensor(np.array(batch))
            )  # batch x height x width x channel

            if obs.dim() == 4:  # Image
                obs = obs.permute(0, 3, 1, 2)  # batch x channel x height x width

            vec = encoder.encode(obs).detach().cpu()  # batch x dim
            processed_dataset.extend([(dp, vec[i]) for i, dp in enumerate(batch)])

        return processed_dataset

    def train(self, encoder, dataset, tensorboard=None):
        self.logger.log(
            "Autoencoder Training Starts with Model Type=%s" % self.model_type
        )

        # Current model
        decoder = DecoderModelWrapper.get_decoder(
            self.model_type,
            height=self.height,
            width=self.width,
            channel=self.channel,
            out_dim=self.out_dim,
            bootstrap_model=None,
        )

        # Model for storing the best model as measured by performance on the test set
        best_decoder = DecoderModelWrapper.get_decoder(
            self.model_type,
            height=self.height,
            width=self.width,
            channel=self.channel,
            out_dim=self.out_dim,
            bootstrap_model=None,
        )

        param_with_grad = filter(lambda p: p.requires_grad, decoder.parameters())
        optimizer = optim.Adam(params=param_with_grad, lr=self.learning_rate)

        self.logger.log("Encoding Dataset of size %d" % len(dataset))
        time_start = time.time()
        dataset = self._encode_dataset(encoder, dataset)
        self.logger.log("Encoding Completed in %s" % elapsed_from_str(time_start))

        random.shuffle(dataset)
        dataset_size = len(dataset)

        train_size = int((1.0 - self.dev_pct) * dataset_size)
        train_dataset = dataset[:train_size]
        test_dataset = dataset[train_size:]

        test_batches = [
            test_dataset[i : i + self.batch_size]
            for i in range(0, len(test_dataset), self.batch_size)
        ]

        best_test_loss, best_epoch, train_loss = float("inf"), -1, float("inf")
        num_train, num_test = 0, 0
        epoch, patience_ctr = -1, 0

        time_start = time.time()
        self.logger.log(
            "Train/Test = %d/%d; Learning Rate %f, Max Epochs %d, Patience %d"
            % (
                len(train_dataset),
                len(test_dataset),
                self.learning_rate,
                self.max_epoch,
                self.patience,
            )
        )

        for epoch in range(1, self.max_epoch + 1):
            time_epoch_start = time.time()

            # Create a batch
            random.shuffle(train_dataset)
            train_batches = [
                train_dataset[i : i + self.batch_size]
                for i in range(0, len(train_dataset), self.batch_size)
            ]

            train_loss, num_train = 0.0, 0
            for train_batch in train_batches:
                loss, info_dict = self._calc_loss(decoder, train_batch, encoder)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), self.grad_clip)
                optimizer.step()

                loss = float(loss)
                tensorboard.log_scalar("Reconstruction_Loss ", loss)

                for key in info_dict:
                    tensorboard.log_scalar(key, info_dict[key])

                batch_size = len(train_batch)
                train_loss += loss * batch_size
                num_train += batch_size

            train_loss = train_loss / float(max(1, num_train))

            # Evaluate on test batches
            test_loss = 0
            num_test = 0
            for test_batch in test_batches:
                loss, _ = self._calc_loss(decoder, test_batch, encoder)

                batch_size = len(test_batch)
                test_loss += float(loss) * batch_size
                num_test += batch_size

            test_loss = test_loss / float(max(1, num_test))
            self.logger.debug(
                "Epoch %d: Train Loss %.4f, Test Loss %.4f, Time taken %s"
                % (epoch, train_loss, test_loss, elapsed_from_str(time_epoch_start))
            )

            if test_loss < best_test_loss:
                patience_ctr = 0
                best_test_loss = test_loss
                best_epoch = epoch
                best_decoder.load_state_dict(decoder.state_dict())
            else:
                # Check patience condition
                patience_ctr += 1  # number of max_epoch since last increase

                if patience_ctr == self.patience:
                    self.logger.log(
                        "Patience Condition Triggered: No improvement for last %d epochs"
                        % patience_ctr
                    )
                    break

        self.logger.log(
            "AutoEncoder Trained [Time Taken %s], Best Tune Loss %.4f at max_epoch %d, "
            "Train Loss after %d epochs is %.4f "
            % (
                elapsed_from_str(time_start),
                best_test_loss,
                best_epoch,
                epoch,
                train_loss,
            )
        )

        results = {
            "decoder/num_train": num_train,
            "decoder/num_test": num_test,
            "decoder/best_test_loss": best_test_loss,
            "decoder/best_epoch": best_epoch,
            "decoder/epoch": epoch,
            "decoder/train_loss": train_loss,
        }

        torch.save(
            {
                "best_encoder": encoder.state_dict(),
                "best_decoder": best_decoder.state_dict(),
                "final_decoder": decoder.state_dict(),
                "encoder_optimizer": optimizer.state_dict(),
            },
            "%s/final_decoder_checkpoint" % self.experiment,
        )

        return best_decoder, results

    def reconstruct(
        self, encoder, decoder, test_dataset, base_folder, max_generate=100
    ):
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)

        test_dataset = test_dataset[:max_generate]
        test_batches = [
            test_dataset[i : i + self.batch_size]
            for i in range(0, len(test_dataset), self.batch_size)
        ]
        test_dataset_size = len(test_dataset)

        total_loss = 0
        ctr = 0

        time_start = time.time()
        self.logger.log(
            "Reconstructing %d Observations with Autoencoder" % (len(test_dataset))
        )

        for test_batch in test_batches:
            obs = cuda_var(
                torch.FloatTensor(np.array(test_batch))
            )  # batch x height x width x channel

            if obs.dim() == 4:  # Image
                obs = obs.permute(0, 3, 1, 2)  # batch x channel x height x width

            vec = encoder.encode(obs)  # batch x dim
            x_pred = decoder.decode(vec)  # batch x channel x height x width

            diff = x_pred - obs
            loss = diff * diff  # batch x  channel x height x width
            batch_size = loss.size(0)
            loss = loss.view(batch_size, -1).mean(1).sum()

            total_loss += loss.item()

            combined_image = torch.cat(
                [obs, x_pred], dim=3
            )  # batch x channel x height x (2 width)
            combined_image = combined_image.permute(
                0, 2, 3, 1
            )  # batch x height x (2 width) x channel
            combined_image = combined_image.data.cpu()

            for i in range(0, combined_image.size(0)):
                ctr += 1
                img = combined_image[i].numpy()  # height x (2 width) x channel
                factor = max(1, int(min(500 // img.shape[0], 500 // img.shape[1])))
                img = resize(img, (img.shape[0] * factor, img.shape[1] * factor))
                imageio.imwrite("%s/image_%d.png" % (base_folder, ctr), img)
                # save_image(torch.from_numpy(img), )

        mean_loss = total_loss / float(max(1, test_dataset_size))
        self.logger.log(
            "Reconstruction: Total observations %d, Mean loss %d, Time taken %s"
            % (test_dataset_size, mean_loss, elapsed_from_str(time_start))
        )

        return mean_loss
