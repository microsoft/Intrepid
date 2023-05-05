import torch
import random
import numpy as np
import torch.optim as optim

from model.misc.independence_test_model import IndependenceTestModel
from utils.cuda import cuda_var


class IndependenceTest:
    def __init__(self, config, constants):
        self.config = config
        self.epoch = 10  # constants["encoder_training_epoch"]
        self.learning_rate = constants["encoder_training_lr"]
        self.batch_size = constants["encoder_training_batch_size"]
        self.validation_size_portion = constants["validation_data_percent"]
        self.entropy_coeff = constants["entropy_reg_coeff"]
        self.num_homing_policies = constants["num_homing_policy"]
        self.patience = constants["patience"]
        self.max_retrials = constants["max_try"]
        self.expected_optima = constants["expected_optima"]  # If

    def _calc_loss(self, model, batch):
        model_input = cuda_var(
            torch.cat(
                [torch.from_numpy(np.array(point[0])).view(1, -1) for point in batch],
                dim=0,
            )
        ).float()
        gold_labels = cuda_var(
            torch.cat(
                [torch.from_numpy(np.array(point[1])).view(1, -1) for point in batch],
                dim=0,
            )
        ).long()

        # Compute loss
        log_probs = model.gen_log_prob(
            model_input=model_input
        )  # outputs a matrix of size batch x 2
        loss = -torch.mean(log_probs.gather(1, gold_labels.view(-1, 1)))

        return loss

    def train_model(self, dataset, logger, tensorboard):
        # torch.manual_seed(ctr)
        print("Solving dataset with stats %r" % (len(dataset)))

        # Current model
        model = IndependenceTestModel(
            self.config, model_input_dim=2 * self.config["atom_dim"], hidden_dim=10
        )

        # Model for storing the best model as measured by performance on the test set
        best_model = IndependenceTestModel(
            self.config, model_input_dim=2 * self.config["atom_dim"], hidden_dim=10
        )

        param_with_grad = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(params=param_with_grad, lr=self.learning_rate)

        random.shuffle(dataset)
        dataset_size = len(dataset)
        batches = [
            dataset[i : i + self.batch_size]
            for i in range(0, dataset_size, self.batch_size)
        ]

        train_batch = int((1.0 - self.validation_size_portion) * len(batches))
        train_batches = batches[:train_batch]
        test_batches = batches[train_batch:]

        best_test_loss, best_epoch, train_loss = 0.69, -1, 0.69  # 0.69 is -log(2)
        num_train_examples, num_test_examples = 0, 0
        patience_counter = 0

        for epoch_ in range(1, self.epoch + 1):
            train_loss, num_train_examples = 0.0, 0
            for train_batch in train_batches:
                loss = self._calc_loss(model, train_batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 40)
                optimizer.step()

                loss = float(loss)
                tensorboard.log_scalar("Encoding_Loss ", loss)

                batch_size = len(train_batch)
                train_loss = train_loss + float(loss) * batch_size
                num_train_examples = num_train_examples + batch_size

            train_loss = train_loss / float(max(1, num_train_examples))

            # Evaluate on test batches
            test_loss = 0
            num_test_examples = 0
            for test_batch in test_batches:
                loss = self._calc_loss(model, test_batch)

                batch_size = len(test_batch)
                test_loss = test_loss + float(loss) * batch_size
                num_test_examples = num_test_examples + batch_size

            test_loss = test_loss / float(max(1, num_test_examples))
            logger.debug(
                "Train Loss after max_epoch %r is %r" % (epoch_, round(train_loss, 2))
            )
            logger.debug(
                "Test Loss after max_epoch %r is %r" % (epoch_, round(test_loss, 2))
            )

            if test_loss < best_test_loss:
                patience_counter = 0
                best_test_loss = test_loss
                best_epoch = epoch_
                best_model.load_state_dict(model.state_dict())
            else:
                # Check patience condition
                patience_counter += 1  # number of max_epoch since last increase
                if (
                    best_test_loss < self.expected_optima or test_loss > 0.8
                ):  # Found good solution or diverged
                    break

                if patience_counter == self.patience:
                    logger.log(
                        "Patience Condition Triggered: No improvement for %r epochs"
                        % patience_counter
                    )
                    break

        logger.log(
            "FactoRL, Train/Test = %d/%d, Best Tune Loss %r at max_epoch %r, "
            "Train Loss after %r epochs is %r "
            % (
                num_train_examples,
                num_test_examples,
                round(best_test_loss, 2),
                best_epoch,
                epoch_,
                round(train_loss, 2),
            )
        )

        return best_model, best_test_loss

    def test_model(self, decoder, children_factor, dataset, k):
        correspondence = np.zeros((2, 2))
        for dp in dataset:
            atoms = np.array([dp.next_obs[i] for i in children_factor])
            out = decoder.encode_observations(atoms)
            correspondence[dp.next_state[k], out] += 1

        print("Learned Decoder Correspondence Matrix \n", correspondence)

    @staticmethod
    def _create_ind_test_dataset(dataset):
        ind_test_dataset = []

        z_examples = [dp[1] for dp in dataset]

        for x, y in dataset:
            # Add positive example dataset
            xy = np.array([x, y])
            ind_test_dataset.append((xy, 1))

            # Add negative example
            z = random.choice(z_examples)
            xz = np.array([x, z])
            ind_test_dataset.append((xz, 0))

        return ind_test_dataset

    def is_independent(self, dataset, logger, tensorboard):
        """
        :param dataset: List of two random variables X, Y
        :return:    True if X and Y are independent else False
        """

        ind_test_dataset = self._create_ind_test_dataset(dataset)

        best_model, best_test_loss = self.train_model(
            ind_test_dataset, logger, tensorboard
        )

        if best_test_loss >= 0.65:
            return True
        else:
            return False
