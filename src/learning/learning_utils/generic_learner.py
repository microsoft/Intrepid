import torch.optim as optim

from model.model_wrapper import ModelWrapper
from learning.learning_utils.clustering_algorithm import *


class GenericLearner:
    """Class for training a supervised learner. Fairly generic with minimal assumption"""

    def __init__(self, exp_setup):
        self.config = exp_setup.config
        self.constants = exp_setup.constants
        self.logger = exp_setup.logger

        self.max_epoch = exp_setup.constants["max_epoch"]
        self.learning_rate = exp_setup.constants["learning_rate"]
        self.batch_size = exp_setup.constants["batch_size"]
        self.dev_pct = exp_setup.constants["validation_data_percent"]
        self.patience = exp_setup.constants["patience"]
        self.grad_clip = exp_setup.constants["grad_clip"]

    def calc_loss(self, model, batch, test=False):
        """
        :param model: Model for calculating loss
        :param batch: Given a batch of datapoints
        :param test: Run in eval mode if True otherwise run in train mode
        returns a loss tensor along with a information dictionary
        """
        raise NotImplementedError()

    def _overwrite_params(self, **kwargs):
        return Setup(
            learning_rate=self.learning_rate
            if "learning_rate" not in kwargs
            else kwargs["learning_rate"],
            batch_size=self.batch_size
            if "batch_size" not in kwargs
            else kwargs["batch_size"],
            max_epoch=self.max_epoch
            if "max_epoch" not in kwargs
            else kwargs["max_epoch"],
            patience=self.patience if "patience" not in kwargs else kwargs["patience"],
            grad_clip=self.grad_clip
            if "grad_clip" not in kwargs
            else kwargs["grad_clip"],
            dev_pct=self.dev_pct if "dev_pct" not in kwargs else kwargs["dev_pct"],
        )

    def do_train(
        self,
        model_type,
        model_name,
        dataset,
        tensorboard=None,
        bootstrap_model=None,
        **kwargs
    ):
        # Current model
        model = ModelWrapper.get_model(
            model_type=model_type,
            model_name=model_name,
            config=self.config,
            constants=self.constants,
            bootstrap_model=bootstrap_model,
            **kwargs
        )

        # Model for storing the best model as measured by performance on the test set
        best_model = ModelWrapper.get_model(
            model_type=model_type,
            model_name=model_name,
            config=self.config,
            constants=self.constants,
            bootstrap_model=bootstrap_model,
            **kwargs
        )

        setup = self._overwrite_params(**kwargs)
        self.logger.log(
            "Generic Learning (%s, %s): [Setup %s]" % (model_type, model_name, setup)
        )

        param_with_grad = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(params=param_with_grad, lr=setup.learning_rate)

        random.shuffle(dataset)
        dataset_size = len(dataset)
        batches = [
            dataset[i : i + setup.batch_size]
            for i in range(0, dataset_size, setup.batch_size)
        ]

        train_batch = int((1.0 - setup.dev_pct) * len(batches))
        train_batches = batches[:train_batch]
        test_batches = batches[train_batch:]

        best_test_loss, best_epoch, train_loss = float("inf"), -1, float("inf")
        num_train, num_test = 0, 0
        epoch, patience_ctr = -1, 0

        for epoch in range(1, setup.max_epoch + 1):
            train_loss, num_train = 0.0, 0
            for train_batch in train_batches:
                loss, info_dict = self.calc_loss(model, train_batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), setup.grad_clip)
                optimizer.step()

                loss = float(loss)

                if tensorboard is not None:
                    tensorboard.log_scalar("%s/Encoding_Loss" % model_type, loss)

                    for key in info_dict:
                        tensorboard.log_scalar(
                            "%s/%s" % (model_type, key), info_dict[key]
                        )

                batch_size = len(train_batch)
                train_loss += loss * batch_size
                num_train += batch_size

            train_loss = train_loss / float(max(1, num_train))

            # Evaluate on test batches
            test_loss = 0
            num_test = 0
            for test_batch in test_batches:
                loss, _ = self.calc_loss(model, test_batch)

                batch_size = len(test_batch)
                test_loss += float(loss) * batch_size
                num_test += batch_size

            test_loss = test_loss / float(max(1, num_test))
            self.logger.debug(
                "Epoch %d: Train Loss %.4f, Test Loss %.4f"
                % (epoch, train_loss, test_loss)
            )

            if test_loss < best_test_loss:
                patience_ctr = 0
                best_test_loss = test_loss
                best_epoch = epoch
                best_model.load_state_dict(model.state_dict())
            else:
                # Check patience condition
                patience_ctr += 1  # number of max_epoch since last increase

                if patience_ctr == setup.patience:
                    self.logger.log(
                        "Patience Condition Triggered: No improvement for last %d epochs"
                        % patience_ctr
                    )
                    break

        self.logger.log(
            "%s Trained, Train/Test = %d/%d, Best Tune Loss %.4f at max_epoch %d, "
            "Train Loss after %d epochs is %.4f "
            % (
                model_type,
                num_train,
                num_test,
                best_test_loss,
                best_epoch,
                epoch,
                train_loss,
            )
        )

        return best_model, {"best_test_loss": best_test_loss}


class Setup:
    def __init__(
        self, learning_rate, batch_size, max_epoch, patience, grad_clip, dev_pct
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.patience = patience
        self.grad_clip = grad_clip
        self.dev_pct = dev_pct

        self.str_fmt = (
            "LR: %f, BS: %d, Max Epoch: %d, Patience: %d, Grad Clip %f, Dev Pct %f%%"
            % (
                self.learning_rate,
                self.batch_size,
                self.max_epoch,
                self.patience,
                self.grad_clip,
                self.dev_pct,
            )
        )

    def __str__(self):
        return self.str_fmt
