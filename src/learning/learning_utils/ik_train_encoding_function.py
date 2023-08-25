import random
import torch
import torch.optim as optim

from learning.state_abstraction.inverse_kinematics import InverseKinematics
from model.inverse_dynamics.simple_feed_forward import SimpleFeedForwardIK


class IKTrainEncodingFunction:
    """Class for training the encoding function"""

    def __init__(self, config, constants):
        self.config = config
        self.constants = constants
        self.epoch = constants["encoder_training_epoch"]
        self.learning_rate = constants["encoder_training_lr"]
        self.batch_size = constants["encoder_training_batch_size"]
        self.validation_size_portion = constants["validation_data_percent"]
        self.entropy_coeff = constants["entropy_reg_coeff"]
        self.num_homing_policies = constants["num_homing_policy"]

        self.ik_learner = InverseKinematics(constants, self.epoch)
        self.patience = constants["patience"]
        self.max_retrials = constants["max_try"]
        self.expected_optima = constants["expected_optima"]  # If the model reaches this loss then we exit

    def train_model(self, dataset, logger, bootstrap_model, discretized, tensorboard):
        # Current model
        model = SimpleFeedForwardIK(self.config, self.constants, bootstrap_model)

        # Model for storing the best model as measured by performance on the test set
        best_model = SimpleFeedForwardIK(self.config, self.constants, bootstrap_model)

        param_with_grad = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(params=param_with_grad, lr=self.learning_rate)

        random.shuffle(dataset)
        dataset_size = len(dataset)
        batches = [dataset[i : i + self.batch_size] for i in range(0, dataset_size, self.batch_size)]

        train_batch = int((1.0 - self.validation_size_portion) * len(batches))
        train_batches = batches[:train_batch]
        test_batches = batches[train_batch:]

        best_test_loss, best_epoch, train_loss = (
            float("inf"),
            -1,
            float("inf"),
        )  # 0.69 is -log(2)
        num_train_examples, num_test_examples = 0, 0
        patience_counter = 0

        test_set_errors, past_entropy = [], []

        for epoch_ in range(1, self.epoch + 1):
            train_loss, mean_entropy, num_train_examples = 0.0, 0.0, 0
            for train_batch in train_batches:
                loss, info_dict = self.ik_learner.calc_loss(
                    model,
                    train_batch,
                    epoch_,
                    discretized,
                    test_set_errors,
                    past_entropy,
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 40)
                optimizer.step()

                loss = float(loss)
                tensorboard.log_scalar("Encoding_Loss ", loss)

                for key in info_dict:
                    tensorboard.log_scalar(key, info_dict[key])

                batch_size = len(train_batch)
                train_loss = train_loss + float(info_dict["classification_loss"]) * batch_size
                mean_entropy = mean_entropy + float(info_dict["mean_entropy"]) * batch_size
                num_train_examples = num_train_examples + batch_size

            train_loss = train_loss / float(max(1, num_train_examples))
            mean_entropy = mean_entropy / float(max(1, num_train_examples))

            # Evaluate on test batches
            test_loss = 0
            num_test_examples = 0
            for test_batch in test_batches:
                _, info_dict = self.ik_learner.calc_loss(
                    model,
                    test_batch,
                    epoch_,
                    discretized,
                    test_set_errors,
                    past_entropy,
                )

                batch_size = len(test_batch)
                test_loss = test_loss + float(info_dict["classification_loss"]) * batch_size
                num_test_examples = num_test_examples + batch_size

            test_loss = test_loss / float(max(1, num_test_examples))
            logger.debug(
                "Train Loss after max_epoch %r is %r, mean entropy %r, entropy coeff %r"
                % (
                    epoch_,
                    round(train_loss, 2),
                    round(mean_entropy, 2),
                    info_dict["entropy_coeff"],
                )
            )
            logger.debug("Test Loss after max_epoch %r is %r" % (epoch_, round(test_loss, 2)))

            test_set_errors.append(test_loss)
            past_entropy.append(mean_entropy)

            if test_loss < best_test_loss:
                patience_counter = 0
                best_test_loss = test_loss
                best_epoch = epoch_
                best_model.load_state_dict(model.state_dict())
            else:
                # Check patience condition
                patience_counter += 1  # number of max_epoch since last increase

                if patience_counter == self.patience:
                    logger.log("Patience Condition Triggered: No improvement for %r epochs" % patience_counter)
                    break

        logger.log(
            "Inverse kinematics (Discretized: %r), Train/Test = %d/%d, Best Tune Loss %r at max_epoch %r, "
            "Train Loss after %r epochs is %r "
            % (
                discretized,
                num_train_examples,
                num_test_examples,
                round(best_test_loss, 2),
                best_epoch,
                epoch_,
                round(train_loss, 2),
            )
        )

        num_state_budget = self.constants["num_homing_policy"]

        return best_model, num_state_budget
