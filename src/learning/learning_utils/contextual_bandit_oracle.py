import torch
import numpy as np
import torch.optim as optim

from utils.cuda import cuda_var
from model.policy.stationary_deterministic_policy import StationaryDeterministicPolicy


class ContextualBanditOracle:
    """ Oracle for offline contextual bandit (Zhang and Langford 2006). The dataset for this problem
     consists of {(x, a, r, p)} where x is an observation, a is an action taken, r is the reward
     for taking action and p is the probability with which action a was taken. """

    def __init__(self, config, constants):

        self.config = config
        self.constants = constants
        self.epochs = constants["cb_oracle_epoch"]
        self.learning_rate = constants["cb_oracle_lr"]
        self.batch_size = constants["cb_oracle_batch_size"]
        self.validation_size_portion = constants["cb_validation_pct"]
        self.max_patience = constants["cb_patience"]

    @staticmethod
    def _log_dataset(dataset, logger):
        """
        :param dataset: Contextual bandit dataset
        :param logger: Logger object for logging
        :return:
        """
        logger.log("Contextual Bandit Dataset Summary:")
        transition = dict()
        counts = dict()
        for dp in dataset:
            key = "%r -> %r -> %r" % (dp[4], dp[2], dp[5])
            if key in transition:
                transition[key] += dp[3]
                counts[key] += 1.0
            else:
                transition[key] = dp[3]
                counts[key] = 1.0
        for key in sorted(transition):
            logger.log("CB:: %r, Count %r, Mean Reward %r " % (key, counts[key], transition[key] / counts[key]))

    def learn_optimal_policy(self, dataset, logger, tensorboard, debug=False):
        """
        :param dataset: list of tuple of observation, action, probability of action and reward
        :param logger: A logger class for logging to file
        :param tensorboard: Tensorboard object for visual display
        :param debug: A boolean flag that if it is true then logs additional information
        :return: reward prediction model
        """

        # A reward regressor class to predict reward for different actions
        reward_prediction_model = StationaryDeterministicPolicy(self.config, self.constants)

        # Best model to save the best performing model
        best_reward_prediction_model = StationaryDeterministicPolicy(self.config, self.constants)

        dataset_size = len(dataset)
        batches = [dataset[i:i + self.batch_size] for i in range(0, dataset_size, self.batch_size)]
        train_batch_size = int((1.0 - self.validation_size_portion) * len(batches))
        train_batches = batches[:train_batch_size]
        test_batches = batches[train_batch_size:]

        optimizer = optim.Adam(params=reward_prediction_model.parameters(), lr=self.learning_rate)

        if debug:
            ContextualBanditOracle._log_dataset(dataset, logger)

        best_train_loss, best_test_loss = float('inf'), float('inf')
        patience = 0
        epoch = 0

        for epoch in range(1, self.epochs + 1):

            train_loss, num_train_examples = 0.0, 0

            # Perform one max_epoch of training on train_batches
            for train_batch in train_batches:
                
                observation_batch = cuda_var(torch.cat([torch.from_numpy(np.array(point[0])).view(1, -1)
                                                        for point in train_batch], dim=0)).float()
                actions_batch = cuda_var(torch.cat([torch.from_numpy(np.array(point[2])).view(1, -1)
                                                    for point in train_batch], dim=0)).long()
                rewards_batch = cuda_var(torch.cat([torch.from_numpy(np.array(point[3])).view(1, -1)
                                                    for point in train_batch], dim=0)).float().view(-1)

                predicted_rewards = reward_prediction_model.gen_q_val(observation_batch)
                selected_rewards = predicted_rewards.gather(1, actions_batch.view(-1,1)).view(-1)
                loss = torch.mean((selected_rewards - rewards_batch) ** 2)  # TODO Multiply by action probability

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reward_prediction_model.parameters(), 40)
                optimizer.step()

                batch_size = len(train_batch)
                train_loss = train_loss + float(loss) * batch_size
                num_train_examples = num_train_examples + batch_size

                if tensorboard is not None:
                    tensorboard.log_scalar("CB-Oracle-Train-Loss", loss)

            train_loss = train_loss / max(1.0, float(num_train_examples))
            best_train_loss = min(best_train_loss, train_loss)

            # Evaluate on test_batches
            test_loss, num_test_examples = 0.0, 0

            for test_batch in test_batches:

                observation_batch = cuda_var(torch.cat([torch.from_numpy(np.array(point[0])).view(1, -1)
                                                        for point in test_batch], dim=0)).float()
                actions_batch = cuda_var(torch.cat([torch.from_numpy(np.array(point[2])).view(1, -1)
                                                    for point in test_batch], dim=0)).long()
                rewards_batch = cuda_var(torch.cat([torch.from_numpy(np.array(point[3])).view(1, -1)
                                                    for point in test_batch], dim=0)).float().view(-1)

                predicted_rewards = reward_prediction_model.gen_q_val(observation_batch)
                selected_rewards = predicted_rewards.gather(1, actions_batch.view(-1, 1)).view(-1)
                loss = torch.mean((selected_rewards - rewards_batch) ** 2)  # TODO Multiply by action probability

                batch_size = len(test_batch)
                test_loss = test_loss + float(loss) * batch_size
                num_test_examples = num_test_examples + batch_size

            test_loss = test_loss / float(max(1, num_test_examples))

            if test_loss <= best_test_loss:
                best_test_loss = test_loss
                patience = 0
                best_reward_prediction_model.load_state_dict(reward_prediction_model.state_dict())
            else:
                patience += 1
                if patience == self.max_patience:
                    logger.debug("Maximum patience reached")
                    break
   
        return best_reward_prediction_model, \
               {"cb-best-train-loss": best_train_loss, "cb-best-test-loss": best_test_loss, "stopping_epoch": epoch}
