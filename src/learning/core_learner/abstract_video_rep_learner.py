import time
import torch
import random
import numpy as np

from torch import optim
from utils.cuda import cuda_var
from utils.tensorboard import Tensorboard
from utils.beautify_time import elapsed_from_str
# from learning.core_learner.ppo_learner import PPOLearner
from learning.learning_utils.reconstruct_observation import ReconstructObservation
from learning.learning_utils.collect_data_with_coverage import CollectDatawithCoverage


class AbstractVideoRepLearner:

    def __init__(self, exp_setup, create_state=False):

        self.exp_setup = exp_setup
        self.save_img_flag = False
        self.save_path = exp_setup.config["save_path"]
        self.experiment = exp_setup.experiment

        # If stream is set to False, then use offline data, else collect as you go
        self.stream = exp_setup.constants["stream"]

        # Environment parameters
        self.horizon = exp_setup.config["horizon"]
        self.actions = exp_setup.config["actions"]

        self.create_state = create_state

        # Learning parameters
        self.max_epoch = exp_setup.constants["encoder_training_epoch"]
        self.learning_rate = exp_setup.constants["encoder_training_lr"]
        self.batch_size = exp_setup.constants["encoder_training_batch_size"]
        self.dev_pct = exp_setup.constants["validation_data_percent"]
        self.patience = exp_setup.constants["patience"]
        self.grad_clip = exp_setup.constants["grad_clip"]
        self.dataset_size = exp_setup.constants["max_episodes"]

        # Data collector
        self.data_collector = CollectDatawithCoverage(exp_setup)

        # Logging and evaluation
        self.logger = exp_setup.logger
        self.reconstructor = ReconstructObservation(exp_setup)

        self.max_k = exp_setup.constants["max_k"]

        if self.save_img_flag:
            self.logger.log("Warning! Save image flag is on meaning that images will be saved during the process. "
                            "This should be ideally set only during debugging as it can significantly slow down "
                            "the experiment.")

    @staticmethod
    def _get_observations_from_episode(episodes, shuffle=True):

        observations = []
        for eps in episodes:
            for obs in eps.get_observations():
                observations.append(obs)

        if shuffle:
            random.shuffle(observations)

        return observations

    def _get_multi_transition_dataset(self, episodes, max_k=None, shuffle=True):

        if max_k is None:
            max_k = self.max_k

        dataset = []
        for eps in episodes:
            for k in range(1, max_k + 1):
                if self.create_state:
                    for transition in eps.get_multi_step_state_observation_transitions(k):
                        dataset.append(transition)
                else:
                    for transition in eps.get_multi_step_observation_transitions(k):
                        dataset.append(transition)

        if shuffle:
            random.shuffle(dataset)

        return dataset

    @staticmethod
    def make_model(exp_setup):
        """
        :param exp_setup: Experimental setup
        :return:    Return a model for training
        """
        raise NotImplementedError()

    def _calc_loss(self, model, prep_batch, test=False):
        """
                Given model and a prepared batch (i.e., which has been processed for device, return calculated loss(es)
        :param model:           Model used for training
        :param prep_batch:      A batch of multi-step transitions
        :param test:            If test=True, then run it test mode else False
        :return:
                    - base loss: This is the main representation loss by which we measure performance
                    - total loss: This will typically include some auxiliary losses along with base loss
                    - info_dict:

        """
        raise NotImplementedError()

    def _accumulate_info_dict(self, info_dicts):
        """
                Given a list of info_dicts, accumulate their result and return a new info_dict with mean results.
        :param info_dicts: List of dictionary containg floats
        :return: return a single dictionary with mean results
        """

        raise NotImplementedError()

    def log_info_dicts(self, prefix, info_dicts):
        results_str = ", ".join(["%s: %f" % (key, val) for key, val in sorted(info_dicts.items())
                                 if type(val) == int or type(val) == float])
        self.logger.log("%s => %s" % (prefix, results_str))

        for key, val in sorted(info_dicts.items()):
            if type(val) != int and type(val) != float:
                self.logger.log("%s: %s -> %r" % (prefix, key, val))

    @staticmethod
    def prep_batch(batch):
        # pdb.set_trace()
        actions = [dp[1] for dp in batch]
        klst = [dp[3] for dp in batch]
        actions = cuda_var(torch.Tensor(actions).long())
        klst = cuda_var(torch.Tensor(klst).long())

        obs1 = np.array([np.array(dp[0]) for dp in batch])
        obs2 = np.array([np.array(dp[2]) for dp in batch])

        obs1 = torch.FloatTensor(obs1)  # 2 batch x height x width x channel
        obs2 = torch.FloatTensor(obs2)

        obs1 = cuda_var(obs1.permute(0, 3, 1, 2))
        obs2 = cuda_var(obs2.permute(0, 3, 1, 2))

        if len(batch[0]) == 6:
            states = [state for _, _, _, _, state, _ in batch]
            next_states = [next_state for _, _, _, _, _, next_state in batch]
            return actions, obs1, obs2, klst, states, next_states
        else:
            return actions, obs1, obs2, klst

    def _collect_dataset(self, env, dataset_size=None, split=True):

        # Collect episodes
        episodes = self.data_collector.collect_episodes(env,
                                                        dataset_size=self.dataset_size
                                                        if dataset_size is None else dataset_size)

        random.shuffle(episodes)
        num_episodes = len(episodes)

        if split:
            train_size = int((1.0 - self.dev_pct) * num_episodes)
            train_episodes = episodes[:train_size]
            test_episodes = episodes[train_size:]

            # Create multi-transition dataset
            train_transition_dataset = self._get_multi_transition_dataset(train_episodes)
            test_transition_dataset = self._get_multi_transition_dataset(test_episodes)

            # Create observation dataset
            train_obs = self._get_observations_from_episode(train_episodes)
            test_obs = self._get_observations_from_episode(test_episodes)

            return train_episodes, test_episodes, train_obs, test_obs, train_transition_dataset, test_transition_dataset
        else:
            # Create multi-transition dataset
            train_transition_dataset = self._get_multi_transition_dataset(episodes)

            # Create observation dataset
            observations = self._get_observations_from_episode(episodes)

            return episodes, observations, train_transition_dataset

    def _train_decoder(self, encoder, train_obs, test_obs, tensorboard):

        best_reconst_decoder, decoder_results = self.reconstructor.train(encoder=encoder,
                                                                         dataset=train_obs,
                                                                         tensorboard=tensorboard)

        base_folder = "%s/final_reconstructed_images" % self.experiment
        self.reconstructor.reconstruct(encoder=encoder,
                                       decoder=best_reconst_decoder,
                                       test_dataset=test_obs,
                                       base_folder=base_folder)

        return best_reconst_decoder, decoder_results

    def _do_rl(self, env, encoder):

        # Reset Progress CSV file (important)
        time_s = time.time()

        # We reset the flags so that we can use the progress.csv file later to plot PPO results
        env.reset_metrics("PPO Training Starts")
        learning_alg = PPOLearner(self.exp_setup, encoder=encoder)
        policy_result = learning_alg.do_train(env)
        ppo_final_policy = learning_alg.ppo_learner.policy

        self.logger.log("PPO training took %s" % elapsed_from_str(time_s))

        return ppo_final_policy, policy_result

    def do_train(self, env):
        """
            Train and evaluate a representation learning model from video data
            :param env: Environment on which to train the representation
            :return: Performance of the learned model
        """

        # Step 1: Initialize model, optimizer, and tensorboard
        model = self.make_model(self.exp_setup)
        best_model = self.make_model(self.exp_setup)  # model for storing the current best model

        param_with_grad = filter(lambda p: p.requires_grad, list(model.parameters()))
        optimizer = optim.Adam(params=param_with_grad, lr=self.learning_rate)

        # Tensorboard
        tensorboard = Tensorboard(log_dir=self.experiment)
        self.logger.log("Model, Adam Optimizer, and Tensorboard created.")

        # Step 2: Collect dataset
        time_dataset = time.time()

        if self.stream:
            val_size = int(self.dev_pct * self.dataset_size)
            test_episodes, test_obs, test_transitions = self._collect_dataset(env, dataset_size=val_size, split=False)
            self.logger.log("Test Dataset collected of size %d in %s" %
                            (len(test_transitions), elapsed_from_str(time_dataset)))
        else:
            train_episodes, test_episodes, train_obs, test_obs, train_transitions, test_transitions = \
                self._collect_dataset(env)
            self.logger.log("Dataset collected with Train=%d and Test=%d in %s" %
                            (len(train_transitions), len(test_transitions), elapsed_from_str(time_dataset)))

        # Sub-sample data so that it doesn't become too big with increasing k
        random.shuffle(test_transitions)
        size = len(test_transitions) // self.max_k
        test_transitions = test_transitions[:size]
        test_batches = [test_transitions[i:i + self.batch_size]
                        for i in range(0, len(test_transitions), self.batch_size)]

        best_test_loss, best_epoch, train_loss = float('inf'), -1, float('inf')
        best_epoch_train_info_dicts, best_epoch_test_info_dicts = None, None
        num_train, num_test = 0, 0
        epoch, patience_ctr = -1, 0
        time_start_epoch = time.time()

        # Step 3: Train a representation learner for video data
        for epoch in range(1, self.max_epoch + 1):

            time_s = time.time()

            if self.stream:
                # Collect train dataset
                train_episodes, train_obs, train_transitions = self._collect_dataset(env, split=False)

            # Sample a smaller data to avoid increasing the epoch time with k. This allows diversity in each epoch
            # without significantly increasing compute time. Note that test set remains fixed throughout training,
            # so that its performance can be compared from one time step another.
            random.shuffle(train_transitions)
            size = len(train_transitions)
            epoch_train_transitions = train_transitions[:size]

            train_batches = [epoch_train_transitions[i:i + self.batch_size]
                             for i in range(0, len(epoch_train_transitions), self.batch_size)]

            # Train on the train batches
            train_loss = 0.0
            num_train = 0
            train_info_dicts = []

            for train_batch in train_batches:

                if len(train_batch) < self.batch_size:
                    continue

                prep_train_batch = self.prep_batch(train_batch)
                base_loss, loss, info_dict = self._calc_loss(model, prep_train_batch)

                train_info_dicts.append(info_dict)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()

                base_loss_val = base_loss.item()
                if tensorboard is not None:
                    tensorboard.log_scalar("Representation_Learning_Video/Mean_Base_Loss ", base_loss_val)

                    for key, val in info_dict.items():
                        if type(val) == float:
                            tensorboard.log_scalar("Representation_Learning_Video/%s" % key, val)

                batch_size = len(train_batch)
                train_loss += base_loss_val * batch_size
                num_train += batch_size

            train_loss = train_loss / float(max(1, num_train))
            train_info_dicts = self._accumulate_info_dict(train_info_dicts)

            self.log_info_dicts(prefix="Epoch %d [Train] Train Loss=%.4f" % (epoch, train_loss),
                                info_dicts=train_info_dicts)

            # Evaluate on test batches
            test_loss = 0
            num_test = 0
            test_info_dicts = []

            for test_batch in test_batches:

                if len(test_batch) < self.batch_size:
                    continue

                prep_test_batch = self.prep_batch(test_batch)
                base_loss, loss, info_dict = self._calc_loss(model, prep_test_batch, test=True)
                test_info_dicts.append(info_dict)

                batch_size = len(test_batch)
                test_loss += base_loss.item() * batch_size
                num_test += batch_size

            test_loss = test_loss / float(max(1, num_test))
            test_info_dicts = self._accumulate_info_dict(test_info_dicts)
            self.log_info_dicts(prefix="Epoch %d [Test] Test Loss=%.4f" % (epoch, test_loss),
                                info_dicts=test_info_dicts)

            if test_loss < best_test_loss:

                patience_ctr = 0
                best_test_loss = test_loss
                best_epoch_train_info_dicts = train_info_dicts.copy()
                best_epoch_test_info_dicts = test_info_dicts.copy()
                best_epoch = epoch

                best_model.load_state_dict(model.state_dict())

                self.logger.log("Epoch %d [Completed]: Best Test Loss Improved to %.4f, Time taken %s" %
                                (epoch, best_test_loss, elapsed_from_str(time_s)))

            else:
                # Check patience condition
                patience_ctr += 1  # number of max_epoch since last increase

                self.logger.log("Epoch %d [Completed]: Best Test Loss remains at %.4f, Patience increased to %d/%d, "
                                "Time taken %s" %
                                (epoch, best_test_loss, patience_ctr, self.patience, elapsed_from_str(time_s)))

                if patience_ctr == self.patience:
                    self.logger.log("Patience Condition Triggered: No improvement for last %d epochs" % patience_ctr)
                    break

        self.logger.log("Base Video Model Trained, Train/Test = %d/%d, Best Tune Base Loss %.4f at Epoch %d, "
                        "Train Base Loss after %d epochs is %.4f. Total time %s" %
                        (num_train, num_test, best_test_loss, best_epoch, epoch, train_loss,
                         elapsed_from_str(time_start_epoch)))

        results = {
            "first_stage/final_train_base_loss": train_loss,
            "first_stage/best_epoch": best_epoch,
            "first_stage/best_test_base_loss": best_test_loss
        }

        for key, val in best_epoch_train_info_dicts.items():
            results["first_stage/best_epoch_train_%s" % key] = val

        for key, val in best_epoch_test_info_dicts.items():
            results["first_stage/best_epoch_test_%s" % key] = val

        for key, val in train_info_dicts.items():
            results["first_stage/final_epoch_train_%s" % key] = val

        for key, val in test_info_dicts.items():
            results["first_stage/final_epoch_test_%s" % key] = val

        models_to_save = {
            "first_stage/best_model": best_model.state_dict(),
            "first_stage/final_model": model.state_dict(),
            "first_stage/optimizer": optimizer.state_dict()
        }

        # Stage 4: Train a decoder to reconstruct images from the dataset
        if hasattr(best_model, "encoder"):

            self.logger.log("Two stage not enabled and but base model has an encoder.")
            self.logger.log("Evaluating the encoder by reconstructing observation from the fixed encoder")

            # if self.stream:
            #     # Collect train dataset
            #     self.logger.log(f"Since we are in streaming mode, collecting 10x more episodes to train the decoder. "
            #                     f"Collecting {10 * self.dataset_size} for training the decoder.")
            #     _, train_obs, _ = self._collect_dataset(env, dataset_size=(10 * self.dataset_size), split=False)
            #
            # best_reconst_decoder, decoder_results = self._train_decoder(best_model.encoder, train_obs,
            #                                                             test_obs, tensorboard)

            self.logger.log("Evaluating Encoder by performing RL with PPO.")
            ppo_final_policy, rl_results = self._do_rl(env, best_model.encoder)

            # models_to_save["decoder/best_model"] = best_reconst_decoder.state_dict()
            models_to_save["encoder/chosen_model"] = best_model.encoder.state_dict()
            models_to_save["rl/final_policy"] = ppo_final_policy.state_dict()

            # for key, val in decoder_results.items():
            #     results["decoder/%s" % key] = val

            for key, val in rl_results.items():
                results["rl/%s" % key] = val

        else:
            self.logger.log("Two stage not enabled and found no encoder in the base model. "
                            "Skipping any evaluation that are done for an encoder such as reconstructing"
                            "decoder or performing RL.")

        torch.save(models_to_save, "%s/final_checkpoint" % self.experiment)

        return results
