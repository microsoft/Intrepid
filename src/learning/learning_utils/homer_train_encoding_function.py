import time
import torch
import random
import numpy as np
import torch.optim as optim

from learning.learning_utils.clustering_algorithm import *
from learning.datastructures.count_conditional_probability import CountConditionalProbability
from learning.learning_utils.clustering_algorithm import (
    ClusteringModel,
    CompositionalFeatureComputation,
    GreedyClustering,
)
from learning.learning_utils.homer_train_encoding_function_utils import (
    log_dataset_stats,
    log_model_performance,
)
from learning.state_abstraction.noise_contrastive_dataset import NoiseContrastiveDataset
from learning.state_abstraction.noise_contrastive_global import NoiseContrastiveGlobal
from model.transition_encoders.encoder_model_wrapper import EncoderModelWrapper
from utils.average import AverageUtil


class TrainEncodingFunction:
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
        self.clustering_threshold = constants["clustering_threshold"]

        # Model type with discretization on x' in (x, a, x')
        self.backward_model_type = constants["backward_model_type"]

        # Model type with discretization on x in (x, a, x')
        self.forward_model_type = constants["forward_model_type"]

        self.from_dataset = constants["nce_from_dataset"]

        if self.from_dataset:
            self.noise_contrastive_learner = NoiseContrastiveDataset(constants, self.epoch)
        else:
            self.noise_contrastive_learner = NoiseContrastiveGlobal(constants, self.epoch)

        self.patience = constants["patience"]

        self.max_retrials = constants["max_try"]
        self.expected_optima = constants["expected_optima"]  # If the model reaches this loss then we exit

    def train_model(
        self,
        dataset,
        logger,
        model_type,
        bootstrap_model,
        category,
        discretized,
        debug,
        tensorboard,
    ):
        # torch.manual_seed(ctr)

        if self.from_dataset:
            count = [0, 1]
            for dp in dataset:
                count[dp.is_valid()] += 1
        else:
            count = len(dataset)

        logger.log("Solving encoder model %r and got dataset with stats %r" % (model_type, count))

        # Current model
        model = EncoderModelWrapper.get_encoder_model(model_type, self.config, self.constants, bootstrap_model)

        # Model for storing the best model as measured by performance on the test set
        best_model = EncoderModelWrapper.get_encoder_model(model_type, self.config, self.constants, bootstrap_model)

        param_with_grad = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(params=param_with_grad, lr=self.learning_rate)

        random.shuffle(dataset)
        dataset_size = len(dataset)
        batches = [dataset[i : i + self.batch_size] for i in range(0, dataset_size, self.batch_size)]

        train_batch = int((1.0 - self.validation_size_portion) * len(batches))
        train_batches = batches[:train_batch]
        test_batches = batches[train_batch:]

        best_test_loss, best_epoch, train_loss = 0.69, -1, 0.69  # 0.69 is -log(2)
        num_train_examples, num_test_examples = 0, 0
        patience_counter = 0

        test_set_errors, past_entropy = [], []

        for epoch_ in range(1, self.epoch + 1):
            train_loss, mean_entropy, num_train_examples = 0.0, 0.0, 0
            for train_batch in train_batches:
                loss, info_dict = self.noise_contrastive_learner.calc_loss(
                    model,
                    train_batch,
                    epoch_,
                    discretized,
                    test_set_errors,
                    past_entropy,
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 40)
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
                _, info_dict = self.noise_contrastive_learner.calc_loss(
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
                if best_test_loss < self.expected_optima or test_loss > 0.8:  # Found good solution or diverged
                    break

                if patience_counter == self.patience:
                    logger.log("Patience Condition Triggered: No improvement for %r epochs" % patience_counter)
                    break

        logger.log(
            "%s (Discretized: %r), Train/Test = %d/%d, Best Tune Loss %r at max_epoch %r, "
            "Train Loss after %r epochs is %r "
            % (
                model_type,
                discretized,
                num_train_examples,
                num_test_examples,
                round(best_test_loss, 2),
                best_epoch,
                epoch_,
                round(train_loss, 2),
            )
        )

        if debug and discretized:
            if category == "backward":
                log_model_performance(
                    self.num_homing_policies,
                    best_model,
                    test_batches,
                    best_test_loss,
                    logger,
                )

        return best_model, best_test_loss

    def do_train(
        self,
        dataset,
        logger,
        tensorboard,
        debug,
        bootstrap_model=None,
        undiscretized_initialization=True,
        category="backward",
    ):
        # Do not bootstrap if not asked to
        if not self.constants["bootstrap_encoder_model"]:
            bootstrap_model = None

        if self.constants["discretization"]:
            # Train using a discretized model
            encoding_function, _ = self.do_train_with_discretized_models(
                dataset,
                logger,
                tensorboard,
                debug,
                bootstrap_model=bootstrap_model,
                undiscretized_initialization=undiscretized_initialization,
                category=category,
            )

            num_state_budget = self.constants["num_homing_policy"]

        else:
            # Train using an undiscretized model
            encoding_function, result_meta = self.do_train_with_undiscretized_models(
                dataset,
                logger,
                tensorboard,
                debug,
                bootstrap_model=bootstrap_model,
                category=category,
            )
            num_state_budget = result_meta["num_clusters"]

        return encoding_function, num_state_budget

    def do_train_with_discretized_models(
        self,
        dataset,
        logger,
        tensorboard,
        debug,
        bootstrap_model=None,
        undiscretized_initialization=True,
        category="backward",
    ):
        """Given a dataset comprising of (x, a, x', y) where y=1 means the x' was observed on taking action a in x and
        y=0 means it was observed independently of x, a. We train a model to differentiate between the dataset.
        The model we use has a certain structure that enforces discretization."""

        overall_best_model, overall_best_test_loss = None, float("inf")

        if category == "backward":
            model_type = self.backward_model_type
        elif category == "forward":
            model_type = self.forward_model_type
        else:
            raise AssertionError("Unhandled category %s" % category)

        if debug:
            log_dataset_stats(dataset, logger)

        for ctr in range(1, self.max_retrials + 1):
            # torch.manual_seed(ctr)

            # Current model
            if undiscretized_initialization:
                # Learn a undiscretized model
                undiscretized_model, best_test_loss = self.train_model(
                    dataset,
                    logger,
                    model_type,
                    bootstrap_model,
                    category,
                    False,
                    debug,
                    tensorboard,
                )

                # Bootstrap from the learned undiscretized model now
                my_bootstrap_model = undiscretized_model
            else:
                # Bootstrap from the input bootstrap model
                my_bootstrap_model = bootstrap_model

            best_model, best_test_loss = self.train_model(
                dataset,
                logger,
                model_type,
                my_bootstrap_model,
                category,
                True,
                debug,
                tensorboard,
            )

            if best_test_loss < overall_best_test_loss:
                overall_best_test_loss = best_test_loss
                overall_best_model = best_model

            if overall_best_test_loss < self.expected_optima:
                break
            else:
                logger.log("Failed to reach expected loss. This was attempt number %d" % ctr)

        return overall_best_model, {
            "loss": overall_best_test_loss,
            "success": overall_best_test_loss < self.expected_optima,
        }

    def do_train_with_undiscretized_models(
        self,
        dataset,
        logger,
        tensorboard,
        debug,
        bootstrap_model=None,
        category="backward",
    ):
        """Given a dataset comprising of (x, a, x', y) where y=1 means the x' was observed on taking action a in x
        and y=0 means it was observed independently of x, a. We train a model to differentiate between the dataset
        and perform clustering on the model output to learn a state abstraction function.
        """

        overall_best_model, overall_best_test_loss = None, float("inf")
        discretized = False

        if category == "backward":
            model_type = self.backward_model_type
        else:
            raise AssertionError("Unhandled category %s" % category)

        for ctr in range(1, self.max_retrials + 1):
            # torch.manual_seed(ctr)

            # Train a undiscretized model on the dataset
            best_model, best_test_loss = self.train_model(
                dataset,
                logger,
                model_type,
                bootstrap_model,
                category,
                discretized,
                debug,
                tensorboard,
            )

            if best_test_loss < overall_best_test_loss:
                overall_best_test_loss = best_test_loss
                overall_best_model = best_model

            if overall_best_test_loss < self.expected_optima:
                break
            else:
                logger.log("Failed to reach expected loss. This was attempt number %d" % ctr)

        curr_obs_actions = [(dp.curr_obs, dp.action) for dp in dataset if dp.is_valid() == 1]
        next_obs = [dp.next_obs for dp in dataset if dp.is_valid() == 1]
        valid_dataset = [dp for dp in dataset if dp.is_valid() == 1]

        # Compute features for observations
        timestep_feature_calc_start = time.time()
        logger.debug("Calculating compositional features for clustering steps. Size of dataset: %d" % len(curr_obs_actions))
        # feature_fn = FeatureComputation(curr_obs_actions=curr_obs_actions,
        #                                 model=overall_best_model,
        #                                 batch_size=1024,
        #                                 discretized=discretized)
        # vectors = [feature_fn.calc_feature(dp_.next_obs) for dp_ in valid_dataset]

        feature_fn = CompositionalFeatureComputation(
            curr_obs_actions=curr_obs_actions,
            model=overall_best_model,
            batch_size=1024,
            discretized=discretized,
        )
        vectors = feature_fn.calc_feature(next_obs)

        logger.debug("Calculated features. Time taken %d sec" % (time.time() - timestep_feature_calc_start))

        # Call the clustering algorithm to generate clusters
        threshold = 0.0  # TODO use the generalization error to define threshold
        cluster_alg = GreedyClustering(threshold=threshold, dim=feature_fn.dim)
        cluster_centers = cluster_alg.cluster(vectors)
        logger.debug("Number of clusters with L1 distance and threshold %f is %d" % (threshold, len(cluster_centers)))

        # Define the state abstraction model
        encoder_model = ClusteringModel(cluster_centers, feature_fn)
        logger.debug("Mapping datapoints to their center")
        timestep_center_assign_start = time.time()
        for dp_, feature_ in zip(valid_dataset, vectors):
            dp_.meta_dict["cluster_center"] = encoder_model.encode_observations({"vec": feature_})
        logger.debug("Done computing in time %d sec " % (time.time() - timestep_center_assign_start))

        return encoder_model, {
            "loss": overall_best_test_loss,
            "success": overall_best_test_loss < self.expected_optima,
            "num_clusters": len(cluster_centers),
        }

    def do_inc_train_with_undiscretized_models(
        self,
        dataset,
        transitions,
        logger,
        tensorboard,
        debug,
        bootstrap_model=None,
        category="backward",
    ):
        """Given a dataset comprising of (x, a, x', y) where y=1 means the x' was observed on taking action a in x
        and y=0 means it was observed independently of x, a. We train a model to differentiate between the dataset
        and perform clustering on the model output to learn a state abstraction function.
        """

        if debug:
            log_dataset_stats(dataset, logger)

        overall_best_model, overall_best_test_loss = None, float("inf")
        discretized = False

        if category == "backward":
            model_type = self.backward_model_type
        else:
            raise AssertionError("Unhandled category %s" % category)

        for ctr in range(1, self.max_retrials + 1):
            # torch.manual_seed(ctr)

            # Train a undiscretized model on the dataset
            if self.from_dataset:
                logger.log("Training model on dataset with fixed imposters")
                best_model, best_test_loss = self.train_model(
                    dataset,
                    logger,
                    model_type,
                    bootstrap_model,
                    category,
                    discretized,
                    debug,
                    tensorboard,
                )
            else:
                logger.log("Training model on transitions")
                best_model, best_test_loss = self.train_model(
                    transitions,
                    logger,
                    model_type,
                    bootstrap_model,
                    category,
                    discretized,
                    debug,
                    tensorboard,
                )

            if best_test_loss < overall_best_test_loss:
                overall_best_test_loss = best_test_loss
                overall_best_model = best_model

            if overall_best_test_loss < self.expected_optima:
                break
            else:
                logger.log("Failed to reach expected loss. This was attempt number %d" % ctr)

        curr_obs_actions = [(tr[0], tr[1]) for tr in transitions]
        curr_obs = [tr[0] for tr in transitions]
        next_obs = [tr[2] for tr in transitions]

        # Compute features for observations
        timestep_feature_calc_start = time.time()
        logger.debug("Calculating compositional features for clustering steps. Size of dataset: %d" % len(curr_obs_actions))

        # feature_fn = FeatureComputation(curr_obs_actions=curr_obs_actions,
        #                                 model=overall_best_model,
        #                                 batch_size=1024,
        #                                 discretized=discretized)
        # vectors = [feature_fn.calc_feature(dp_.next_obs) for dp_ in valid_dataset]

        feature_fn = CompositionalFeatureComputation(
            curr_obs_actions=curr_obs_actions,
            model=overall_best_model,
            batch_size=1024,
            discretized=discretized,
            compose=False,
        )

        curr_vectors = feature_fn.calc_feature(curr_obs)
        next_vectors = feature_fn.calc_feature(next_obs)

        vectors = list(curr_vectors)
        vectors.extend(next_vectors)

        logger.debug("Calculated features. Time taken %d sec" % (time.time() - timestep_feature_calc_start))

        # Call the clustering algorithm to generate clusters
        cluster_alg = GreedyClustering(threshold=self.clustering_threshold, dim=feature_fn.dim)
        cluster_centers = cluster_alg.cluster(vectors)
        logger.debug(
            "Number of clusters with L1 distance and threshold %f is %d" % (self.clustering_threshold, len(cluster_centers))
        )

        # Define the state abstraction model
        encoder_model = ClusteringModel(cluster_centers, feature_fn)

        debug_info = dict()

        state_to_abstract_map = CountConditionalProbability()
        abstract_to_state_map = CountConditionalProbability()

        logger.debug("Mapping datapoints to their center")
        timestep_center_assign_start = time.time()
        marked_transitions = []
        for tr, curr_vec_, next_vec_ in zip(transitions, curr_vectors, next_vectors):
            curr_abstract_state_ = encoder_model.encode_observations({"vec": curr_vec_})
            action_ = tr[1]
            next_abstract_state_ = encoder_model.encode_observations({"vec": next_vec_})
            marked_transitions.append((curr_abstract_state_, action_, next_abstract_state_))

            ####################
            if tr[3] in debug_info:
                debug_info[tr[3]].append((curr_abstract_state_, curr_vec_))
            else:
                debug_info[tr[3]] = [(curr_abstract_state_, curr_vec_)]

            if tr[4] in debug_info:
                debug_info[tr[4]].append((next_abstract_state_, next_vec_))
            else:
                debug_info[tr[4]] = [(next_abstract_state_, next_vec_)]

            state_to_abstract_map.add(curr_abstract_state_, tr[3])
            state_to_abstract_map.add(next_abstract_state_, tr[4])
            abstract_to_state_map.add(tr[3], curr_abstract_state_)
            abstract_to_state_map.add(tr[4], next_abstract_state_)
            ####################

        ######## TSNE PLOT #############
        # self._tsne(curr_vectors, next_vectors, transitions)

        seen_curr_state_action = set()
        chosen_indices = []
        for ix, tr in enumerate(transitions):
            if (tr[3], tr[1]) not in seen_curr_state_action:
                chosen_indices.append(ix)
                seen_curr_state_action.add((tr[3], tr[1]))
        chosen_indices = chosen_indices[:25]

        curr_state_actions = " ".join([str((transitions[i][3], transitions[i][1])) for i in chosen_indices])
        logs = []
        for real_state, state_vec_ls in debug_info.items():
            logger.debug("Real State %r " % (real_state,))
            logger.debug(" %r" % curr_state_actions)

            sampled_state_vec_ls = random.choices(state_vec_ls, k=100)
            max_intra_class_dispersion = 0
            mean_intra_class_dispersion = AverageUtil()

            for i, (abstract_state1, vec1) in enumerate(sampled_state_vec_ls):
                printable = " ".join(["%.2f" % vec1[i] for i in chosen_indices])
                logger.debug("%d -> %r" % (abstract_state1, printable))

                for abstract_state2, vec2 in sampled_state_vec_ls[i + 1 :]:
                    dispersion = np.mean(np.abs(vec1 - vec2))
                    max_intra_class_dispersion = max(max_intra_class_dispersion, dispersion)
                    mean_intra_class_dispersion.acc(dispersion)

            logs.append(
                "%r -> Max-Dispersion: %.4f, Mean-Dispersion: %s, Num occurrence %d"
                % (
                    real_state,
                    max_intra_class_dispersion,
                    mean_intra_class_dispersion,
                    len(state_vec_ls),
                )
            )

        for log in logs:
            logger.debug(log)

        logger.debug("Real state to Abstract state map")
        for (
            real_state,
            abstract_state_map,
        ) in state_to_abstract_map.get_conditions().items():
            logger.debug("Real state %r -> %s" % (real_state, abstract_state_map))

        logger.log("Abstract state to Real state map")
        for (
            abstract_state,
            real_state_map,
        ) in abstract_to_state_map.get_conditions().items():
            logger.debug("Abstract state %r -> %s" % (abstract_state, real_state_map))
        ################################

        logger.debug("Done computing in time %d sec " % (time.time() - timestep_center_assign_start))

        return (
            encoder_model,
            marked_transitions,
            {
                "loss": overall_best_test_loss,
                "success": overall_best_test_loss < self.expected_optima,
                "num_clusters": len(cluster_centers),
                "sa_map": state_to_abstract_map,
                "as_map": abstract_to_state_map,
            },
        )

    def _tsne(self, curr_vectors, next_vectors, transitions):
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        vectors = list(curr_vectors)
        vectors.extend(next_vectors)

        states = [tr[3] for tr in transitions]
        states.extend([tr[4] for tr in transitions])

        states_visited = set()
        for tr in transitions:
            states_visited.add(tr[3])
            states_visited.add(tr[4])

        states_visited = list(states_visited)

        vectors_batch = np.vstack(vectors)
        print("Vectors shape ", vectors_batch.shape)
        X_embedded = TSNE(n_components=2).fit_transform(vectors_batch)

        plt.figure(figsize=(6, 5))
        colors = "r", "g", "b", "c", "m", "y", "k", "w", "orange", "purple"

        for ix, state in enumerate(states_visited):
            X_, Y_ = [], []
            d = X_embedded.shape[0]

            for i in range(d):
                if states[i] == state:
                    X_.append(X_embedded[i, 0])
                    Y_.append(X_embedded[i, 1])

            X_ = np.vstack(X_)
            Y_ = np.vstack(Y_)

            plt.scatter(X_, Y_, c=colors[ix], label="%r" % (state,))

        plt.legend()
        plt.savefig("tsne.png")
