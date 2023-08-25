import time
import torch
import random
import numpy as np

from utils.cuda import cuda_var


class FeatureComputation:
    def __init__(self, curr_obs_actions, model, batch_size, discretized):
        self.curr_obs_actions = curr_obs_actions
        self.model = model
        self.batch_size = batch_size
        self.discretized = discretized
        self.dim = len(self.curr_obs_actions)

    def calc_feature(self, observation):
        next_obs_np = torch.from_numpy(np.array(observation)).view(1, -1)

        vec = []
        for i in range(0, self.dim, self.batch_size):
            this_batch_size = min(self.dim - i, self.batch_size)

            prev_observations = cuda_var(
                torch.cat(
                    [
                        torch.from_numpy(np.array(curr_obs_action_[0])).view(1, -1)
                        for curr_obs_action_ in self.curr_obs_actions[i : i + this_batch_size]
                    ],
                    dim=0,
                )
            ).float()

            actions = cuda_var(
                torch.cat(
                    [
                        torch.from_numpy(np.array(curr_obs_action_[1])).view(1, -1)
                        for curr_obs_action_ in self.curr_obs_actions[i : i + this_batch_size]
                    ],
                    dim=0,
                )
            ).long()

            observations = cuda_var(torch.cat([next_obs_np] * this_batch_size, dim=0)).float()

            prob, _ = self.model.gen_prob(
                prev_observations=prev_observations,
                actions=actions,
                observations=observations,
                discretized=self.discretized,
            )
            batch_vec = prob.cpu().data.numpy().flatten()
            vec.append(batch_vec)

        vec = np.concatenate(vec, axis=0)

        return vec


class CompositionalFeatureComputation:
    def __init__(self, curr_obs_actions, model, batch_size, discretized, compose=True):
        self.curr_obs_actions = curr_obs_actions
        self.model = model
        self.batch_size = batch_size
        self.discretized = discretized
        self.dim = len(self.curr_obs_actions)
        self.compose = compose

        # Encode previous and current actions
        time_obs_act_start = time.time()
        self.prev_obs_act_encoded_list = []

        for i in range(0, self.dim, self.batch_size):
            this_batch_size = min(self.dim - i, self.batch_size)

            prev_observations = cuda_var(
                torch.cat(
                    [
                        torch.from_numpy(np.array(curr_obs_action_[0])).view(1, -1)
                        for curr_obs_action_ in self.curr_obs_actions[i : i + this_batch_size]
                    ],
                    dim=0,
                )
            ).float()

            actions = cuda_var(
                torch.cat(
                    [
                        torch.from_numpy(np.array(curr_obs_action_[1])).view(1, -1)
                        for curr_obs_action_ in self.curr_obs_actions[i : i + this_batch_size]
                    ],
                    dim=0,
                )
            ).long()

            prev_obs_act_encoded = self.model.encode_prev_obs_action(
                prev_observations=prev_observations, actions=actions
            ).detach()

            self.prev_obs_act_encoded_list.append(prev_obs_act_encoded)

        print(
            "Time taken to encode previous action and observation is ",
            time.time() - time_obs_act_start,
        )

    def calc_feature(self, observations):
        if self.compose:
            return self._calc_feature_compose(observations)
        else:
            return self._calc_feature_generic(observations).round()

    def _calc_feature_compose(self, observations):
        if not isinstance(observations, list):
            observations = [observations]

        # Encode observations
        num_obs = len(observations)
        obs_encoded_list = []

        for i in range(0, num_obs, self.batch_size):
            this_batch_size = min(num_obs - i, self.batch_size)

            obs_batch = cuda_var(
                torch.cat(
                    [torch.from_numpy(np.array(obs_)).view(1, -1) for obs_ in observations[i : i + this_batch_size]],
                    dim=0,
                )
            ).float()

            obs_encoded = self.model.encode_curr_obs(obs_batch).cpu().data.numpy()
            obs_encoded_list.append(obs_encoded)

        all_vectors = np.concatenate(obs_encoded_list, axis=0)

        return all_vectors

    def _calc_feature_generic(self, observations):
        if not isinstance(observations, list):
            observations = [observations]

        # Encode observations
        num_obs = len(observations)
        obs_encoded_list = []

        for i in range(0, num_obs, self.batch_size):
            this_batch_size = min(num_obs - i, self.batch_size)

            obs_batch = cuda_var(
                torch.cat(
                    [torch.from_numpy(np.array(obs_)).view(1, -1) for obs_ in observations[i : i + this_batch_size]],
                    dim=0,
                )
            ).float()

            obs_encoded = self.model.encode_curr_obs(obs_batch)
            obs_encoded_list.append(obs_encoded)

        all_vectors = []

        for obs_encoded in obs_encoded_list:
            prob_batch = []

            for prev_obs_act_encoded in self.prev_obs_act_encoded_list:
                # Returns prob which is equal to number of obs and action
                prob, _ = self.model.gen_batch_prob_from_encodings(prev_obs_act_encoded, obs_encoded)

                prob = prob.cpu().data.numpy()
                prob_batch.append(prob)

            batch_vector = np.concatenate(prob_batch, axis=1)

            all_vectors.append(batch_vector)  # Would be a list of type batch x number of obs and action

        all_vectors = np.concatenate(all_vectors, axis=0)

        return all_vectors


class GreedyClustering:
    """Given a set of vectors, repeatedly pick a vector and cover elements within a certain distance and remove them
    from the set. Keep doing until the set is empty (i.e., all covered). The chosen points are the center.
    """

    def __init__(self, threshold, dim):
        self.threshold = threshold
        self.dim = dim
        assert dim > 0, "Dimension of feature space cannot be 0"

    def cluster(self, vectors):
        vectors = list(vectors)  # Work on the copy
        cluster_centers = []  # Ordering of center is important

        while vectors:
            n = len(vectors)  # Number of vectors remaining
            center = vectors[random.randint(0, n - 1)]  # Pick a random point in the set
            cluster_centers.append(center)

            # Find points covered by this center and remove them from the set
            new_vectors = [point for point in vectors if np.mean(np.abs(point - center)) > self.threshold]
            vectors = new_vectors

        return cluster_centers


class ClusteringModel:
    def __init__(self, cluster_centers, feature_fn):
        self.cluster_centers = cluster_centers
        self.feature_fn = feature_fn

    def encode_observations(self, observation):
        """
        :param observation:  Either an observation or the embedding of the observation in the feature space. This
                            embedding is provided as a dictionary {"vec": vector}. As the feature computation is an
                            time expensive process proving the embeddings for previously seen observation is beneficial.
        :return: An integer denoting the cluster ID
        """

        if not isinstance(observation, dict) or "vec" not in observation:
            vec = self.feature_fn.calc_feature(observation)
        else:
            vec = observation["vec"]

        chosen_cluster, min_distance = -1, float("inf")

        for i, cluster in enumerate(self.cluster_centers):
            dist = np.mean(np.abs(vec - cluster))
            if dist < min_distance:
                chosen_cluster = i
                min_distance = dist

        return chosen_cluster

    def save(self, model_folder_name, file_name):
        print("Warning saving functionality for cluster model not implemented.")
