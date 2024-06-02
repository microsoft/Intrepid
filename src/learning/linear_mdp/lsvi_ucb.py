import torch

from learning.datastructures.elliptic_potential import EllipticPotential


class LSVIUCB:
    """
    LSVI UCB algorithm of Chi Jin et al.
    https://arxiv.org/pdf/1907.05388.pdf
    """

    def __init__(self, exp_setup):

        self.horizon = exp_setup.config["horizon"]

        self.max_episodes = exp_setup.constants["max_episodes"]
        self.batch_size = exp_setup.constants["batch_size"]
        self.lam = exp_setup.constants["elliptic_potential_lam"]

    def _get_all_features(self, obs, encoder):

        features = torch.cat([encoder.encode(obs, action).unsqueeze(1) for action in self.actions], dim=1)  # dim x num_actions

        return features

    def _get_vals(self, features, weight, elliptic_potential, v_max):
        """
        :param features:    Tensor of size dim x actions denoting {\phi(x, a1), ..., \phi(x, aK)}
                            for all actions and a given observation
        :param weight:      Weight vector for representing the ERM solution
        :param elliptic_potential:  Elliptic potential datastructure to compute elliptic potential
        :param v_max:       Maximum value of a Q-function
        :return:
        """

        bonus = elliptic_potential.get_elliptic_bonus(features.T)  # actions
        q_vals = (weight.T @ features).view(-1) + self.beta * bonus  # actions

        if v_max is not None:
            q_vals = torch.minimum(q_vals, torch.ones_like(q_vals) * v_max)

        return q_vals

    def do_train(self, env, encoder):
        """
        :param env:         Environment to solve
        :param encoder:     Encoder that encodes a given state and action to generate features
        :return:            A policy and metric
        """

        dataset = []

        elliptic_bonuses = dict()
        for h in range(0, self.horizon):
            elliptic_bonuses[h] = EllipticPotential(lam=self.lam)

        # Weights for each time step to denote the ERM for value function
        weights = dict()

        # Assuming reward is in [0, 1], Maximum V_max from time step h is H - h + 1
        v_maxes = {h: env.horizon - h for h in range(env.horizon + 1)}

        for eps in range(0, self.max_episodes, self.batch_size):

            for h in range(self.horizon - 1, -1, -1):
                inv_mat_h = elliptic_bonuses[h].get_inv_mat_det()

                feature_val = 0.0
                for dp in dataset:

                    obs, features, action, reward, next_obs = dp

                    if h == self.horizon - 1:
                        next_q_val = 0.0
                    else:
                        next_q_val = self._get_vals(next_obs, encoder, weights[h + 1], elliptic_bonuses[h + 1], v_maxes[h + 1])
                    feature_val += features[action] * (reward + next_q_val)

                weights[h] = inv_mat_h @ feature_val

            # Generate a batch of episode. In the original Jin et al., you have self.batch_size=1
            for _ in range(self.batch_size):

                total_return = 0.0
                obs, info = env.reset()

                for h in range(self.horizon):

                    # Get Q_h(x, .) vals
                    features = self._get_all_features(obs, encoder)
                    q_vals = self._get_vals(features, weights[h], elliptic_bonuses[h], v_maxes[h])  # num_actions

                    a = q_vals.argmax()
                    next_obs, r, done, info = env.step(a)
                    obs = next_obs
                    total_return += r

            # TODO: return average of these policies
