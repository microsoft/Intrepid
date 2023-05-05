import torch
import numpy as np
import torch.nn as nn


class RichIDHTKModel(nn.Module):
    """Take as input a sequence of observations y_{0:t} and a single observation y_{t+k} and predicts a vector
    of length k * action_dim"""

    def __init__(
        self,
        feature_type,
        obs_dim,
        state_dim,
        hat_f,
        hat_A_k_1,
        hat_A_k,
        hat_B,
        hat_K,
        hat_M_k,
    ):
        """
        :param feature_type: Type of feature
        :param obs_dim       observation dimension
        :param state_dim:    state dimension
        :param hat_f:        a function that maps t+1 observations to an observation
        :param hat_A_k_1:    estimate of A matrix to the power k - 1
        :param hat_A_k:      estimate of A matrix to the power k
        :param hat_B:        estimate of B matrix
        :param hat_K:        estimate of K matrix
        :param hat_M_k:      \hat{M}_k matrix
        """
        super(RichIDHTKModel, self).__init__()

        self.feature_type = feature_type
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.hat_f = hat_f
        self.hat_A_k_1 = hat_A_k_1
        self.hat_A_k = hat_A_k
        self.hat_B = hat_B
        self.hat_K = hat_K
        self.hat_M_k = hat_M_k

        if self.feature_type == "feature":
            self.h_model = nn.Sequential(
                nn.Linear(obs_dim, state_dim),
                nn.LeakyReLU(),
                nn.Linear(state_dim, state_dim),
            )

            self.m_layer = nn.Linear(state_dim, self.state_dim)

        elif self.feature_type == "image":
            # TODO fix channel size below
            self.h_model = nn.Sequential(
                nn.Conv2d(6, 16, 8, 4), nn.LeakyReLU(), nn.Conv2d(16, 16, 8, 2)
            )

            self.m_layer = nn.Linear(16, self.state_dim)  # TODO 16 is hard-coded
        else:
            raise AssertionError("Unhandled feature_type %r" % self.feature_type)

        self.abk = torch.matmul(self.hat_A_k_1, torch.matmul(self.hat_B, self.hat_K))

        if torch.cuda.is_available():
            self.cuda()

    def get_h_val(self, obs):
        batch_size = obs.size(0)
        x = self.h_model(obs)

        x = x.view(batch_size, -1)

        return self.m_layer(x)

    def forward(self, curr_obs_seq, k_step_obs):
        """
        obs_shape below can be a tuple
        :param curr_obs_seq: batch x t x obs_shape
        :param k_step_obs:   batch x obs_shape
        :return:
        """

        y_t = curr_obs_seq[:, -1, :]  # batch x obs_shape

        embed_y_t_k = self.get_h_val(k_step_obs)  # h(y_{t+k}) of size batch x state_dim
        embed_y_t = self.get_h_val(y_t)  # h(y_t) of size batch x state_dim
        embed_y_0_t = self.hat_f(
            curr_obs_seq
        ).detach()  # \hat{f}(y_{0:t}) of size batch x state_dim

        out = (
            embed_y_t_k
            - torch.matmul(embed_y_t, torch.transpose(self.hat_A_k, 0, 1))
            - torch.matmul(embed_y_0_t, torch.transpose(self.abk, 0, 1))
        )  # batch x state_dim

        out = torch.matmul(out, torch.transpose(self.hat_M_k, 0, 1))

        return out


class RichIDHTModel(nn.Module):
    """Take as input a sequence of observations y_{0:t} and a single observation y_{t+k} and predicts a vector
    of length k * action_dim"""

    def __init__(
        self, feature_type, obs_dim, state_dim, hat_f, hat_A, hat_B, hat_K, hat_M
    ):
        """
        :param feature_type:  Type of feature
        :param obs_dim        observation dimension
        :param state_dim:     state dimension
        :param hat_f:         a function that maps t+1 observations to an observation
        :param hat_A:         estimate of A matrix
        :param hat_B:         estimate of B matrix
        :param hat_K:         estimate of K matrix
        :param hat_M:         \hat{M} matrix
        """
        super(RichIDHTModel, self).__init__()

        self.feature_type = feature_type
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.hat_f = hat_f
        self.hat_A = hat_A
        self.hat_B = hat_B
        self.hat_K = hat_K
        self.hat_M = hat_M

        if self.feature_type == "feature":
            self.h_model = nn.Sequential(
                nn.Linear(obs_dim, self.state_dim),
                nn.LeakyReLU(),
                nn.Linear(self.state_dim, self.state_dim),
            )

            self.m_layer = nn.Linear(self.state_dim, self.state_dim)

        elif self.feature_type == "image":
            # TODO fix channel size below
            self.h_model = nn.Sequential(
                nn.Conv2d(6, 16, 8, 4), nn.LeakyReLU(), nn.Conv2d(16, 16, 8, 2)
            )

            self.m_layer = nn.Linear(16, self.state_dim)  # TODO 16 is hard-coded
        else:
            raise AssertionError("Unhandled feature_type %r" % self.feature_type)

        self.bk = torch.matmul(self.hat_B, self.hat_K)

        if torch.cuda.is_available():
            self.cuda()

    def get_h_val(self, obs):
        batch_size = obs.size(0)
        x = self.h_model(obs)

        x = x.view(batch_size, -1)

        return self.m_layer(x)

    def forward(self, curr_obs_seq, next_obs):
        """
        :param curr_obs_seq: Denote y_{0:t}
        :param next_obs: Denote y_{t+1}
        :return:
        """

        y_t = curr_obs_seq[:, -1, :]

        embed_y_t_1 = self.get_h_val(next_obs)  # h(y_{t+1}) of size batch x state_dim
        embed_y_t = self.get_h_val(y_t)  # h(y_t) of size batch x state_dim
        embed_y_0_t = self.hat_f(
            curr_obs_seq
        ).detach()  # \hat{f}(y_{0:t}) of size batch x state_dim

        term1 = embed_y_t_1
        term2 = torch.matmul(embed_y_t, self.hat_A.transpose(0, 1))
        term3 = torch.matmul(embed_y_0_t, self.bk.transpose(0, 1))

        out = term1 - term2 - term3
        out = torch.matmul(out, torch.transpose(self.hat_M, 0, 1))

        return out


class RichIDFModel(nn.Module):
    """Take as input a sequence of observations y_{0:t} and a single observation y_{t+k} and predicts a vector
    of length k * action_dim"""

    def __init__(self, t, state_dim, A, prev_f_model, h_t_model):
        """
        :param k:           an integer denoting the number of actions you predict
        :param state_dim:     observation dimension
        :param act_dim:     action dimension
        :param hat_f:       a function that maps t+1 observations to an observation
        :param hat_A_k_1:   estimate of A matrix to the power k - 1
        :param hat_A_k:     estimate of A matrix to the power k
        :param hat_B:       estimate of B matrix
        :param hat_K:       estimate of K matrix
        :param hat_M:       \hat{M} matrix
        """
        super(RichIDFModel, self).__init__()

        self.t = t
        self.At = A.transpose(0, 1)
        self.h_t_model = h_t_model
        self.state_dim = state_dim
        self.prev_f_model = prev_f_model

    def forward(self, obs_seq):
        """
        :param obs_seq: Denote y_{0:t}
        :return:
        """

        assert obs_seq.size(1) == self.t + 1, "Found %d and obs_seq size of %d" % (
            self.t + 1,
            obs_seq.size(1),
        )

        if self.t == 0:
            return torch.zeros(self.state_dim).float()
        else:
            last_obs = obs_seq[:, -1, :]  # y_t
            snd_last_obs = obs_seq[:, -2, :]  # y_{t-1}
            prev_obs_seq = obs_seq[:, :-1, :]  # y_{0: t-1}

            term1 = self.h_t_model.get_h_val(last_obs)
            term2 = torch.matmul(self.h_t_model.get_h_val(snd_last_obs), self.At)
            term3 = torch.matmul(self.prev_f_model(prev_obs_seq), self.At)

            # if obs_seq.size(1) > 1:
            #     import pdb
            #     pdb.set_trace()

            return (term1 - term2 + term3).detach()


class RichIDPolicy(nn.Module):
    """Take as input a sequence of observations y_{0:t} and a single observation y_{t+k} and predicts a vector
    of length k * action_dim"""

    def __init__(self, K, f_model, add_noise=False):
        """
        :param k:           an integer denoting the number of actions you predict
        :param state_dim:     observation dimension
        :param act_dim:     action dimension
        :param hat_f:       a function that maps t+1 observations to an observation
        :param hat_A_k_1:   estimate of A matrix to the power k - 1
        :param hat_A_k:     estimate of A matrix to the power k
        :param hat_B:       estimate of B matrix
        :param K:       estimate of K matrix
        :param hat_M:       \hat{M} matrix
        """
        super(RichIDPolicy, self).__init__()

        self.Kt = K.transpose(0, 1)
        self.f_model = f_model
        self.add_noise = add_noise

    def forward(self, obs_seq):
        """
        :param obs_seq: Denote y_{0:t}
        :return:
        """

        if self.add_noise:
            raise AssertionError("Unhandled")
        else:
            return torch.matmul(self.f_model(obs_seq), self.Kt).detach().numpy()[0]


class RichIDPsiModel(nn.Module):
    """Take as input two observation and action and output a sequence of k actions"""

    def __init__(self, k, obs_dim, act_dim):
        super(RichIDPsiModel, self).__init__()

        self.k = k

        assert self.k > 0, "Nothing to predict"

        self.input_dim = 2 * obs_dim + act_dim
        self.output_dim = self.k * act_dim
        self.hidden_dim = 56

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, curr_obs, new_obs, action):
        x = torch.cat(
            [curr_obs, new_obs, action], dim=1
        )  # Batch x (2 x Obs_dim + Act_dim)
        out = self.network(x)

        return out
