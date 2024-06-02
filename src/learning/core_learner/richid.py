import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from learning.datastructures.episode import Episode
from learning.learning_utils.ricatti_solver import RicattiSolver
from model.misc.richid_decoder import (
    RichIDHTKModel,
    RichIDFModel,
    RichIDPolicy,
    RichIDHTModel,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from utils.cuda import cuda_var


class RichId:
    """
    Implements Phase 3 of
            Learning the Linear Quadratic Regulator from Nonlinear Observations
            Zakaria Mhammedi, Dylan J. Foster, Max Simchowitz, Dipendra Misra, Wen Sun, Akshay Krishnamurthy,
            Alexander Rakhlin, John Langford, NeurIPS 2020
    """

    def __init__(self, exp_setup):
        self.config = exp_setup.config
        self.constants = exp_setup.constants
        self.logger = exp_setup.logger

        self.obs_dim = self.config["obs_dim"]
        self.state_dim = self.config["state_dim"]
        self.action_dim = self.config["world_dim"]
        self.max_iter = self.constants["max_iter"]
        self.horizon = self.config["horizon"]
        self.n = self.constants["samples"]
        self.batch_size = self.constants["batch_size"]
        self.max_epoch = self.constants["max_epoch"]
        self.grad_clip = self.constants["grad_clip"]

        self.kappa = 3 * self.state_dim
        self.sigma = 1.0
        self.sigma2 = self.sigma * self.sigma
        self.universal_early_stop_error = 0.01

        self.ricatti_solver = RicattiSolver(logger=exp_setup.logger)

    def _gather_datapoint(self, env, policy, t):
        """Gather a single datapoint by roll-in with policy till time step t and roll-out with random noise"""

        obs, info = env.reset()

        obs_seq = [torch.from_numpy(obs).unsqueeze(0)]

        episode = Episode(observation=obs, state=info["state"])

        for t_ in range(0, t + 1):
            obs_seq_var = torch.cat(obs_seq, dim=0).unsqueeze(0).float()  # 1 x (t_ + 1) x obs_shape

            action = policy[t_](obs_seq_var)
            noise = np.random.normal(0, self.sigma, self.action_dim)
            perturbed_action = action + noise

            obs, reward, done, info = env.step(perturbed_action)

            obs_seq.append(torch.from_numpy(obs).unsqueeze(0))

            episode.add(
                action=(action, noise),
                reward=reward,
                new_obs=obs,
                new_state=info["state"],
            )

        for _ in range(0, self.kappa):
            noisy_action = np.random.normal(0, self.sigma, self.action_dim)

            obs, reward, done, info = env.step(noisy_action)
            episode.add(action=noisy_action, reward=reward, new_obs=obs, new_state=info["state"])

        episode.terminate()

        return episode

    def _update_phi_model(self, t, k, phi_model, dataset):
        optimizer = torch.optim.Adam(params=phi_model.parameters(), lr=self.constants["learning_rate"])

        phi_dataset = []

        for dp in dataset:
            observations = dp.get_observations()
            actions = dp.get_actions()

            curr_obs_seq = (
                torch.cat(
                    [torch.from_numpy(observations[t_]).unsqueeze(0) for t_ in range(0, t + 1)],
                    dim=0,
                )
                .unsqueeze(0)
                .float()
            )  # y{0:t}
            k_step_obs = torch.from_numpy(observations[t + k]).unsqueeze(0).float()  # y_{t+k}

            noisy_actions = [actions[t][1]] + actions[t + 1 : t + k]  # Actions from time t to t + k - 1
            noisy_actions = torch.cat(
                [torch.from_numpy(action_).view(1, -1) for action_ in noisy_actions],
                dim=1,
            ).float()

            phi_dataset.append((curr_obs_seq, k_step_obs, noisy_actions))

        dataset_size = len(phi_dataset)

        for epoch in range(0, self.max_epoch):
            sum_loss = 0.0
            num_examples = 0

            for i in range(0, dataset_size, self.batch_size):
                curr_obs_seq = cuda_var(torch.cat([dp[0] for dp in phi_dataset[i : i + self.batch_size]], dim=0).float())

                k_step_obs = cuda_var(torch.cat([dp[1] for dp in phi_dataset[i : i + self.batch_size]], dim=0).float())

                noisy_actions = cuda_var(
                    torch.cat([dp[2] for dp in phi_dataset[i : i + self.batch_size]], dim=0).float()
                ).detach()

                phi_prediction = phi_model(curr_obs_seq, k_step_obs)
                loss = ((phi_prediction - noisy_actions) ** 2).sum(1).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(phi_model.parameters(), self.grad_clip)
                optimizer.step()

                sum_loss += float(loss) * int(curr_obs_seq.size(0))
                num_examples += int(curr_obs_seq.size(0))

            avg_loss = sum_loss / float(max(1, num_examples))

            self.logger.log("Phi-Model Training (t=%d, k=%d): Epoch %d, Loss %f" % (t, k, epoch, avg_loss))

            if avg_loss < self.universal_early_stop_error:
                break

    def _update_psi_model(self, t, psi_model, phi_models, dataset):
        optimizer = torch.optim.Adam(params=psi_model.parameters(), lr=self.constants["learning_rate"])

        psi_dataset = []

        for dp in dataset:
            observations = dp.get_observations()
            curr_obs_seq = (
                torch.cat(
                    [torch.from_numpy(observations[t_]).unsqueeze(0) for t_ in range(0, t + 1)],
                    dim=0,
                )
                .unsqueeze(0)
                .float()
            )  # y_{0:t}

            next_obs = dict()
            for k in range(1, self.kappa + 1):
                next_obs[k] = torch.from_numpy(observations[t + k]).unsqueeze(0).float()  # y_{t+k}

            psi_dataset.append((curr_obs_seq, next_obs))

        dataset_size = len(psi_dataset)

        for epoch in range(0, self.max_epoch):
            sum_loss = 0.0
            num_examples = 0

            for i in range(0, dataset_size, self.batch_size):
                curr_obs_seq = cuda_var(torch.cat([dp[0] for dp in psi_dataset[i : i + self.batch_size]], dim=0).float())

                next_obs = dict()
                for k in range(1, self.kappa + 1):
                    next_obs[k] = cuda_var(
                        torch.cat(
                            [dp[1][k] for dp in psi_dataset[i : i + self.batch_size]],
                            dim=0,
                        ).float()
                    )

                predictions = []
                for k in range(1, self.kappa + 1):
                    prediction_k = phi_models[(t, k)](curr_obs_seq, next_obs[k])
                    predictions.append(prediction_k)

                predictions = torch.cat(predictions, dim=1).detach()

                phi_prediction = psi_model(curr_obs_seq, next_obs[1])
                loss = ((phi_prediction - predictions) ** 2).sum(1).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(psi_model.parameters(), self.grad_clip)
                optimizer.step()

                sum_loss += float(loss) * int(curr_obs_seq.size(0))
                num_examples += int(curr_obs_seq.size(0))

            avg_loss = sum_loss / float(max(1, num_examples))

            self.logger.log("Psi-Model Training (t=%d): Epoch %d, Loss %f" % (t, epoch, avg_loss))

            if avg_loss < self.universal_early_stop_error:
                break

    def _eval_model(self, t, f_models, dataset):
        state_pred_dataset = []

        for dp in dataset:
            observations = dp.get_observations()
            states = dp.get_states()
            curr_obs_seq = (
                torch.cat(
                    [torch.from_numpy(observations[t_]).unsqueeze(0) for t_ in range(0, t + 1)],
                    dim=0,
                )
                .unsqueeze(0)
                .float()
            )  # y_{0:t}

            state_var = torch.from_numpy(states[t + 1]).unsqueeze(0).float()

            state_pred_dataset.append((curr_obs_seq, state_var))

        dataset_size = len(state_pred_dataset)

        sum_loss = 0.0
        num_examples = 0

        for i in range(0, dataset_size, self.batch_size):
            curr_obs_seq = cuda_var(torch.cat([dp[0] for dp in state_pred_dataset[i : i + self.batch_size]], dim=0).float())
            gold_state = cuda_var(torch.cat([dp[1] for dp in state_pred_dataset[i : i + self.batch_size]], dim=0).float())

            predicted_state = f_models[t](curr_obs_seq)
            loss = ((gold_state - predicted_state) ** 2).sum(1).mean()

            sum_loss += float(loss) * int(gold_state.size(0))
            num_examples += int(gold_state.size(0))

        avg_loss = sum_loss / float(max(1, num_examples))

        self.logger.log("Time step %d: State Abstraction Mean Squared Loss on %d examples is %f" % (t, num_examples, avg_loss))

    def train(self, env, latent_lqr):
        A = latent_lqr.A
        B = latent_lqr.B
        Q = latent_lqr.Q
        R = latent_lqr.R
        Sigma_w = latent_lqr.Sigma_W

        _, K = self.ricatti_solver.solve(A, B, Q, R)

        powers_of_A = {0: np.eye(self.state_dim)}
        c_matrices = dict()
        m_matrices = dict()

        sigma_term = 0
        for k in range(1, self.kappa + 1):
            if k == 1:
                c_matrices[k] = B
            else:
                c_matrices[k] = np.concatenate([powers_of_A[k - 1] @ B, c_matrices[k - 1]], axis=1)

            sigma_term += powers_of_A[k - 1] @ Sigma_w @ powers_of_A[k - 1].T
            inner_term = c_matrices[k] @ c_matrices[k].T + (1.0 / self.sigma2) * sigma_term
            inv_matrix = np.linalg.inv(inner_term)  # TODO fix for non-invertible matrices
            m_matrices[k] = c_matrices[k].T @ inv_matrix
            powers_of_A[k] = powers_of_A[k - 1] @ A

        m_matrix = np.concatenate(
            [np.matmul(m_matrices[k], powers_of_A[k - 1]).transpose(1, 0) for k in range(1, self.kappa + 1)],
            axis=1,
        ).transpose(1, 0)

        # Convert to pytorch
        A = cuda_var(torch.from_numpy(A).float())
        B = cuda_var(torch.from_numpy(B).float())
        K = cuda_var(torch.from_numpy(K).float())
        m_matrix = cuda_var(torch.from_numpy(m_matrix).float())
        powers_of_A[0] = cuda_var(torch.from_numpy(powers_of_A[0]).float())

        for k in range(1, self.kappa + 1):
            c_matrices[k] = cuda_var(torch.from_numpy(c_matrices[k]).float())
            m_matrices[k] = cuda_var(torch.from_numpy(m_matrices[k]).float())
            powers_of_A[k] = cuda_var(torch.from_numpy(powers_of_A[k]).float())

        h_t_k_models = dict()
        h_t_models = dict()
        f_models = {0: RichIDFModel(0, self.state_dim, A=A, prev_f_model=None, h_t_model=None)}
        policy = {0: RichIDPolicy(K=K, f_model=f_models[0])}

        for t in range(0, self.horizon):
            self.logger.log("Beginning time step %d of horizon=%d" % (t, self.horizon))

            # Collect 2 * n_{op} trajectories by taking perturbed actions taken by the policy upto time step t
            # and then taking random actions for next kappa steps

            dataset = [self._gather_datapoint(env, policy, t) for _ in range(0, 3 * self.n)]

            for k in range(1, self.kappa + 1):
                h_t_k_model = RichIDHTKModel(
                    feature_type=self.config["feature_type"],
                    obs_dim=self.obs_dim,
                    state_dim=self.state_dim,
                    hat_f=f_models[t],
                    hat_A_k_1=powers_of_A[k - 1],
                    hat_A_k=powers_of_A[k],
                    hat_B=B,
                    hat_K=K,
                    hat_M_k=m_matrices[k],
                )

                self._update_phi_model(t, k, h_t_k_model, dataset[: self.n])

                h_t_k_models[(t, k)] = h_t_k_model

            # Train the h_t models

            h_t_model_ = RichIDHTModel(
                feature_type=self.config["feature_type"],
                obs_dim=self.obs_dim,
                state_dim=self.state_dim,
                hat_f=f_models[t],
                hat_A=A,
                hat_B=B,
                hat_K=K,
                hat_M=m_matrix,
            )

            self._update_psi_model(t, h_t_model_, h_t_k_models, dataset[self.n : 2 * self.n])

            h_t_models[t] = h_t_model_

            f_models[t + 1] = RichIDFModel(
                t=t + 1,
                state_dim=self.state_dim,
                A=A,
                prev_f_model=f_models[t],
                h_t_model=h_t_models[t],
            )

            policy[t + 1] = RichIDPolicy(K=K, f_model=f_models[t + 1])

            self._eval_model(t, f_models, dataset[2 * self.n :])


class MyRichIDHModel(nn.Module):
    """Take as input two observation and action and output a sequence of k actions"""

    def __init__(self, k, obs_dim, act_dim):
        super(MyRichIDHModel, self).__init__()

        assert k > 0, "Nothing to predict"

        self.output_dim = k * act_dim
        self.obs_dim = obs_dim

        if isinstance(obs_dim, int):
            self.hidden_dim = 56
            self.network = nn.Sequential(
                nn.Linear(obs_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )

        elif isinstance(obs_dim, list):
            _, _, channel = obs_dim

            self.network = nn.Sequential(nn.Conv2d(channel, 16, 8, 4), nn.LeakyReLU(), nn.Conv2d(16, 16, 8, 2))

            self.hidden_dim = 16
            self.layer = nn.Linear(self.hidden_dim, self.output_dim)

        else:
            raise AssertionError("Unhandled obs_dim")

    def forward(self, curr_obs):
        if isinstance(self.obs_dim, int):
            out = self.network(curr_obs)
            return out

        elif isinstance(self.obs_dim, list):
            height, width, channel = self.obs_dim
            batch_size = curr_obs.size(0)

            curr_obs = curr_obs.view(batch_size, channel, height, width)
            out = self.network(curr_obs).view(batch_size, self.hidden_dim)
            out = self.layer(out)

            return out
        else:
            raise AssertionError("Unhandled obs_dim")


class RichIDFModel_Deprecated(nn.Module):
    """Take as input two observation and action and output a sequence of k actions"""

    def __init__(self, h_model, v_matrix):
        super(RichIDFModel_Deprecated, self).__init__()

        self.h_model = h_model  # Maps self.obs_dim  -> (k * self.act_dim)
        self.v_matrix = v_matrix  # (k * self.act_dim) x self.state_dim

    def forward(self, curr_obs):
        h_out = self.h_model(curr_obs)
        out = torch.matmul(h_out, self.v_matrix)

        return out


class SysID:
    """
    Implements Phase 2 of RichID algorithm
            Learning the Linear Quadratic Regulator from Nonlinear Observations
            Zakaria Mhammedi, Dylan J. Foster, Max Simchowitz, Dipendra Misra, Wen Sun, Akshay Krishnamurthy,
            Alexander Rakhlin, John Langford, NeurIPS 2020
    """

    def __init__(self, exp_setup):
        self.config = exp_setup.config
        self.constants = exp_setup.constants

        self.state_dim = self.config["state_dim"]
        self.act_dim = self.config["act_dim"]
        self.obs_dim = self.config["obs_dim"]
        self.logger = exp_setup.logger

        self.num_samples = self.constants["samples"]
        self.batch_size = self.constants["batch_size"]
        self.max_epoch = self.constants["max_epoch"]
        self.learning_rate = self.constants["learning_rate"]
        self.k0 = self.constants["k0"]

    def _train_hmodel(self, dataset, k0, k):
        h_model = MyRichIDHModel(k=k, obs_dim=self.obs_dim, act_dim=self.act_dim)

        k1 = k0 + k
        fmodel_dataset = []

        for dp in dataset:
            obs = dp.get_observations()[k1]
            action = np.concatenate(dp.get_actions()[k0:k1])  # Ensure edge cases are handled
            fmodel_dataset.append((obs, action))

        random.shuffle(fmodel_dataset)
        dataset_size = len(dataset)
        batches = [fmodel_dataset[i : i + self.batch_size] for i in range(0, dataset_size, self.batch_size)]

        optimizer = optim.Adam(params=h_model.parameters(), lr=self.learning_rate)

        for epoch in range(1, self.max_epoch + 1):
            for it, batch in enumerate(batches):
                obs_batch = cuda_var(torch.cat([torch.from_numpy(pt[0]).view(1, -1) for pt in batch])).float()
                gold_act_seq = cuda_var(torch.cat([torch.from_numpy(pt[1]).view(1, -1) for pt in batch])).float()

                pred_act_seq = h_model(obs_batch)
                loss = torch.mean((gold_act_seq - pred_act_seq) ** 2)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(h_model.parameters(), 40)
                optimizer.step()

                loss = float(loss)
                self.logger.log("H-Model: Epoch %d, Iteration %d, Loss %r" % (epoch, it + 1, loss))

        return h_model

    def _train_v_matrix(self, dataset, h_model, k1):
        matrix = None

        for dp in dataset:
            obs = dp.get_observations()[k1]
            obs_var = cuda_var(torch.from_numpy(obs).view(1, -1)).float()
            encoding = h_model(obs_var).detach().view(-1, 1)
            encoding_tranpose = torch.transpose(encoding, 0, 1)

            if matrix is None:
                matrix = torch.matmul(encoding, encoding_tranpose)
            else:
                matrix += torch.matmul(encoding, encoding_tranpose)

        matrix /= float(max(1, len(dataset)))

        matrix = matrix.data.cpu().numpy()  # (k self.act_dim) x (k self.act_dim)

        # Perform PCA
        pca = PCA(n_components=self.state_dim)
        pca.fit(matrix)

        # print("PCA explained variance: ", pca.explained_variance_ratio_)
        orthogonal_basis = cuda_var(torch.from_numpy(pca.components_)).float()  # (self.state_dim) x (k self.act_dim)
        orthogonal_basis = torch.transpose(orthogonal_basis, 0, 1)  # (k self.act_dim) x (self.state_dim)

        return orthogonal_basis

    def _train_a_b(self, dataset, f_model, k1, S1=None):
        x_batch = []
        y_batch = []

        for dp in dataset:
            obs = dp.get_observations()[k1]
            action = dp.get_actions()[k1]
            next_obs = dp.get_observations()[k1 + 1]

            curr_obs_batch = cuda_var(torch.from_numpy(obs).view(1, -1)).float()
            curr_encoding = f_model(curr_obs_batch).view(-1).data.cpu().numpy()
            if S1 is not None:
                curr_encoding = np.matmul(S1, curr_encoding)
            # curr_encoding = curr_obs_batch.view(-1).data.cpu().numpy()

            next_obs_batch = cuda_var(torch.from_numpy(next_obs).view(1, -1)).float()
            next_encoding = f_model(next_obs_batch).view(-1).data.cpu().numpy()
            if S1 is not None:
                next_encoding = np.matmul(S1, next_encoding)
            # next_encoding = next_obs_batch.view(-1).data.cpu().numpy()

            x_input = np.concatenate([curr_encoding, action], axis=0)
            y_output = next_encoding

            x_batch.append(x_input)
            y_batch.append(y_output)

        x_batch = np.vstack(x_batch)
        y_batch = np.vstack(y_batch)

        model = LinearRegression(fit_intercept=False).fit(x_batch, y_batch)
        weight = model.coef_

        A_numpy = weight[:, : self.state_dim]
        B_numpy = weight[:, self.state_dim :]

        return A_numpy, B_numpy

    def _calc_s1_matrix(self, dataset, f_model, k1):
        curr_state_batch = []
        learned_curr_obs_batch = []

        for dp in dataset:
            state = dp.get_states()[k1]
            obs = dp.get_observations()[k1]

            curr_obs_batch_ = cuda_var(torch.from_numpy(obs).view(1, -1)).float()
            curr_encoding = f_model(curr_obs_batch_).view(-1).data.cpu().numpy()

            curr_state_batch.append(state)
            learned_curr_obs_batch.append(curr_encoding)

        curr_state_batch = np.vstack(curr_state_batch)
        learned_curr_obs_batch = np.vstack(learned_curr_obs_batch)

        model_obs = LinearRegression(fit_intercept=False).fit(learned_curr_obs_batch, curr_state_batch)
        model_obs_loss = model_obs.score(learned_curr_obs_batch, curr_state_batch)
        weight_obs = model_obs.coef_

        self.logger.log("S1 loss is %f" % model_obs_loss)
        self.logger.log("S1 is %r\n" % weight_obs)

        return weight_obs

    def train(self, env):
        k = 5 * self.state_dim
        k1 = self.k0 + k

        self.logger.log("Using K0=%d, K=%d, which gives K1 (K0+K1) = %d" % (self.k0, k, k1))

        # Collect dataset
        dataset = []
        for ix in range(1, 3 * self.num_samples + 1):
            if ix % 100 == 0:
                self.logger.log("Collecting episode number %d" % ix)

            obs, info = env.reset()
            eps = Episode(state=info["state"], observation=obs)

            for j in range(0, k1 + 1):
                action = np.random.normal(loc=0.0, scale=1.0, size=self.act_dim)
                obs, reward, done, info = env.step(action)

                # Removing observations that will never be used to save memory
                # The next line can be removed when memory is not a concern
                obs = None if self.k0 + 2 < j < k1 - 2 else obs

                eps.add(action=action, reward=reward, new_obs=obs, new_state=info["state"])

                assert not done, "Cannot be done"

            eps.terminate()
            dataset.append(eps)

        # Solve the first regression problem
        h_model = self._train_hmodel(dataset[: self.num_samples], self.k0, k)

        # Perform PCA
        v_matrix = self._train_v_matrix(dataset[self.num_samples : 2 * self.num_samples], h_model, k1)
        f_model = RichIDFModel(h_model=h_model, v_matrix=v_matrix)

        s1 = self._calc_s1_matrix(dataset, f_model, k1)

        # Solve for A and B
        A, B = self._train_a_b(dataset[2 * self.num_samples :], f_model, k1)

        # Check whether our estimates recover A/B using S_1 as a similarity transform
        s1inv = np.linalg.inv(s1)
        A_est = s1.dot(A.dot(s1inv))
        B_est = s1.dot(B)

        gold_A, gold_B, _, _ = env.env.get_model()

        A_diff = np.linalg.norm(gold_A - A_est)
        B_diff = np.linalg.norm(gold_B - B_est)

        self.logger.log("================")
        self.logger.log("Gold A:")
        self.logger.log("%r" % A)
        self.logger.log("Gold B:")
        self.logger.log("%r" % B)
        self.logger.log("================")

        self.logger.log("================")
        self.logger.log("Estimated A:")
        self.logger.log("%r" % A_est)
        self.logger.log("Estimated B:")
        self.logger.log("%r" % B_est)
        self.logger.log("================")

        self.logger.log("A_diff %.3f \t B_diff %.3f" % (A_diff, B_diff))

        return {"A_diff": A_diff, "B_diff": B_diff}
