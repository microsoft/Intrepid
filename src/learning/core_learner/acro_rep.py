import torch
import numpy as np

from utils.cuda import cuda_var
from learning.learning_utils.generic_learner import GenericLearner


class ACRORepLearner(GenericLearner):

    def __init__(self, exp_setup, discrete):
        super(ACRORepLearner, self).__init__(exp_setup)
        self.discrete = discrete

    def calc_loss(self, model, batch, test=False):

        obs1 = np.array([dp[0] for dp in batch])
        action = [dp[1] for dp in batch]
        obs2 = np.array([dp[2] for dp in batch])
        k = [dp[3] for dp in batch]

        info_dict = dict()

        obs1 = torch.FloatTensor(obs1)
        obs2 = torch.FloatTensor(obs2)

        obs1 = cuda_var(obs1.permute(0, 3, 1, 2))
        obs2 = cuda_var(obs2.permute(0, 3, 1, 2))
        action = cuda_var(torch.LongTensor(action))
        k = cuda_var(torch.LongTensor(k))

        if self.discrete:
            # Use log loss
            predicted_action_log_prob = model(obs1, obs2, k)                               # Batch x num_actions

            if predicted_action_log_prob.dim() == 2:
                action = action.unsqueeze(1)                                            # Batch x 1
                loss = - torch.gather(predicted_action_log_prob, dim=1, index=action)   # Batch
                loss = loss.mean()
            else:
                raise AssertionError("Predicted action log_prob should be of size batch x num_actions")
        else:
            # Use MSE
            predicted_action = model(obs1, obs2)
            loss = ((predicted_action - action) ** 2).sum(1).mean()

        return loss, info_dict


class ACRORep:
    """
    Trains ACRO representation using multi-step dynamics
    Does not use any latent bottleneck
    """

    def __init__(self, exp_setup):
        self.exp_setup = exp_setup

    def train(self, dataset, discrete, tensorboard=None, bootstrap_model=None, **kwargs):
        """
            Given an offline dataset consisting of list of episodes, we train a multi-step representation
            :param dataset: Either a list of multi-step transitions where a single multi-step transition has a format of
                            obs, action, next_obs, k  where k is a number denoting the number of steps between next_obs
                            and obs. If k=1, then this is a usual 1-step transition.

            :param discrete: If true then actions are discrete, otherwise continuous
        """

        learner = ACRORepLearner(self.exp_setup, discrete=discrete)

        model_type = "inv_dynamics"
        model_name = "mlp-conv"     # TODO make it general

        best_model, info = learner.do_train(model_type=model_type,
                                            model_name=model_name,
                                            dataset=dataset,
                                            tensorboard=tensorboard,
                                            bootstrap_model=bootstrap_model,
                                            discrete_action=discrete,
                                            eval_iter=100,          # 100
                                            max_epoch=75,          # 75
                                            learning_rate=self.exp_setup.constants["encoder_training_lr"],
                                            scale_down=1,          # kwargs["max_k"]
                                            **kwargs)

        return best_model, info
