import torch
import numpy as np

from utils.cuda import cuda_var
from learning.learning_utils.entropy_decay_policy import EntropyDecayPolicy


class GeneralizedInverseKinematics:
    """
    State abstraction using generalized inverse kinematics
    """

    def __init__(self, constants, epoch):
        self.entropy_decay_policy = EntropyDecayPolicy(constants, epoch)
        self.entropy_coeff = constants["entropy_reg_coeff"]

    def calc_loss(
        self, model, batch, epoch, discretized, test_set_errors=None, past_entropy=None
    ):
        past_observations = cuda_var(
            torch.cat(
                [
                    torch.from_numpy(np.array(point.get_curr_obs())).view(1, -1)
                    for point in batch
                ],
                dim=0,
            )
        ).float()
        past_actions = cuda_var(
            torch.cat(
                [
                    torch.from_numpy(np.array(point.get_action())).view(1, -1)
                    for point in batch
                ],
                dim=0,
            )
        ).long()
        observations = cuda_var(
            torch.cat(
                [
                    torch.from_numpy(np.array(point.get_next_obs())).view(1, -1)
                    for point in batch
                ],
                dim=0,
            )
        ).float()

        # Compute loss
        log_probs, meta_dict = model.gen_log_prob(
            prev_observations=past_observations,
            observations=observations,
            discretized=discretized,
        )  # outputs of size batch x num_actions
        classification_loss = -torch.mean(log_probs.gather(1, past_actions.view(-1, 1)))

        if discretized:
            # For discretized models, there is an internal classification step representation by a probability
            # distribution that can be controlled using entropy bonus
            # NOT SUPPORTED AT THE MOMENT
            decay_coeff = self.entropy_decay_policy.get_entropy_coeff(
                epoch, test_set_errors, past_entropy
            )
            loss = (
                classification_loss
                - self.entropy_coeff * decay_coeff * meta_dict["mean_entropy"]
            )
        else:
            decay_coeff = None
            loss = classification_loss

        info_dict = dict()

        info_dict["classification_loss"] = classification_loss

        if discretized:
            info_dict["mean_entropy"] = meta_dict["mean_entropy"]
            info_dict["entropy_coeff"] = self.entropy_coeff * decay_coeff
        else:
            info_dict["mean_entropy"] = -1
            info_dict["entropy_coeff"] = 0.0

        return loss, info_dict
