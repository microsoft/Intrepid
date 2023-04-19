import torch
import numpy as np

from utils.cuda import cuda_var
from learning.learning_utils.entropy_decay_policy import EntropyDecayPolicy


class NoiseContrastiveGlobal:
    """
        State abstraction using noise contrastive learning with globally normalized probabilities
    """

    def __init__(self, constants, epoch):
        self.entropy_decay_policy = EntropyDecayPolicy(constants, epoch)
        self.entropy_coeff = constants["entropy_reg_coeff"]

    @staticmethod
    def calc_loss(model, batch, epoch, discretized, test_set_errors=None, past_entropy=None):

        prev_observations = cuda_var(torch.cat([torch.from_numpy(np.array(point[0])).view(1, -1)
                                                for point in batch], dim=0)).float()
        actions = cuda_var(torch.cat([torch.from_numpy(np.array(point[1])).view(1, -1)
                                      for point in batch], dim=0)).long()
        observations = cuda_var(torch.cat([torch.from_numpy(np.array(point[2])).view(1, -1)
                                           for point in batch], dim=0)).float()

        # Generate a matrix M of size batch x batch where M[i, j] denotes p(y = 1 | x_i, a_i, x'_j)
        # diagonal elements are real transitions, non-diagonal elements are imposter candidates
        scores = model.gen_scores(prev_observations=prev_observations,
                                  actions=actions,
                                  observations=observations)

        classification_loss = (torch.diagonal(scores, 0) - torch.logsumexp(scores, 1)).mean()

        info_dict = dict()
        info_dict["classification_loss"] = classification_loss
        info_dict["mean_entropy"] = 0.0
        info_dict["entropy_coeff"] = 0.0

        return classification_loss, info_dict
