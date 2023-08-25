import torch
import numpy as np
import torch.nn.functional as F

from utils.cuda import cuda_var
from learning.learning_utils.entropy_decay_policy import EntropyDecayPolicy


class NoiseContrastiveLocal:
    """
    State abstraction using noise contrastive learning with locally normalized probabilities
    """

    def __init__(self, constants, epoch):
        self.entropy_decay_policy = EntropyDecayPolicy(constants, epoch)
        self.entropy_coeff = constants["entropy_reg_coeff"]

    @staticmethod
    def calc_loss(model, batch):
        prev_observations = cuda_var(
            torch.cat(
                [torch.from_numpy(np.array(point[0])).view(1, -1) for point in batch],
                dim=0,
            )
        ).float()
        actions = cuda_var(
            torch.cat(
                [torch.from_numpy(np.array(point[1])).view(1, -1) for point in batch],
                dim=0,
            )
        ).long()
        observations = cuda_var(
            torch.cat(
                [torch.from_numpy(np.array(point[2])).view(1, -1) for point in batch],
                dim=0,
            )
        ).float()

        # Generate a matrix M of size batch x batch where M[i, j] denotes p(y = 1 | x_i, a_i, x'_j)
        # diagonal elements are real transitions, non-diagonal elements are imposter candidates
        scores = model.gen_scores(
            prev_observations=prev_observations,
            actions=actions,
            observations=observations,
        )

        batch_size = len(batch)

        # Single negative example
        log_probs = F.logsigmoid((2 * torch.eye(batch_size) - 1).cuda() * scores)

        classification_loss = -torch.sum(
            log_probs
            * (torch.eye(batch_size).cuda() / batch_size + (1 - torch.eye(batch_size).cuda()) / batch_size / (batch_size - 1))
        )

        info_dict = dict()
        info_dict["classification_loss"] = classification_loss
        info_dict["mean_entropy"] = 0.0
        info_dict["entropy_coeff"] = 0.0

        return classification_loss, info_dict
