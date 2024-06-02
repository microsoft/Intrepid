import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.cuda import cuda_var
from model.encoder.encoder_wrapper import EncoderModelWrapper
from learning.learning_utils.reconstruct_observation import ReconstructObservation
from learning.core_learner.abstract_video_rep_learner import AbstractVideoRepLearner


class TemporalContrastiveVideoModel(nn.Module):

    def __init__(self, exp_setup):
        super(TemporalContrastiveVideoModel, self).__init__()

        self.encoder_type = exp_setup.constants["encoder_type"]
        self.height, self.width, self.channel = exp_setup.config["obs_dim"]
        self.vq_dim = exp_setup.constants["vq_dim"]

        self.loss_normalize = exp_setup.constants["normalize"] > 0
        self.model_normalize = False  # exp_setup.constants["normalize"] > 0

        exp_setup.logger.log(
            f"Running temporal contrastive with Model={self.encoder_type}, Model Normalization={self.model_normalize}, "
            f"Loss Normalization={self.loss_normalize}.")

        self.encoder = EncoderModelWrapper.get_encoder(
            self.encoder_type,
            height=self.height,
            width=self.width,
            channel=self.channel,
            out_dim=self.vq_dim,
            bootstrap_model=None,
            normalize=self.model_normalize
        )

        if torch.cuda.is_available():
            self.cuda()


class TemporalContrastiveVideo(AbstractVideoRepLearner):

    NAME = "tempcont"

    def __init__(self, exp_setup):
        super(TemporalContrastiveVideo, self).__init__(exp_setup)

        self.exp_setup = exp_setup
        self.height, self.width, self.channel = exp_setup.config["obs_dim"]
        self.batch_size = exp_setup.constants["encoder_training_batch_size"]
        self.temperature = exp_setup.constants["temperature"]

        self.mask = dict()
        for b in range(1, self.batch_size + 1):
            self.mask[b] = self.get_negative_mask(b)

        self.logger = exp_setup.logger
        self.reconstructor = ReconstructObservation(exp_setup)

    @staticmethod
    def get_negative_mask(batch_size):

        negative_mask = torch.ones((batch_size, 2 * batch_size)).float()
        for i in range(batch_size):
            negative_mask[i, i] = 0.0
            negative_mask[i, i + batch_size] = 0.0

        negative_mask = torch.cat((negative_mask, negative_mask), 0).bool()

        return cuda_var(negative_mask)

    @staticmethod
    def make_model(exp_setup):
        return TemporalContrastiveVideoModel(exp_setup)

    def _calc_loss(self, model, prep_batch, test=False):

        train_flag = model.training
        if test:
            model.eval()

        _, obs1, obs2, _ = prep_batch
        my_batch_size = obs1.size(0)

        obs = torch.cat([obs1, obs2], dim=0)                           # 2 batch x (...)
        ftrs = model.encoder(obs)                                      # 2 batch x dim
        if model.loss_normalize:
            ftrs = F.normalize(ftrs, dim=1)
        ftrs_1, ftrs_2 = torch.chunk(ftrs, 2, dim=0)                   # both are batch x dim

        pos = (ftrs_1 * ftrs_2).sum(1)                                # batch
        pos = torch.cat([pos, pos], dim=0)                            # 2 batch

        neg_scores = torch.mm(ftrs, ftrs.t().contiguous())            # 2 batch x 2 batch

        # 2 batch x 2 batch with each row having 2 * (batch - 1) 1's and remaining 0's
        default_mask = self.mask[my_batch_size]
        neg_scores = neg_scores.masked_select(default_mask).view(2 * my_batch_size, -1)     # 2 batch x k

        denom = torch.cat([pos.view(-1, 1), neg_scores], dim=1)      # 2 batch x (k + 1)
        base_loss = - pos / self.temperature + torch.logsumexp(denom / self.temperature, dim=1)  # 2 batch
        base_loss = base_loss.mean()
        loss = base_loss        # loss is the same as base_loss as there is no auxiliary loss

        info_dict = {"batch_size": my_batch_size, "base_loss": base_loss.item()}

        if test and train_flag:
            model.train()

        return base_loss, loss, info_dict

    def _accumulate_info_dict(self, info_dicts):
        """
                Given a list of info_dicts, accumulate their result and return a new info_dict with mean results.
        :param info_dicts: List of dictionary containg floats
        :return: return a single dictionary with mean results
        """

        merged_mean_dict = dict()

        if len(info_dicts) == 0:
            return merged_mean_dict

        keys = info_dicts[0].keys()

        num_examples = sum([info_dict["batch_size"] for info_dict in info_dicts])
        merged_mean_dict["num_examples"] = num_examples

        for key in keys:

            if key == "batch_size":
                continue

            else:
                sum_val = sum([info_dict[key] * info_dict["batch_size"] for info_dict in info_dicts])
                merged_mean_dict[key] = sum_val / float(max(1, num_examples))

        return merged_mean_dict
