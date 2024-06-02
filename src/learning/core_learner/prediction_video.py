import torch
import torch.nn as nn

from torchvision.utils import save_image as save_image

from model.bottleneck.vq_bottleneck import VQBottleneckWrapper
from model.decoder.decoder_wrapper import DecoderModelWrapper
from model.encoder.encoder_wrapper import EncoderModelWrapper
from model.bottleneck.gaussian_bottleneck import GaussianBottleneck
from learning.core_learner.abstract_video_rep_learner import AbstractVideoRepLearner


class PredictionVideoModel(nn.Module):

    def __init__(self, exp_setup):
        super(PredictionVideoModel, self).__init__()

        self.horizon = exp_setup.config["horizon"]
        self.num_actions = exp_setup.config["num_actions"]
        self.encoder_type = exp_setup.constants["encoder_type"]
        self.reconstructor_type = exp_setup.constants["decoder_type"]
        self.height, self.width, self.channel = exp_setup.config["obs_dim"]
        self.vq_dim = exp_setup.constants["vq_dim"]
        self.use_middle = exp_setup.constants["use_middle"]
        # self.encoder_dim = exp_setup.constants["hidden_dim"]

        # Time step encoding
        self.max_k = exp_setup.constants["max_k"]
        self.use_klst = True if self.max_k > 1 and exp_setup.constants["rep_alg"] == "next-frame" else False

        if self.use_klst:
            self.k_embed_dim = 32  # Size of k_embed # Was same as self.vq_dim
            self.k_embedding = nn.Embedding(self.max_k + 1, self.k_embed_dim)

        self.use_vq = exp_setup.constants["use_vq"] > 0
        self.use_gb = exp_setup.constants["use_gb"] > 0

        if self.use_vq:
            self.vq = VQBottleneckWrapper.get_bottleneck("vq",
                                                         constants=exp_setup.constants,
                                                         heads=1,
                                                         codebook_size=exp_setup.constants["vq_codebook_size"])

        if self.use_gb:
            self.gaussian_bottleneck = GaussianBottleneck(hidden_dim=self.vq_dim)

        self.bn = nn.BatchNorm1d(self.vq_dim)

        self.encoder = EncoderModelWrapper.get_encoder(
            self.encoder_type,
            height=self.height,
            width=self.width,
            channel=self.channel,
            out_dim=self.vq_dim,
            bootstrap_model=None
        )

        self.action_encode_mlp2 = nn.Sequential(nn.Linear(self.vq_dim, self.vq_dim),
                                                nn.LeakyReLU(),
                                                nn.Linear(self.vq_dim, self.vq_dim),
                                                nn.LeakyReLU(),
                                                nn.Linear(self.vq_dim, self.vq_dim)
                                                )

        self.phi_probe_decoder = DecoderModelWrapper.get_decoder(
            model_name=self.reconstructor_type,
            height=self.height,
            width=self.width,
            channel=self.channel,
            out_dim=self.vq_dim,
            bootstrap_model=None
        )

        # self.decoder = Conditional_UNet(self.vq_dim)
        self.decoder = DecoderModelWrapper.get_decoder(
            model_name=self.reconstructor_type,
            height=self.height,
            width=self.width,
            channel=self.channel,
            out_dim=(self.vq_dim + self.k_embed_dim if self.use_klst else self.vq_dim),
            bootstrap_model=None
        )

        if torch.cuda.is_available():
            self.cuda()


class PredictionVideo(AbstractVideoRepLearner):

    def __init__(self, exp_setup, is_autoencoder):
        super(PredictionVideo, self).__init__(exp_setup)

        self.exp_setup = exp_setup
        self.save_img_flag = False
        self.save_path = exp_setup.config["save_path"]
        self.experiment = exp_setup.experiment

        # Learning parameters
        self.is_autoencoder = is_autoencoder
        self.use_vq = exp_setup.constants["use_vq"] > 0
        self.use_gb = exp_setup.constants["use_gb"] > 0
        self.do_randn = exp_setup.constants["use_randn"] > 0
        self.max_k = exp_setup.constants["max_k"]
        self.use_klst = True if self.max_k > 1 and not self.is_autoencoder else False

        self.img_ctr = 0

        if self.is_autoencoder:
            self.logger.log(f"Created Autoencoder Model with Gaussial Bottleneck set to {self.use_gb}, VQ bottleneck"
                            f"set to {self.use_vq}, time encoding {self.use_klst} and "
                            f"using base image as noisy set to {self.do_randn}")
        else:
            self.logger.log(f"Created Next-Frame Prediction Model with Gaussial Bottleneck set to {self.use_gb}, VQ"
                            f"bottleneck set to {self.use_vq}, time encoding {self.use_klst} and "
                            f"using base image as noisy set to {self.do_randn}")

        if self.save_img_flag:
            self.logger.log("Warning! Save image flag is on meaning that images will be saved during the process. "
                            "This should be ideally set only during debugging as it can significantly slow down "
                            "the experiment.")

    @staticmethod
    def make_model(exp_setup):
        return PredictionVideoModel(exp_setup)

    def _calc_loss(self, model, prep_batch, test=False):

        latent_prediction_vec, latent_prediction_loss, info_dict = self._encoder_observation(model, prep_batch, test)

        train_flag = model.training
        if test:
            model.eval()

        _, obs1, obs2, klst = prep_batch

        # if self.do_randn:
        #     base_image = torch.randn_like(obs1)
        # else:
        #     base_image = torch.zeros_like(obs1)
        # pred = model.decoder(base_image, latent_prediction_vec)

        if self.use_klst:
            kemb = model.k_embedding(klst)
            latent_prediction_vec = torch.cat([latent_prediction_vec, kemb], dim=1)

        pred = model.decoder(latent_prediction_vec)

        self.img_ctr += 1
        if self.img_ctr % 1000 == 0:  # self.save_img_flag:
            img_ctr_ix = int(self.img_ctr / 1000)
            save_image(obs1, f'%s/obs_{img_ctr_ix}.png' % self.experiment)
            save_image(pred, f'%s/pred_{img_ctr_ix}.png' % self.experiment)

        if self.is_autoencoder:
            # Take difference from the input image
            diff = (obs1 - pred)
        else:
            # Take difference from the future image
            diff = (obs2 - pred)

        generation_loss = (diff * diff).mean()

        # Total loss is the sum of Generation Loss and Latent action Model loss
        loss = generation_loss + latent_prediction_loss

        # Base loss that doesn't use action data and that we use for early stopping is just the generation loss
        base_loss = generation_loss

        info_dict["generation_loss"] = generation_loss.item()

        if test and train_flag:
            model.train()

        return base_loss, loss, info_dict

    def _encoder_observation(self, model, prep_batch, test=False):

        train_flag = model.training
        if test:
            model.eval()

        actions, obs1, _, _ = prep_batch
        info_dict = {"batch_size": obs1.size(0)}

        
        h = model.encoder(obs1)

        x_rec = model.phi_probe_decoder(h.detach())
        # import pdb
        # pdb.set_trace()

        if self.save_img_flag:
            _, _, obs2, _ = prep_batch
            save_image(obs1, '%s/obs1.png' % self.experiment)
            save_image(obs2, '%s/obs2.png' % self.experiment)
            save_image(x_rec, '%s/x_rec.png' % self.experiment)

        # Reconstruction loss
        
        loss = ((x_rec - obs1) ** 2).mean()

        info_dict["dynamics_reconst_loss"] = loss.item()

        if model.use_middle:
            h = model.action_encode_mlp2(h)
            h = model.bn(h)

        if self.use_vq:
            # print('VQ bottleneck: {}'.format(self.use_vq))
            h, indices, vq_loss = VQBottleneckWrapper.vq_helper(model.vq, h)
            loss += vq_loss
            info_dict["vq_loss"] = vq_loss.item()

        if self.use_gb:
            h, klb_loss = model.gaussian_bottleneck.gb_helper(h)
            loss += klb_loss
            info_dict["klb_loss"] = float(klb_loss)     # klb_loss at times can be float, hence .item() doesn't work

        if self.is_autoencoder:
            info_dict["prediction_video_auto_loss"] = loss.item()
        else:
            info_dict["prediction_video_next_loss"] = loss.item()

        if test and train_flag:
            model.train()

        return h, loss, info_dict

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

    def generate(self, model, prep_batch):

        raise NotImplementedError()

        test = True
        latent_prediction_vec, latent_prediction_loss, info_dict = self._calc_latent_prediction(model, prep_batch, test)

        train_flag = model.training
        if test:
            model.eval()

        _, obs1, obs2, _ = prep_batch

        if self.do_randn:
            base_image = torch.randn_like(obs1)
        else:
            base_image = torch.zeros_like(obs1)

        pred = model.decoder(base_image, latent_prediction_vec)
        # save_image(obs1, '%s/original.png' % self.experiment)
        # save_image(pred, '%s/autoencoder_prediction.png' % self.experiment)

        if test and train_flag:
            model.train()


class AutoencoderVideo(PredictionVideo):

    NAME = "autoencoder"

    def __init__(self, exp_setup):
        assert exp_setup.constants["rep_alg"] == AutoencoderVideo.NAME, \
            f"rep_alg name in constants should be {AutoencoderVideo.NAME}"
        super(AutoencoderVideo, self).__init__(exp_setup, is_autoencoder=1)


class NextFrameVideo(PredictionVideo):

    NAME = "next-frame"

    def __init__(self, exp_setup):
        assert exp_setup.constants["rep_alg"] == NextFrameVideo.NAME, \
            f"rep_alg name in constants should be {NextFrameVideo.NAME}"
        super(NextFrameVideo, self).__init__(exp_setup, is_autoencoder=0)
