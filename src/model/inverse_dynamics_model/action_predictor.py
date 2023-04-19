import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionPredictor(nn.Module):

    def __init__(self, config, constants, bootstrap_model):
        super(ActionPredictor, self).__init__()

        input_dim = 2 * config["obs_dim"]

        self.network = nn.Sequential(
            nn.Linear(input_dim, constants["hidden_dim"]),
            nn.LeakyReLU(),
            nn.Linear(constants["hidden_dim"], constants["hidden_dim"]),
            nn.LeakyReLU(),
            nn.Linear(constants["hidden_dim"], config["num_actions"])
        )

        if bootstrap_model is not None:
            self.load_state_dict(bootstrap_model.state_dict())

        if torch.cuda.is_available():
            self.cuda()

    def gen_log_prob(self, curr_obs, next_obs):

        x = torch.cat([curr_obs, next_obs], dim=1)

        out = self.network(x)
        return F.log_softmax(out, dim=1)

    def predict_action(self, curr_obs, next_obs):
        log_prob = self.gen_log_prob(curr_obs, next_obs)
        action = log_prob.argmax(dim=-1)
        return action


class ActionPredictorFlatNN(nn.Module):

    def __init__(self, config, constants, bootstrap_model):
        super(ActionPredictorFlatNN, self).__init__()

        self.input_shape = config["obs_dim"]
        n = self.input_shape[0]
        m = self.input_shape[1]
        self.image_conv1 = nn.Sequential(
            nn.Flatten()
        )
        self.image_conv2 = nn.Sequential(
            nn.Flatten()
        )
        self.embedding_size = m*n*3

        self.network = nn.Sequential(
            nn.Linear(self.embedding_size*2, constants["hidden_dim"]),
            nn.LeakyReLU(),
            nn.Linear(constants["hidden_dim"], constants["hidden_dim"]),
            nn.LeakyReLU(),
            nn.Linear(constants["hidden_dim"], config["num_actions"])
        )

        if bootstrap_model is not None:
            self.load_state_dict(bootstrap_model.state_dict())

        if torch.cuda.is_available():
            self.cuda()

    def obs_preprocess(self, obs):
        x = obs.reshape(-1, *self.input_shape)
        x = x.transpose(1, 3).transpose(2, 3)
        return x

    def gen_log_prob(self, curr_obs, next_obs):
        curr_f = self.image_conv1(self.obs_preprocess(curr_obs))
        next_f = self.image_conv2(self.obs_preprocess(next_obs))
        x = torch.cat([curr_f, next_f], dim=1)
        out = self.network(x)
        return F.log_softmax(out, dim=1)

    def predict_action(self, curr_obs, next_obs):
        log_prob = self.gen_log_prob(curr_obs, next_obs)
        action = log_prob.argmax(dim=-1)
        return action


class ActionPredictorCNN1(nn.Module):

    def __init__(self, config, constants, bootstrap_model):
        super(ActionPredictorCNN1, self).__init__()

        self.input_shape = config["obs_dim"]
        n = self.input_shape[0]
        m = self.input_shape[1]
        self.image_conv1 = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten()
        )
        self.image_conv2 = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten()
        )
        self.embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        self.network = nn.Sequential(
            nn.Linear(self.embedding_size*2, constants["hidden_dim"]),
            nn.LeakyReLU(),
            nn.Linear(constants["hidden_dim"], constants["hidden_dim"]),
            nn.LeakyReLU(),
            nn.Linear(constants["hidden_dim"], config["num_actions"])
        )

        if bootstrap_model is not None:
            self.load_state_dict(bootstrap_model.state_dict())

        if torch.cuda.is_available():
            self.cuda()

    def obs_preprocess(self, obs):
        x = obs.reshape(-1, *self.input_shape)
        x = x.transpose(1, 3).transpose(2, 3)
        return x

    def gen_log_prob(self, curr_obs, next_obs):
        curr_f = self.image_conv1(self.obs_preprocess(curr_obs))
        next_f = self.image_conv2(self.obs_preprocess(next_obs))
        x = torch.cat([curr_f, next_f], dim=1)
        out = self.network(x)
        return F.log_softmax(out, dim=1)

    def predict_action(self, curr_obs, next_obs):
        log_prob = self.gen_log_prob(curr_obs, next_obs)
        action = log_prob.argmax(dim=-1)
        return action


class ActionPredictorCNN2(nn.Module):

    def __init__(self, config, constants, bootstrap_model):
        super(ActionPredictorCNN2, self).__init__()

        self.input_shape = config["obs_dim"]
        n = self.input_shape[0]
        m = self.input_shape[1]
        self.image_conv1 = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((4, 4)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Flatten()
        )
        self.image_conv2 = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((4, 4)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Flatten()
        )
        self.embedding_size = ((n - 1) // 4 - 1) * ((m - 1) // 4 - 1) * 32

        self.network = nn.Sequential(
            nn.Linear(self.embedding_size*2, constants["hidden_dim"]),
            nn.LeakyReLU(),
            nn.Linear(constants["hidden_dim"], constants["hidden_dim"]),
            nn.LeakyReLU(),
            nn.Linear(constants["hidden_dim"], config["num_actions"])
        )

        if bootstrap_model is not None:
            self.load_state_dict(bootstrap_model.state_dict())

        if torch.cuda.is_available():
            self.cuda()

    def obs_preprocess(self, obs):
        x = obs.reshape(-1, *self.input_shape)
        x = x.transpose(1, 3).transpose(2, 3)
        return x

    def gen_log_prob(self, curr_obs, next_obs):
        curr_f = self.image_conv1(self.obs_preprocess(curr_obs))
        next_f = self.image_conv2(self.obs_preprocess(next_obs))
        x = torch.cat([curr_f, next_f], dim=1)
        out = self.network(x)
        return F.log_softmax(out, dim=1)

    def predict_action(self, curr_obs, next_obs):
        log_prob = self.gen_log_prob(curr_obs, next_obs)
        action = log_prob.argmax(dim=-1)
        return action


class ActionPredictorCNN3(nn.Module):

    def __init__(self, config, constants, bootstrap_model):
        super(ActionPredictorCNN3, self).__init__()

        self.input_shape = config["obs_dim"]
        n = self.input_shape[0]
        m = self.input_shape[1]
        self.embedding_size = ((n - 1) // 4 - 1) * ((m - 1) // 4 - 1) * 32  # TODO

        self.image_conv1 = nn.Sequential(
            nn.Conv2d(3, 16, (8, 8), 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, (4, 4), 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(800, 256)
        )
        self.image_conv2 = nn.Sequential(
            nn.Conv2d(3, 16, (8, 8), 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, (4, 4), 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(800, 256)
        )

        self.network = nn.Sequential(
            nn.Linear(2 * 256, constants["hidden_dim"]),
            nn.LeakyReLU(),
            nn.Linear(constants["hidden_dim"], config["num_actions"])
        )

        if bootstrap_model is not None:
            self.load_state_dict(bootstrap_model.state_dict())

        if torch.cuda.is_available():
            self.cuda()

    def obs_preprocess(self, obs):
        x = obs.reshape(-1, *self.input_shape)
        x = x.transpose(1, 3).transpose(2, 3)
        return x

    def gen_log_prob(self, curr_obs, next_obs):
        curr_f = self.image_conv1(self.obs_preprocess(curr_obs))
        next_f = self.image_conv2(self.obs_preprocess(next_obs))
        x = torch.cat([curr_f, next_f], dim=1)
        out = self.network(x)
        return F.log_softmax(out, dim=1)

    def predict_action(self, curr_obs, next_obs):
        log_prob = self.gen_log_prob(curr_obs, next_obs)
        action = log_prob.argmax(dim=-1)
        return action


class ActionPredictorCNN4(nn.Module):

    def __init__(self, config, constants, bootstrap_model):
        super(ActionPredictorCNN4, self).__init__()

        self.input_shape = config["obs_dim"]
        n = self.input_shape[0]
        m = self.input_shape[1]
        self.embedding_size = ((n - 1) // 4 - 1) * ((m - 1) // 4 - 1) * 32  # TODO

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (8, 8), 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, (4, 4), 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(800, 256)
        )

        self.network = nn.Sequential(
            nn.Linear(2 * 256, constants["hidden_dim"]),
            nn.LeakyReLU(),
            nn.Linear(constants["hidden_dim"], config["num_actions"])
        )

        if bootstrap_model is not None:
            self.load_state_dict(bootstrap_model.state_dict())

        if torch.cuda.is_available():
            self.cuda()

    def obs_preprocess(self, obs):
        x = obs.reshape(-1, *self.input_shape)
        x = x.transpose(1, 3).transpose(2, 3)
        return x

    def gen_log_prob(self, curr_obs, next_obs):
        curr_f = self.image_conv(self.obs_preprocess(curr_obs))
        next_f = self.image_conv(self.obs_preprocess(next_obs))
        x = torch.cat([curr_f, next_f], dim=1)
        out = self.network(x)
        return F.log_softmax(out, dim=1)

    def predict_action(self, curr_obs, next_obs):
        log_prob = self.gen_log_prob(curr_obs, next_obs)
        action = log_prob.argmax(dim=-1)
        return action


class ActionPredictorCNN5(nn.Module):

    def __init__(self, config, constants, bootstrap_model):
        super(ActionPredictorCNN5, self).__init__()

        self.input_shape = config["obs_dim"]
        n = self.input_shape[0]
        m = self.input_shape[1]
        self.embedding_size = ((n - 1) // 4 - 1) * ((m - 1) // 4 - 1) * 32  # TODO

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (8, 8), 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, (4, 4), 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(800, 256)
        )

        self.network = nn.Sequential(
            nn.Linear(256, constants["hidden_dim"]),
            nn.LeakyReLU(),
            nn.Linear(constants["hidden_dim"], config["num_actions"])
        )

        if bootstrap_model is not None:
            self.load_state_dict(bootstrap_model.state_dict())

        if torch.cuda.is_available():
            self.cuda()

    def obs_preprocess(self, obs):
        x = obs.reshape(-1, *self.input_shape)
        x = x.transpose(1, 3).transpose(2, 3)
        return x

    def gen_log_prob(self, curr_obs, next_obs):
        curr_f = self.image_conv(self.obs_preprocess(curr_obs))
        next_f = self.image_conv(self.obs_preprocess(next_obs))
        x = curr_f * next_f
        out = self.network(x)
        return F.log_softmax(out, dim=1)

    def predict_action(self, curr_obs, next_obs):
        log_prob = self.gen_log_prob(curr_obs, next_obs)
        action = log_prob.argmax(dim=-1)
        return action


class ActionPredictorCNN6(nn.Module):

    def __init__(self, config, constants, bootstrap_model):
        super(ActionPredictorCNN6, self).__init__()

        self.input_shape = config["obs_dim"]
        n = self.input_shape[0]
        m = self.input_shape[1]
        self.embedding_size = ((n - 1) // 4 - 1) * ((m - 1) // 4 - 1) * 32  # TODO

        self.image_conv = nn.Sequential(
            nn.Conv2d(6, 16, (8, 8), 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, (4, 4), 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(800, config["num_actions"])
        )

        if bootstrap_model is not None:
            self.load_state_dict(bootstrap_model.state_dict())

        if torch.cuda.is_available():
            self.cuda()

    def obs_preprocess(self, obs):
        x = obs.reshape(-1, *self.input_shape)
        x = x.transpose(1, 3).transpose(2, 3)
        return x

    def gen_log_prob(self, curr_obs, next_obs):
        curr_obs = self.obs_preprocess(curr_obs)
        next_obs = self.obs_preprocess(next_obs)
        obs = torch.cat([curr_obs, next_obs], dim=1)
        x = self.image_conv(obs)
        return F.log_softmax(x, dim=1)

    def predict_action(self, curr_obs, next_obs):
        log_prob = self.gen_log_prob(curr_obs, next_obs)
        action = log_prob.argmax(dim=-1)
        return action


class ActionPredictorCNN7(nn.Module):

    def __init__(self, config, constants, bootstrap_model):
        super(ActionPredictorCNN7, self).__init__()

        self.input_shape = config["obs_dim"]
        n = self.input_shape[0]
        m = self.input_shape[1]
        self.embedding_size = ((n - 1) // 4 - 1) * ((m - 1) // 4 - 1) * 32  # TODO

        self.image_conv = nn.Sequential(
            nn.Conv2d(6, 16, (8, 8), 8),
            nn.ReLU(),
            nn.Conv2d(16, 32, (4, 4), 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, config["num_actions"])
        )

        if bootstrap_model is not None:
            self.load_state_dict(bootstrap_model.state_dict())

        if torch.cuda.is_available():
            self.cuda()

    def obs_preprocess(self, obs):
        x = obs.reshape(-1, *self.input_shape)
        x = x.transpose(1, 3).transpose(2, 3)
        return x

    def gen_log_prob(self, curr_obs, next_obs):
        curr_obs = self.obs_preprocess(curr_obs)
        next_obs = self.obs_preprocess(next_obs)
        obs = torch.cat([curr_obs, next_obs], dim=1)
        x = self.image_conv(obs)
        return F.log_softmax(x, dim=1)

    def predict_action(self, curr_obs, next_obs):
        log_prob = self.gen_log_prob(curr_obs, next_obs)
        action = log_prob.argmax(dim=-1)
        return action

