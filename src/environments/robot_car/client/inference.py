import numpy as np
import os
import random
import argparse
import torch
import torch.nn.functional as F

from environments.robot_car.client.state import CarState
from learning.planning.high_low_plan import HighLowPlanner
from model.misc.robot_car.autoencoder_train import CarAutoencoder
from model.misc.robot_car.latent_forward import LatentForward


# Mock classes for high level planner
# Use these to run HighLowPlanner with only low level planning
class MockKMeans:
    def __init__(self, **kwargs):
        pass

    def predict(self, state):
        return [0]


class MockMDP:
    def __init__(self, **kwargs):
        self.discrete_transition = np.array([[0]])
        self.unique_states = []


# Model inference wrapper class for using latent space forward model
class LatentForwardInference:
    def __init__(self, goal_state, model_path, planner_type="low"):
        self.goal_state = goal_state
        self.latent_dim = 512 * 3
        self.action_dim = 4
        self.device = "cpu"
        # self.device = "cuda"

        # seed
        torch.manual_seed(0)
        np.random.seed(0)
        if self.device == "cuda":
            torch.cuda.manual_seed(0)

        # load dynamics
        encoder_path = os.path.join(model_path, "autoencoder.ckpt")
        forward_path = os.path.join(model_path, "latent_forward.ckpt")
        self.encoder = CarAutoencoder.load_from_checkpoint(encoder_path, strict=False, map_location=self.device).eval()
        self.forward = LatentForward.load_from_checkpoint(forward_path, strict=False, map_location=self.device).eval()

        if planner_type == "low":
            self.planner = HighLowPlanner(
                nz=self.latent_dim,
                nu=self.action_dim,
                enc=self.encoder,
                forward_dyn=self.forward,
                kmeans=MockKMeans(),
                MDP=MockMDP(),
            )
        elif planner_type == "random":
            self.planner = RandomShootingPlanner(self.latent_dim, self.action_dim, self.encoder, self.forward)
        else:
            raise ValueError(f"Unknown planner type {planner_type}")

    def _to_tensor(self, state):
        array = state.stack()
        array = array / 256.0
        assert array.max() <= 1.0
        assert array.min() >= 0.0
        assert array.shape == (3, 3, 256, 256), f"Array shape should be (3, 3, 256, 256), got {array.shape}"
        return torch.FloatTensor(array).unsqueeze(0).to(self.device)

    def _unnormalize_action(self, action):
        if isinstance(action, torch.Tensor):
            assert action.dim() == 1, f"Action should be 1D, got {action.dim()}"
            assert action.size() == (4,), f"Action should be size 4, got {action.size()}"
        if isinstance(action, np.ndarray):
            assert action.ndim == 1, f"Action should be 1D, got {action.ndim}"
            assert action.shape == (4,), f"Action should be size 4, got {action.shape}"
        # Action: [angle, direction, speed, time]
        # 0 if step_info['direction'] == 'forward' else 1,
        ACTION_MAX = np.array([50.0, 1.0, 0.5, 0.5])
        ACTION_MIN = np.array([-10.0, 0.0, 0.0, 0.1])
        unnormed = (action * (ACTION_MAX - ACTION_MIN)) + ACTION_MIN
        return {
            "angle": float(unnormed[0]),
            "direction": "forward" if unnormed[1] < 0.5 else "reverse",
            "speed": float(unnormed[2]),
            "time": float(unnormed[3]),
        }

    def _normalize_action(self, action):
        assert all([key in action for key in ["angle", "direction", "speed", "time"]])
        action = np.array([action["angle"], 0.0 if action["direction"] == "forward" else 1.0, action["speed"], action["time"]])
        ACTION_MAX = np.array([50.0, 1.0, 0.5, 0.5])
        ACTION_MIN = np.array([-10.0, 0.0, 0.0, 0.1])
        normed = (action - ACTION_MIN) / (ACTION_MAX - ACTION_MIN)
        return normed

    def get_next_action(self, current_state, **kwargs):
        src_tensor = self._to_tensor(current_state)
        target_tensor = self._to_tensor(self.goal_state)
        with torch.no_grad():
            actions = self.planner.get_action(
                src_tensor,
                target_tensor,
                u_min=torch.zeros(self.action_dim).float(),
                u_max=torch.ones(self.action_dim).float(),
                opt_params=None,
                method="hj-prox",
                # method="cem",
            )
        # unnormalized_actions = [self._unnormalize_action(action) for action in actions]
        return self._unnormalize_action(actions[0])


class RandomShootingPlanner:
    def __init__(self, latent_dim, action_dim, encoder_model, forward_model, **kwargs):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.encoder = encoder_model
        self.forward = forward_model

    def get_action(self, init_obs, target_obs, num_actions=10000, **kwargs):
        init_z = self.encoder(init_obs)
        target_z = self.encoder(target_obs)
        init_z = init_z.repeat(num_actions, 1)
        target_z = target_z.repeat(num_actions, 1)

        # generate random actions
        random_actions = self.generate_random_actions(num_actions)
        random_actions = torch.FloatTensor(random_actions).to(init_z.device)
        pred_z = self.forward(init_z, random_actions)

        # find closest action
        dist = F.mse_loss(pred_z, target_z, reduction="none")
        dist = dist.mean(dim=1)
        index = torch.argmin(dist)

        # print debugging
        a = random_actions[:10].cpu().numpy()
        d = dist[:10].cpu().numpy()
        print(f"Random actions: {a}")
        print(f"Distances: {d}")

        action = random_actions[index].unsqueeze(0).cpu().numpy()
        print(f"Chosen action: {action}")
        print(f"Distance: {dist[index].cpu().numpy()}")

        return action

    def generate_random_actions(self, n):
        ANGLES = [-10, 0, 10, 20, 30, 40, 50]
        DIRECTIONS = [0.0, 1.0]  # ["forward", "reverse"]
        SPEEDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        # TIMES = [0.1, 0.2, 0.3, 0.4, 0.5]
        TIMES = [0.2, 0.3, 0.4, 0.5]
        ACTION_MAX = np.array([50.0, 1.0, 0.5, 0.5])
        ACTION_MIN = np.array([-10.0, 0.0, 0.0, 0.1])

        actions = []
        for i in range(n):
            rand_angle = random.choice(ANGLES)
            rand_dir = random.choice(DIRECTIONS)
            rand_speed = random.choice(SPEEDS)
            rand_time = random.choice(TIMES)
            action = np.array([rand_angle, rand_dir, rand_speed, rand_time])
            normed = (action - ACTION_MIN) / (ACTION_MAX - ACTION_MIN)
            actions.append(normed)

        return np.array(actions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("start_state_dir", type=str)
    parser.add_argument("goal_state_dir", type=str)
    parser.add_argument("model_path", type=str, nargs="?", default=os.path.join(os.getcwd(), "models"))
    args = parser.parse_args()

    num_cameras = 3
    num_external_cameras = num_cameras - 1

    print("Loading states...")
    start_state = CarState()
    start_state.load_from_files(args.start_state_dir, num_external_cameras)
    goal_state = CarState()
    goal_state.load_from_files(args.goal_state_dir, num_external_cameras)

    print("Loading planner...")
    planner = LatentForwardInference(goal_state, args.model_path)

    print("Predicting action...")
    action = planner.get_next_action(start_state)

    print("Action:")
    print(action)
