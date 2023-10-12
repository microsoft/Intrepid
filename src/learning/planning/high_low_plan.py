from .high_level_planner.dijkstra_planner import Dijkstra_Planner
from .hj_prox.hj_prox_alg import Tracking_Cost, HJ_Prox_Optimizer
import torch
from .cem.cem_optimizer import CEM_Optimizer


class TrajOptParams:
    def __init__(self) -> None:
        self.t_param = 0.1
        self.n_batch = 10
        self.horizon = 3

        # number of iterations for gradient descent or CEM iteration
        self.num_iter = 10

        # number of samples used in CEM
        self.num_samples = 500


class HighLowPlanner:
    def __init__(self, nz, nu, enc, forward_dyn, kmeans, MDP) -> None:
        # latent state dimension
        self.nz = nz
        # control input dimension
        self.nu = nu

        self.enc = enc
        self.forward_dyn = forward_dyn

        self.kmeans = kmeans
        self.MDP = MDP
        self.high_level_planner = Dijkstra_Planner(MDP)

        self.n_batch = 10
        self.horizon = 3

    def get_action(self, init_obs, target_obs, u_min=None, u_max=None, opt_params=None, method="hj-prox"):
        if opt_params is None:
            opt_params = TrajOptParams()

        # u_min, u_max: tensor of size (nu,), should be given entrywise
        device = init_obs.device

        # extract optimization parameters
        t_param = opt_params.t_param
        num_iter = opt_params.num_iter
        n_batch, horizon = opt_params.n_batch, opt_params.horizon

        init_lat_state = self.enc(init_obs)
        target_lat_state = self.enc(target_obs)

        kmeans = self.kmeans
        high_level_planner = self.high_level_planner

        # find the MDP state of the initial and target observation
        cur_mdp_state = kmeans.predict(init_lat_state.detach().cpu())[0]
        target_mdp_state = kmeans.predict(target_lat_state.detach().cpu())[0]

        if cur_mdp_state != target_mdp_state:
            next_mdp_state = high_level_planner.step(cur_mdp_state, target_mdp_state)
            next_lat_state = kmeans.cluster_centers_[next_mdp_state]
        else:
            next_mdp_state = target_mdp_state
            next_lat_state = target_lat_state

        nu = self.nu
        init_actions = u_min + (u_max - u_min) * torch.rand((n_batch * horizon, nu)).to(device)
        init_actions = init_actions.view((n_batch, horizon, nu))

        tracking_cost_fcn = Tracking_Cost(self.forward_dyn, init_lat_state, next_lat_state)

        if method == "hj-prox":
            hj_optimizer = HJ_Prox_Optimizer(tracking_cost_fcn, init_actions, t_param, x_min=u_min, x_max=u_max)
            output_action, action_list = hj_optimizer.prox_grad_descent(iter_num=num_iter, x_init=init_actions, t=t_param)

        if method == "cem":
            cem_optimizer = CEM_Optimizer(tracking_cost_fcn, x_min=u_min, x_max=u_max)
            output_action, _ = cem_optimizer.cem_iter(init_actions, num_samples=opt_params.num_samples, num_iter=num_iter)

        rollout_costs = tracking_cost_fcn(output_action)
        selected_action = output_action[rollout_costs[0].argmin().item()]

        # action = selected_action[0:1, :]

        return selected_action
