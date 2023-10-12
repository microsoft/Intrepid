import os, sys
import torch
import torch.nn as nn
import pickle
import numpy as np
# from autograd import grad
import matplotlib.pyplot as plt
import time


class HJ_Prox_Optimizer:
    def __init__(self, f, x_init, t, x_min=None, x_max=None) -> None:
        self.f, self.x_init, self.t = f, x_init, t
        self.x = x_init

        self.x_min, self.x_max = x_min, x_max

    def set_t_value(self, t):
        self.t = t

    def step(self, x=None, t=None, opt=None):
        if x is None:
            x = self.x

        if t is None:
            t = self.t

        prox = compute_hj_prox(x, t, self.f)
        return prox

    def grad_descent(self, iter_num=10, x_init=None, t=None):
        # N: number of steps in gradient descent
        if x_init is None:
            x_init = self.x_init

        x = x_init

        if t is None:
            t = self.t

        t0 = t

        sol_list = [x_init]

        for i in range(iter_num):
            start_time = time.time()

            # schedule the stepsize t
            t = t0 / (i + 1)

            prox = self.step(x, t)
            run_time = time.time() - start_time

            cost = self.f(x)
            # print('grad desc. iter {:d} takes {:.2f} secs. cost min: {:.2f}, cost max: {:.2f}'.format(i, run_time, cost.min().item(), cost.max().item()))

            if self.x_min is not None:
                prox = torch.clamp(prox, min=self.x_min)

            if self.x_max is not None:
                prox = torch.clamp(prox, max=self.x_max)

            x = prox

            sol_list.append(x)

        return x, sol_list

    def prox_grad_descent(self, iter_num=10, x_init=None, t=None, beta=1.0):
        if x_init is None:
            x_init = self.x_init

        x = x_init

        if t is None:
            t = self.t
        t0 = t

        sol_list = [x_init]

        for i in range(iter_num):
            start_time = time.time()

            t = t0 / (i + 1)

            # schedule the stepsize t
            gamma = 2 * (1 - t / (beta + t))

            prox = self.step(x, t)
            run_time = time.time() - start_time

            cost = self.f(x)
            # print('grad desc. iter {:d} takes {:.2f} secs. cost min: {:.2f}, cost max: {:.2f}'.format(i, run_time, cost.min().item(), cost.max().item()))

            next_x = x + gamma * (prox - x)

            if self.x_min is not None:
                next_x = torch.clamp(next_x, min=self.x_min)

            if self.x_max is not None:
                next_x = torch.clamp(next_x, max=self.x_max)

            x = next_x

            sol_list.append(x)

        return x, sol_list


class Tracking_Cost(nn.Module):
    def __init__(self, forward_model, lat_state_init, lat_state_target) -> None:
        super().__init__()
        self.forward_model = forward_model
        self.lat_state_init = lat_state_init
        self.lat_state_target = lat_state_target

    def forward(self, actions):
        cost = tracking_cost_helper_batch(actions, self.forward_model, self.lat_state_init, self.lat_state_target)
        return cost


def tracking_cost_helper_batch(actions, forward_model, lat_state_init, lat_state_target):
    # SC: add codes to adjust dimensions of actions
    if actions.dim() == 3:
        actions = actions.unsqueeze(0)

    num_samples, n_batch, mpc_T, nu = actions.size()
    nz = lat_state_init.size(-1)

    total_cost = 0
    device = lat_state_init.device

    # lat_state_target = lat_state_target.repeat((n_batch, 1)).repeat((num_samples, 1, 1))
    lat_state = lat_state_init.repeat((n_batch, 1)).repeat((num_samples, 1, 1))

    gamma = 1.0
    for i in range(mpc_T):
        output = forward_model(lat_state.view(-1, nz), actions[:, :, i, :].view(-1, nu).float())
        output = output.view(lat_state.size())
        lat_state = output
        diff = output - lat_state_target
        running_cost = torch.norm(diff.view(-1, nz), p=2, dim=1)
        running_cost = running_cost.view((num_samples, n_batch))
        total_cost = total_cost + gamma ** (mpc_T - i - 1) * running_cost
    # only use the terminal cost
    # total_cost = running_cost
    return total_cost.detach() / mpc_T


def tracking_cost_helper(actions, forward_model, lat_state_init, lat_state_target):
    n_batch = actions.size(0)
    N = actions.size(1)

    device = lat_state_init.device

    traj = torch.zeros((n_batch, N + 1, lat_state_init.size(1))).to(device)
    traj[0, 0, :] = torch.clone(lat_state_init)

    lat_state = lat_state_init
    costs = []
    for i in range(N):
        output = forward_model(lat_state, actions[:, i, :])
        lat_state = output
        traj[0, i + 1, :] = torch.clone(output)
        diff = output - lat_state_target
        running_cost = torch.norm(diff, p=2)
        costs.append(running_cost)
    cost = torch.tensor(costs).sum() / N
    return cost


def compute_hj_prox(x, t, f, delta=1e-1, int_samples=1000, alpha=0.1,
                    recursion_depth=0, alpha_decay=0.5, tol=1.0e-9,
                    tol_underflow=0.9, device='cpu', verbose=False,
                    return_samples=False):
    """ Estimate proximals from function value sampling via HJ-Prox Algorithm.

        Args:
            x (tensor): Input vector
            t (tensor): Time > 0
            f: Function to minimize

        Returns:
            tensor: Estimate of the proximal of f at x

        Reference:
            [A Hamilton-Jacobi-based Proximal Operator](https://arxiv.org/pdf/2211.12997.pdf)
    """
    # valid_tensor_shape = x.shape[1] == 1 and x.shape[0] >= 1
    # assert valid_tensor_shape, "Input tensor shape incorrect."

    # SC: not sure if it is necessary to run the codes in cpu
    device = x.device

    start_time = time.time()

    recursion_depth += 1
    std_dev = np.sqrt(delta * t / alpha)
    n_batch = x.shape[0]

    y = std_dev * torch.randn(tuple([int_samples] + list(x.size())), device=device)
    y = y + x
    z = -f(y) * (alpha / delta)

    underflow = torch.exp(z) <= tol
    underflow_freq = float(underflow.sum()) / underflow.shape[0]
    observe_underflow = underflow_freq > tol_underflow

    run_time = time.time() - start_time
    # print(f'HJ prox first phase runtime: {run_time} s.')

    # try to mute observe underflow
    # observe_underflow = False

    if observe_underflow:
        alpha *= alpha_decay
        return compute_hj_prox(x, t, f, delta=delta, int_samples=int_samples,
                               alpha=alpha, recursion_depth=recursion_depth,
                               alpha_decay=alpha_decay, tol=tol,
                               tol_underflow=tol_underflow, device=device,
                               verbose=verbose, return_samples=return_samples)
    else:
        start_time = time.time()

        soft_max = torch.nn.Softmax(dim=0)

        z = z.to(device)

        for _ in range(2):
            z = z.unsqueeze(-1)

        HJ_prox = torch.mul(soft_max(z), y).sum(dim=0)

        run_time = time.time() - start_time
        # print(f'HJ prox second phase runtime: {run_time} s.')

        valid_prox_shape = HJ_prox.shape == x.shape
        assert valid_prox_shape

        prox_is_finite = (HJ_prox < np.inf).all()
        assert prox_is_finite

        if verbose:
            envelope = - (delta / alpha) * torch.log(torch.mean(torch.exp(z)))
            return HJ_prox, recursion_depth, envelope
        elif return_samples:
            return HJ_prox, y, alpha
        else:
            return HJ_prox