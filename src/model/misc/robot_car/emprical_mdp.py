import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


class EmpiricalMDP:
    def __init__(self, state, action, next_state, reward, action_min, action_max, action_discrete_interval):
        assert len(action_min) == len(action_max) == len(action_discrete_interval) == action.shape[1]
        self.unique_states = sorted(np.unique(np.concatenate((state, next_state), axis=0)))
        self.unique_states_dict = {k: i for i, k in enumerate(self.unique_states)}
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.transition = self.__estimate_transition()
        self._action_max = action_max
        self._action_min = action_min
        self._action_discrete_interval = action_discrete_interval

        self.discrete_transition = self.__discrete_transition(action_min, action_max, action_discrete_interval)

    def __discrete_transition(self, action_min, action_max, action_discrete_interval):
        # discretize actions
        actions = []

        def _discretization(_min, _max, _discrete_interval, action_val):
            for val in np.arange(_min[0], _max[0] + _discrete_interval[0], _discrete_interval[0]):
                if len(_min) == len(_max) == 1:
                    actions.append((*action_val, round(val, 2)))
                else:
                    _discretization(_min[1:], _max[1:], _discrete_interval[1:], (*action_val, round(val, 2)))

        _discretization(action_min, action_max, action_discrete_interval, ())
        actions = np.array(sorted(actions))
        self.discrete_action_space = np.unique(actions, axis=0)

        # generate discrete transition matrix containing visit count
        action_value_idx_map = {tuple(val): idx for idx, val in enumerate(self.discrete_action_space)}
        transition = np.zeros((len(self.unique_states), len(self.discrete_action_space), len(self.unique_states)))
        for state in tqdm(range(len(self.transition)), desc="state :"):
            for next_state, action in tqdm(enumerate(self.transition[state]), desc="next-state :"):
                if not np.isnan(action).all():
                    transition[state][action_value_idx_map[tuple(self.find_closest_value(action, actions))]][next_state] += 1
                    # transition[state][action_value_idx_map[tuple(np.round(action, 2))]][next_state] += 1

        return transition

    def __estimate_transition(self):
        transition = np.empty((len(self.unique_states), len(self.unique_states), len(self.action[0])))
        transition[:, :, :] = np.nan
        for state in tqdm(self.unique_states, desc="unique"):
            # threshold off spurious MDP transitions
            # sample_num = sum(self.state == state)
            trans_sample_num = sum(np.logical_and(self.state == state, self.next_state != state))

            for next_state in self.unique_states:
                _filter = np.logical_and(self.state == state, self.next_state == next_state)
                if True in _filter and sum(_filter) > 0.1 * trans_sample_num:
                    transition[self.unique_states_dict[state], self.unique_states_dict[next_state], :] = self.action[
                        _filter
                    ].mean(axis=0)
        return transition

    def visualize_transition(self, save_path=None):
        graph = nx.DiGraph()
        edges = []
        for state in self.unique_states:
            for next_state in self.unique_states:
                if not np.isnan(self.transition[self.unique_states_dict[state], self.unique_states_dict[next_state], 0]):
                    edges.append((state, next_state))

        graph.add_edges_from(edges)
        nx.draw(graph, with_labels=True)
        if save_path is not None:
            plt.savefig(save_path)
        return graph

    def visualize_path(self, path, save_path=None):
        graph = nx.DiGraph()
        edges = []
        for state in self.unique_states:
            for next_state in self.unique_states:
                if not np.isnan(self.transition[self.unique_states_dict[state], self.unique_states_dict[next_state], 0]):
                    edges.append((state, next_state))

        graph.add_edges_from(edges)
        nx.draw(graph, with_labels=True)
        if save_path is not None:
            plt.savefig(save_path)
        return graph

    def step(self, state, action_idx):
        """samples a next state from current state and action"""
        next_state_visit_count = self.discrete_transition[state][action_idx]
        next_state_prob = self.__normalize(next_state_visit_count, next_state_visit_count.min(), next_state_visit_count.max())
        next_state_sample = np.random.choice(np.arange(0, len(next_state_visit_count)), 1, replace=False, p=next_state_prob)

        return next_state_sample[0]

    @staticmethod
    def __normalize(arr, t_min, t_max):
        norm_arr = []
        diff = t_max - t_min
        diff_arr = max(arr) - min(arr)
        for i in arr:
            temp = (((i - min(arr)) * diff) / diff_arr) + t_min
            norm_arr.append(temp)
        return norm_arr

    @staticmethod
    def find_closest_value(target, arrays):

        # find index where target would need to be inserted
        index = np.searchsorted([np.linalg.norm(target - a) for a in arrays], 0)

        # check if target is already in the list
        if index == 0:
            closest = arrays[0]
        elif index == len(arrays):
            closest = arrays[-1]
        else:
            before = arrays[index - 1]
            after = arrays[index]
            if np.linalg.norm(target - before) < np.linalg.norm(after - target):
                closest = before
            else:
                closest = after
        return closest
