from dijkstra import make_ls, DP_goals
import torch

import heapq

def dijkstra_shortest_path(graph, start, target):
    """
    Find the shortest path from a given start node to a target node in a weighted graph using Dijkstra's algorithm.

    Parameters:
        - graph (dict): a dictionary representing the graph, where the keys are the nodes and the values are dictionaries
          of the form {neighbour_node: weight}.
        - start (str): the starting node from which to calculate the shortest path.
        - target (str): the target node to which to calculate the shortest path.

    Returns:
        - path (list): a list containing the nodes in the shortest path from the start node to the target node.
          If there is no path from the start node to the target node, an empty list is returned.
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    heap = [(0, start)]
    predecessors = {node: None for node in graph}

    while heap:
        current_distance, current_node = heapq.heappop(heap)

        if current_distance > distances[current_node]:
            continue

        if current_node == target:
            break

        for neighbour, weight in graph[current_node].items():
            distance = current_distance + weight

            if distance < distances[neighbour]:
                distances[neighbour] = distance
                predecessors[neighbour] = current_node
                heapq.heappush(heap, (distance, neighbour))

    path = []
    while target is not None:
        path.insert(0, target)
        target = predecessors[target]

    if path[0] == start:
        return path
    else:
        return []

class Dijkstra_Planner:
    def __init__(self, empirical_mdp) -> None:
        self.empirical_mdp = empirical_mdp
        transition = empirical_mdp.discrete_transition
        transition_tensor = torch.from_numpy(transition)
        adj_mat = transition_tensor.sum(dim = 1)
        edges = torch.nonzero(adj_mat)

        graph = dict.fromkeys(range(len(empirical_mdp.unique_states)))
        for i in range(len(empirical_mdp.unique_states)):
            ind = edges[edges[:,0]==i][:,1].tolist()
            # remove self-loop
            if i in ind:
                ind.remove(i)
            graph[i] = dict(zip(ind, [1]*len(ind)))

        self.graph = graph

    def step(self, current_state, goal_state):
        path = dijkstra_shortest_path(self.graph, current_state, goal_state)
        return path[1]

# class Dijkstra_Planner:
#     def __init__(self, empirical_mdp) -> None:
#         self.empirical_mdp = empirical_mdp
#         num_states, num_actions, _ = empirical_mdp.discrete_transition.shape
#         ls, _ = make_ls(torch.Tensor(empirical_mdp.discrete_transition), num_states, num_actions)
#         self.ls = ls

#     def step(self, current_state, goal_state, dp_step_use = 1):
#         ls = self.ls
#         distance_to_goal, g, step_action_idx = DP_goals(ls, init_state=current_state, goal_index=goal_state,
#                                                             dp_step=dp_step_use, code2ground={})
#         next_mdp_state = self.empirical_mdp.step(current_state, step_action_idx)
#         return next_mdp_state
