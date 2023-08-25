from learning.policy_search.fqi import FQI
from learning.policy_search.psdp import PSDP
from learning.policy_search.greedy_policy_search import GreedyPolicySearch
from learning.policy_search.path_policy_search import PathPolicySearch


class PolicySearchWrapper:
    def __init__(self):
        pass

    @staticmethod
    def generate_policy_search(policy_search_type, config, constants):
        if policy_search_type == "psdp":
            # Works for any setting. O(H^2) execution in horizon H.
            return PSDP(config, constants)

        elif policy_search_type == "fqi":
            # Works for any setting. O(H) execution in horizon H.
            return FQI(config, constants)

        elif policy_search_type == "gps":
            # Can work for stochastic problems provided a greedy optimization works. O(1) execution.
            return GreedyPolicySearch(config, constants)

        elif policy_search_type == "pps":
            # Designed for deterministic policies. O(1) execution and fastesr than GreedyPolicySearch
            return PathPolicySearch(config, constants)

        else:
            raise AssertionError("Unhandled policy search type %r" % policy_search_type)

    @staticmethod
    def get_filtered_data(policy_cover_dataset, horizon, policy_search_routine):
        """
            It is possible for various policy search routines to use existing data in replay memory. However,
            different policy methods use part of the replay memory and some may use nothing at all. Further,
            transfering the whole replay memory to asynchronous processes can take memory therefore, we filter
            the replay memory for every process by policy_search_routine.

        :param policy_cover_dataset: A dataset of observed transitions (dictionary indexed by timestep)
        :param horizon: Horizon of the policy search procedure
        :param policy_search_routine: type of policy search routine
        :return: filtered dataset
        """

        # TODO put replay memory on shared memory and pass it each process and therefore, avoid filtering.

        # GPS can reuse part of the collected dataset which represents real transition.
        if isinstance(policy_search_routine, GreedyPolicySearch):
            filtered_dataset = policy_cover_dataset[horizon]  # Only data for last time step

        elif isinstance(policy_search_routine, PathPolicySearch):
            filtered_dataset = policy_cover_dataset[horizon]  # Only data for last time step

        elif isinstance(policy_search_routine, FQI):
            filtered_dataset = policy_cover_dataset  # Utilizes data for each time step

        elif isinstance(policy_search_routine, PSDP):
            filtered_dataset = []  # Utilizes no data at all

        else:
            raise AssertionError("Failed to find policy")

        return filtered_dataset
