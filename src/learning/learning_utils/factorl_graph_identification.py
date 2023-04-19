from learning.learning_utils.independence_test import IndependenceTest


class FactoRLGraphIdentification:

    def __init__(self, config, constants):
        self.num_atoms = config["obs_dim"]
        assert self.num_atoms > 0, "observation dimension should be positive but found %d" % self.num_atoms
        self.ind_test = IndependenceTest(config, constants)

    def get_factors(self, dataset, logger, tensorboard):

        # Here we run a more efficient but less correct version of independence test, where we use data without
        # fixing previous state and actions. This means some amount of dependence can seep between two different
        # factors. This will not work in many cases, and there, one must roll-in with different policies and actions
        # and try independence test for each (u, v).

        factors = []
        queue = list(range(0, self.num_atoms))

        while queue:

            u = queue.pop()
            factor = [u]
            new_queue = []
            for v in queue:

                pair_dataset = [(dp.next_obs[u], dp.next_obs[v]) for dp in dataset]

                ind_test_ = self.ind_test.is_independent(pair_dataset, logger, tensorboard)
                if ind_test_:
                    # If u and v are independent then keep v separate
                    new_queue.append(v)
                else:
                    # If u and v are dependent then we can merge them into the same factor
                    factor.append(v)

            queue = new_queue

            factors.append(factor)

        return factors

    @staticmethod
    def test_factors(factors_gold, factors_inferred):

        hashed_factors_gold = dict()
        for ix, factor in enumerate(factors_gold):
            factor_key = tuple(sorted(factor))
            assert factor_key not in hashed_factors_gold
            hashed_factors_gold[factor_key] = ix

        hashed_factors_inferred = dict()
        for ix, factor in enumerate(factors_inferred):
            factor_key = tuple(sorted(factor))
            assert factor_key not in hashed_factors_inferred
            hashed_factors_inferred[factor_key] = ix

        infer_to_gold_map = dict()
        found = True

        for factor_key, inferred_ix in hashed_factors_inferred.items():

            if factor_key in hashed_factors_gold:
                infer_to_gold_map[inferred_ix] = hashed_factors_gold[factor_key]
            else:
                found = False
                infer_to_gold_map[inferred_ix] = None

        return found, infer_to_gold_map
