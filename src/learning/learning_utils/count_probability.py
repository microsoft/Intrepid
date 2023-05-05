from utils.generic_policy import sample_action_from_prob


class CountProbability:
    """
    A simple class to estimate probabilities based on counts
    """

    def __init__(self):
        self._total_count = 0
        self._values = {}

    def add(self, entry):
        self._total_count += 1

        if entry in self._values:
            self._values[entry] += 1.0
        else:
            self._values[entry] = 1.0

    def get_probability(self):
        z = float(max(1.0, self._total_count))
        prob = [(key, value / z) for (key, value) in sorted(self._values.items())]

        return prob

    def get_probability_dict(self):
        z = float(max(1.0, self._total_count))
        return {key: value / z for (key, value) in self._values.items()}

    def get_prob_entry(self, entry):
        if entry in self._values:
            return self._values[entry] / float(max(1.0, self._total_count))
        else:
            return 0.0

    def sample(self):
        key_prob = self.get_probability()
        prob = [key_prob_[1] for key_prob_ in key_prob]
        ix = sample_action_from_prob(prob)

        return key_prob[ix][0]

    def total_count(self):
        return self._total_count

    def get_entries(self):
        return self._values.keys()

    def total_variation(self, other_prob):
        union_keys = set(self._values.keys()).union(set(other_prob._values.keys()))

        tv = 0.0
        for key in union_keys:
            tv += abs(self.get_prob_entry(key) - other_prob.get_prob_entry(key))

        return 0.5 * tv

    def __str__(self):
        prob = self.get_probability()

        return "{%s}" % (
            "; ".join(["%r: %f" % (entry_, prob_) for entry_, prob_ in prob])
        )
