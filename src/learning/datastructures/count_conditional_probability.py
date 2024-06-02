from learning.datastructures.count_probability import CountProbability


class CountConditionalProbability:
    """
    A simple class to estimate conditional probabilities based on counts
    """

    def __init__(self):
        self._conditions = {}

    def add(self, entry, condition):
        if condition not in self._conditions:
            self._conditions[condition] = CountProbability()

        self._conditions[condition].add(entry)

    def get_conditions(self):
        return self._conditions

    def get_entry(self, condition):
        if condition not in self._conditions:
            return None
        else:
            return self._conditions[condition]

    def total_count(self, condition):
        if condition not in self._conditions:
            return 0
        else:
            return self._conditions[condition].total_count()

    def get_prob_entry(self, entry, condition):
        if condition not in self._conditions:
            return None
        else:
            return self._conditions[condition].get_prob_entry(entry)

    def __str__(self):
        return "{%s}" % (
            "; ".join(["%r -> %s" % (condition, str(prob)) for (condition, prob) in sorted(self._conditions.items())])
        )
