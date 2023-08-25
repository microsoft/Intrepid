class AverageUtil:
    def __init__(self, init_val=None):
        if init_val is None:
            self._sum_val = 0.0
            self._num_items = 0
        else:
            self._sum_val = init_val
            self._num_items = 1

    def get_num_items(self):
        return self._num_items

    def acc(self, val):
        self._sum_val += val
        self._num_items += 1

    def get_mean(self):
        return self._sum_val / float(max(1, self._num_items))

    def __str__(self):
        return "%f (count: %d)" % (self.get_mean(), self._num_items)
