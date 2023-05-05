class EntropyDecayPolicy:
    def __init__(self, constants, epoch):
        self.epoch = epoch

        if constants["entropy_policy"] == "fix":
            self.decay_fn = self._fixed_entropy_coeff
        elif constants["entropy_policy"] == "none":
            self.decay_fn = self._no_entropy
        elif constants["entropy_policy"] == "step":
            self.decay_fn = self._step_decayed_entropy_coeff
        elif constants["entropy_policy"] == "decay":
            self.decay_fn = self._decay_entropy_coeff
        elif constants["entropy_policy"] == "linear":
            self.decay_fn = self._linear_decay
        elif constants["entropy_policy"] == "pos-neg":
            self.decay_fn = self._step_decrease
        elif constants["entropy_policy"] == "smart":
            self.decay_fn = self._smart
            self.current = dict()
            self.current[1] = 1.0
        else:
            raise AssertionError(
                "Unhandled entropy policy %r " % constants["entropy_policy"]
            )

    def get_entropy_coeff(self, epoch, test_set_errors, past_entropy):
        return self.decay_fn(epoch, test_set_errors, past_entropy)

    @staticmethod
    def _fixed_entropy_coeff(epoch, test_set_errors, past_entropy):
        return 1.0

    @staticmethod
    def _linear_decay(epoch, test_set_errors, past_entropy):
        return 1.0 / float(max(1, epoch))

    @staticmethod
    def _no_entropy(epoch, test_set_errors, past_entropy):
        return 0.0

    def _step_decayed_entropy_coeff(self, epoch, test_set_errors, past_entropy):
        return (
            1
            if epoch <= self.epoch // 2 + 1
            else (1 - (2 * epoch - self.epoch) / float(max(1.0, self.epoch)))
        )

    def _decay_entropy_coeff(self, epoch, test_set_errors, past_entropy):
        return 1 - epoch / float(max(1.0, self.epoch))

    def _smart(self, epoch, test_set_errors, past_entropy):
        # If test set loss is not decreasing and entropy is high then decrease by 0.1
        # If test set loss is not increasing and entropy is low then increase by 0.1

        if epoch in self.current:
            return self.current[epoch]

        current = self.current[epoch - 1]

        if len(test_set_errors) <= 5:  # Too little information
            self.current[epoch] = current
        else:
            last_error = test_set_errors[-1]
            snd_last_error = test_set_errors[-2]

            last_entropy = past_entropy[-1]

            if last_error < snd_last_error - 0.001:
                pass  # Nothing to do
            elif last_error >= snd_last_error - 0.001:
                if last_entropy > 0.1:  # entropy is high. Reduce it
                    current = current - 0.1
                elif last_entropy < 0.1:  # entropy is low. Increase it
                    current = current + 0.1

        self.current[epoch] = current

        return current

    def _step_decrease(self, epoch, test_set_errors, past_entropy, step=10):
        if epoch <= step:  # Exploration Started
            return 1.0
        elif epoch <= 2 * step:
            return 0.5
        elif epoch <= 4 * step:
            return 0.25
        elif epoch <= 7 * step:
            return 0.125
        elif epoch <= 10 * step:
            return 0.0
        elif epoch <= 12 * step:  # Convergence started
            return -0.125
        elif epoch <= 14 * step:
            return -0.25
        elif epoch <= 15 * step:
            return -0.5
        else:
            return -1
