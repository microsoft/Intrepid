import pdb
import time
import scipy.optimize
import numpy as np


class LinearDisagModel:
    TOTAL_TIME = 0.0
    TOTAL_CALL = 0
    TOTAL_OPT_FAILURE = 0
    TOTAL_FALLBACK = 0
    NUMERICAL_ERROR = 1e-4

    SUCCEEDED, OPT_FAIL_FLAG, SAFE_ACTION = range(3)

    def __init__(self, safety_dataset, safe_action):
        A, b = self._create_disag_model(safety_dataset)
        self.A = A
        self.b = b
        self.safe_action = safe_action
        self.method = "highs-ds"  # "revised simplex"  # "highs-ds"  # "highs"
        self.fallback_method = "highs-ipm"  # "interior-point"  # "highs-ipm"

        self.total_time = 0
        self.total_call = 0
        self.total_warning = 0
        # print("scipy version is %r" % scipy.__version__)

    @staticmethod
    def load_from_file(fname, safe_action):
        disag_model = LinearDisagModel([], safe_action)

        data = np.load(fname)
        disag_model.A = data["A"]
        disag_model.b = data["b"]

        return disag_model

    @staticmethod
    def _create_disag_model(safety_dataset):
        """
        :param safety_dataset:
        :return:
        """

        # We need to satisfy the following:
        #   y_i(w^T \phi(x_i, a_i) + b) >= 0
        #
        # which is equivalent to y_i ([w; b]^T [\phi(x_i, a_i); 1]) >= 0
        #
        # or equivalently, [-y_i [\phi(x_i, a_i); 1]^T] \tilde{w} <= 0
        # where \tilde{w} = [w; b]
        # Note that constraints of type -1 <= \tilde{w} <= 1 are provided separately to the Lp solver.

        if len(safety_dataset) == 0:
            return None, None

        A = []
        b = []

        for safety_ftr, safety_label, _ in safety_dataset:
            if type(safety_label) == bool:
                safety_label = 1.0 if safety_label else -1.0

            A.append(-safety_label * np.append(safety_ftr, 1))
            b.append(0.0)

        A = np.vstack(A).astype(np.float32)
        b = np.array(b).astype(np.float32)

        return A, b

    def update(self, safety_ftr, safety_label):
        if type(safety_label) == bool:
            safety_label = 1.0 if safety_label else -1.0

        new_row = -safety_label * np.append(safety_ftr, 1)

        if self.A is None or self.b is None:
            self.A = new_row.reshape(1, -1)
            self.b = np.zeros(1).astype(np.float32)
        else:
            self.A = np.append(self.A, new_row.reshape(1, -1), axis=0)
            self.b = np.append(self.b, 0.0)

    def _get_max_min_val(self, safety_ftr, action, method=None):
        """
        :param safety_ftr: Safety features for state and action
        :param model: Current region of disagreement model
        :param mehtod: Method used to solve Lp. If none, then this.method is used
        :return: True if the (state, action) pair is in region of disagreement and False otherwise
        """

        if action == self.safe_action:
            # We know the safe action is safe
            return None, None, LinearDisagModel.SAFE_ACTION

        safety_ftr_bias = np.append(safety_ftr, 1)

        time_start = time.time()

        if method is None:
            method = self.method

        try:
            if self.A is None:
                min_val_res = scipy.optimize.linprog(
                    safety_ftr_bias, bounds=(-1, 1), method=method
                )

                max_val_res = scipy.optimize.linprog(
                    -safety_ftr_bias, bounds=(-1, 1), method=method
                )

            else:
                min_val_res = scipy.optimize.linprog(
                    safety_ftr_bias,
                    A_ub=self.A,
                    b_ub=self.b,
                    bounds=(-1, 1),
                    method=method,
                )

                max_val_res = scipy.optimize.linprog(
                    -safety_ftr_bias,
                    A_ub=self.A,
                    b_ub=self.b,
                    bounds=(-1, 1),
                    method=method,
                )

            LinearDisagModel.TOTAL_TIME += time.time() - time_start
            LinearDisagModel.TOTAL_CALL += 1

        except:
            if method != self.fallback_method:
                # Fall back to a more powerful but slower solver
                LinearDisagModel.TOTAL_FALLBACK += 1
                return self._get_max_min_val(
                    safety_ftr, action, method=self.fallback_method
                )
            else:
                LinearDisagModel.TOTAL_OPT_FAILURE += 1
                return None, None, LinearDisagModel.OPT_FAIL_FLAG

        # if self.total_call % 100 == 0:
        #     print("Mean time is %f and A has %d many rows" % (self.total_time / float(self.total_call),
        #                                                       self.A.shape[0]))

        if max_val_res.fun is None or min_val_res.fun is None:
            if method != self.fallback_method:
                # Fall back to a more powerful but slower solver
                LinearDisagModel.TOTAL_FALLBACK += 1
                return self._get_max_min_val(
                    safety_ftr, action, method=self.fallback_method
                )
            else:
                LinearDisagModel.TOTAL_OPT_FAILURE += 1
                return None, None, LinearDisagModel.OPT_FAIL_FLAG

        min_val = min_val_res.fun
        max_val = -max_val_res.fun

        return min_val, max_val, LinearDisagModel.SUCCEEDED

    def is_surely_safe(self, safety_ftr, action, method=None):
        """
        :param safety_ftr: Safety features for state and action
        :param model: Current region of disagreement model
        :param method: Method used to solve Lp. If none, then this.method is used
        :return: True if the (state, action) pair is surely safe, i.e., neither known to be unsafe
                 nor in region of disagreement
        """

        min_val, max_val, FLAG = self._get_max_min_val(safety_ftr, action, method)

        if FLAG == LinearDisagModel.SAFE_ACTION:
            return True
        elif FLAG == LinearDisagModel.OPT_FAIL_FLAG:
            return False
        else:
            return min_val >= -LinearDisagModel.NUMERICAL_ERROR

    def in_region_of_disag(self, safety_ftr, action, method=None):
        """
        :param safety_ftr: Safety features for state and action
        :param model: Current region of disagreement model
        :param mehtod: Method used to solve Lp. If none, then this.method is used
        :return: True if the (state, action) pair is in region of disagreement and False otherwise
        """

        min_val, max_val, FLAG = self._get_max_min_val(safety_ftr, action, method)

        if FLAG == LinearDisagModel.SAFE_ACTION:
            return False
        elif FLAG == LinearDisagModel.OPT_FAIL_FLAG:
            return True
        else:
            if min_val >= -LinearDisagModel.NUMERICAL_ERROR:
                return False
            elif max_val <= LinearDisagModel.NUMERICAL_ERROR:
                return False
            else:
                return True

    def test(self, dataset):
        num_safe = 0.0
        num_marked_safe = 0.0

        for safety_ftr, safety_label, action in dataset:
            in_rd = self.in_region_of_disag(safety_ftr, action)

            if not in_rd and not safety_label:
                pdb.set_trace()
                raise AssertionError(
                    "Model Predicts Unsafe Features to be outside Region of Disagreement (RD)"
                )

            if safety_label:
                num_safe += 1
                if not in_rd:
                    num_marked_safe += 1

        pct = (num_marked_safe * 100.0) / float(max(1, num_safe))
        return num_safe, pct
