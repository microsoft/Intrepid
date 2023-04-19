import time
import numpy as np

from utils.beautify_time import beautify


class RicattiSolver:

    def __init__(self, logger, max_it=1000, min_change=0.000001):

        self.max_it = max_it
        self.logger = logger
        self.min_change = min_change

    def solve(self, A, B, Q, R):

        time_start = time.time()
        self.logger.debug("Performing Ricatti Iterations")
        P = np.eye(A.shape[0])

        for it in range(0, self.max_it):

            inv_term = np.linalg.inv(R + B.T @ P @ B)
            new_P = A.T @ P @ A + Q - A.T @ P @ B @ inv_term @ B.T @ P @ A

            change = np.linalg.norm(P - new_P)

            if it % 10 == 0:
                self.logger.debug("Ricatti Solver: Iteration=%d, Change in P %f" % (it, change))

            P = new_P

            if change < self.min_change:
                break

        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        self.logger.debug("Ricatti Iterations Performed. Time taken %s" % beautify(time.time() - time_start))

        return P, K
