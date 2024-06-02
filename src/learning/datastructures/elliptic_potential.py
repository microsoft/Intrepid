import torch


class EllipticPotential:

    def __init__(self, lam=1.0):
        """
            A function to keep track of the matrix
               Lambda =  lambda I + \sum_{i=1}^0 v_i v_i^T
               and return Lambda^{-1} and det(Lambda^{-1}) efficiently

            Supports features in both numpy and torch format.
        """
        self.lam = lam
        self._inv_matrix = None
        self._det_inv_matrix = None

    def reset(self):
        self._inv_matrix = None
        self._det_inv_matrix = None

    def add_feature(self, feature):
        """
        :param feature: A torch tensor or a numpy ndarray of 1-d or 2-d (of type 1xd or dx1)
            Sherman Woodbury Morrison update
            (A + uv^T)^{-1} = A^{-1} - (A^-1 uv^T A^-1) / (1 + v^T A^-1 u)
        """

        if feature.ndim == 1:
            feature = feature.unsqueeze(0)

        elif feature.ndim == 2:
            pass        # TODO
        else:
            raise AssertionError("Feature dimension must be either 1-D or 2D of type 1xd or dx1")

        if self._inv_matrix is None:
            self._inv_matrix = (1.0 / self.lam) * torch.eye(feature.shape[0])

        rho = 1.0 / (1.0 + feature.T @ self._inv_matrix @ feature)
        self._inv_matrix = self._inv_matrix - (self._inv_matrix @ feature @ feature.T @ self._inv_matrix) * rho

    def get_inverse(self):
        return self._inv_matrix

    def get_inv_mat_det(self):
        return self._det_inv_matrix

    def get_elliptic_bonus(self, features):
        """
            :param features: Given a feature of size either dim or batch x dim
            :return: Bonus which is either scalar if input is 1-d or batch if 2-d
        """

        bonus = torch.sqrt(
            torch.diagonal(
                features @ self._inv_matrix @ features.T
            ))

        if features.ndim == 1:
            bonus = bonus[0]

        return bonus


