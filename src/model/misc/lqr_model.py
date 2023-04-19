class LQRModel:

    def __init__(self, A, B, Q, R, Sigma_W, Sigma_0):
        """
            LQR model describes a simple continuous control dynamics where
            the state s evolves as

            s_1 ~ N(0, Sigma_0)
            s_{t+1} = A s_t + B u_t + epsilon_t, for all t
            epsilon_t ~ N(0, Sigma_W)

            where s_t and u_t is the state and action respectively at time step t

            cost at time step t is given by s_t^T Q s_t + u_t^T R u_t
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Sigma_W = Sigma_W
        self.Sigma_0 = Sigma_0

    def copy(self):

        return LQRModel(
            A=self.A.copy(),
            B=self.B.copy(),
            Q=self.Q.copy(),
            R=self.R.copy(),
            Sigma_W=self.Sigma_W.copy(),
            Sigma_0=self.Sigma_0.copy()
        )