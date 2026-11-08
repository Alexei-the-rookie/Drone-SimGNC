import numpy as np
from quadfc.params import QuadParams

class MixerController:
    """
    A mixer controller inputed (f,tau) and output omega_cmd(4,) in rad/s.
    """
    name = "mixer"
    def __init__(self, params: QuadParams):
        self.params = params
        self.saturated = False
        self.B =self.params.allocation
        self.B_inv = np.linalg.inv(self.B)
        self.w2_max = self.params.omega_max ** 2


    def step(self, f: float, tau: np.ndarray) -> np.ndarray:
        """
        Compute the motor speed command (rad/s) for the next interval.
        """
        w2 = self.B_inv @ np.array([f, *tau])
        # scaling...
        if w2.max() > self.w2_max or w2.min() < 0.0:
            w2 = self._scale(w2)
            self.saturated = True
        else:
            self.saturated = False
        return np.sqrt(np.clip(w2, 0.0, self.w2_max))

    def _scale(self, w2):
        # scale down w2 to fit in [0, w2_max]
        # keep the mean of w2 the same, but scale the difference.
        mean = float(np.clip(w2.mean(), 0.0, self.w2_max))
        diff = w2 - w2.mean()
        lam = 1.0
        for diff_i in diff:
            if diff_i > 1e-9:
                lam = min(lam, (self.w2_max - mean) / diff_i)
            elif diff_i < -1e-9:
                lam = min(lam, mean / (-diff_i))
        return mean + lam * diff