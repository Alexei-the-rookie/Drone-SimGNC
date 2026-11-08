import numpy as np
from quadfc.params import QuadParams

class RateController:
    """
    A rate controller inputed (omega, omega_cmd) and output tau(3,) in N*m.
    """
    name = "rate"
    def __init__(self, params: QuadParams):
        self.params = params
        self.kp = np.array([0.71, 1.41, 0.5])
        self.ki = np.zeros(3)
        self.int = np.zeros(3)

    def reset(self) -> None:
        self.int[:] = 0.0

    def step(self, omega, w_des, dt) -> np.ndarray:
        """
        Compute the torque command (N*m) for the next interval.
        """
        error = np.asarray(w_des, float) - np.asarray(omega, float)
        self.int = np.clip(self.int + error * dt, -1.0, 1.0)
        tau = self.kp * error + self.ki * self.int
        return tau