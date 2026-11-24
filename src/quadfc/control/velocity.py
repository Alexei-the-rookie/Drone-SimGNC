import numpy as np

class VelocityController():
    """
    A simple velocity controller that computes the desired acceleration based on the current velocity and desired velocity.
    """
    name = "velocity"

    def __init__(self, kp: float = 2.0):
        self.kp = np.array([kp, kp])  # Proportional gain for velocity control
        

    def reset(self) -> None:
        pass

    def step(self, vel, v_des, dt) -> np.ndarray:
        """
        Compute the desired acceleration based on the current velocity and desired velocity.
        """
        return self.kp * (np.asarray(v_des, float) - np.asarray(vel, float))