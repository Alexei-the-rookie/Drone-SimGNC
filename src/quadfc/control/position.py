import numpy as np

class PositionController():
    """
    A simple position controller that computes the desired velocity based on the current position and desired position.
    """
    name = "position"

    def __init__(self, kp: float = 0.5):
        self.kp = np.array([kp, kp])  # Proportional gain for position control

    def reset(self) -> None:
        pass

    def step(self, pos, p_des, dt) -> np.ndarray:
        """
        Compute the desired velocity based on the current position and desired position.
        """
        return self.kp * (np.asarray(p_des, float) - np.asarray(pos, float))