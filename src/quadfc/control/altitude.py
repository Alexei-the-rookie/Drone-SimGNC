import numpy as np

class AltitudeController:
    """
    A altitude controller inputed (z, z_cmd) and output acc_z in m/s^2.
    """
    name = "altitude"
    def __init__(self):
        self.kp = 2.5
        self.ki = 0.0
        self.kd = 3.2
        self.int = 0.0
        self.z_des = None

    def reset(self, z0: float) -> None:
        self.int = 0.0
        if self.z_des is None:
            self.z_des = z0

    def step(self, z, vz, dt) -> float:
        """
        Compute the acc_z for the next interval.
        """
        error = self.z_des - z
        self.int = np.clip(self.int + error * dt, -1.0, 1.0)
        acc_z = self.kp * error - self.kd * vz + self.ki * self.int
        return acc_z