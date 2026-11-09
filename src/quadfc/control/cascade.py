import numpy as np
from quadfc.control.base import Controller
from quadfc.params import G
from .mixer import MixerController
from .rate import RateController
from .altitude import AltitudeController
from .attitude import AttitudeController

class CascadeController(Controller):
    """
    A cascade controller that combines altitude and rate controllers.
    """
    name = "cascade"
    def __init__(self, params):
        self.params = params
        
        self.alt = AltitudeController()
        self.att = AttitudeController()
        self.rate = RateController(params)
        self.mix = MixerController(params)
        self.yaw_des = None

    def reset(self, state) -> None:
        self.alt.reset(state.position[2])
        self.rate.reset()
        if self.yaw_des is None:
            self.yaw_des = state.euler()[2]  # Initialize desired yaw to current yaw


    def step(self, state, dt) -> np.ndarray:
        """
        Compute the torque command (N*m) and acc_z for the next interval.
        """
        acc_z = self.alt.step(state.position[2], state.velocity[2], dt)
        a_des = np.array([0.0, 0.0, acc_z])
        f_des = self.params.mass * (a_des + np.array([0.0, 0.0, G]))
        w_des = self.att.step(state.rotation(), f_des, self.yaw_des)
        tau = self.rate.step(state.omega, w_des, dt)
        return self.mix.step(float(np.linalg.norm(f_des)), tau)
        