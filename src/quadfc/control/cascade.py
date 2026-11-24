import numpy as np
from quadfc.control.base import Controller
from quadfc.params import G
from .mixer import MixerController
from .rate import RateController
from .altitude import AltitudeController
from .attitude import AttitudeController
from .velocity import VelocityController
from .position import PositionController

class CascadeController(Controller):
    """
    Position -> Velocity -> Acceleration -> Attitude -> Rate -> Mixer
    """
    name = "cascade"
    def __init__(self, params, tilt_max: float = 30.0):
        self.params = params
        self.p_des = None  # Desired position in the XY plane
        self.a_des = np.zeros(3)  # Desired acceleration
        self.a_max = G * np.tan(np.radians(tilt_max))  # Maximum horizontal acceleration based on tilt limit
        
        self.vel = VelocityController()
        self.pos = PositionController()
        self.alt = AltitudeController()
        self.att = AttitudeController()
        self.rate = RateController(params)
        self.mix = MixerController(params)
        self.yaw_des = None

    def reset(self, state) -> None:
        self.alt.reset(state.position[2])
        self.pos.reset()
        self.rate.reset()
        self.vel.reset()
        if self.yaw_des is None:
            self.yaw_des = state.euler()[2]  # Initialize desired yaw to current yaw
        if self.p_des is None:
            self.p_des = state.position[:2].copy()  # Initialize desired position to current position
        self.a_des[:] = 0.0  # Reset desired acceleration to zero

    def step(self, state, dt) -> np.ndarray:
        """
        Compute the torque command (N*m) and acc_z for the next interval.
        """
        # 1.height control
        acc_z = self.alt.step(state.position[2], state.velocity[2], dt)
        # 2.pos -> vel_des
        v_des = self.pos.step(state.position[:2], self.p_des, dt)
        # 3.horizontal acceleration control
        a_des_xy = self.vel.step(state.velocity[:2], v_des, dt)
        a_des_xy = np.clip(a_des_xy, -self.a_max, self.a_max)  # Limit horizontal acceleration
        # 4. combine horizontal and vertical acceleration
        self.a_des[:] = [a_des_xy[0], a_des_xy[1], acc_z]
        f_des = self.params.mass * (self.a_des + np.array([0.0, 0.0, G]))
        
        w_des = self.att.step(state.rotation(), f_des, self.yaw_des)
        tau = self.rate.step(state.omega, w_des, dt)
        
        return self.mix.step(float(np.linalg.norm(f_des)), tau)
        