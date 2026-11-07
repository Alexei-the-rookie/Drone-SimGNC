# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np

from quadfc.control.base import Controller
from quadfc.params import QuadParams
from quadfc.state import State

class RateAltController(Controller):

    name = "rate_alt"

    def __init__(self, params: QuadParams):
        self.params = params
        self.g = 9.81
        self.B = params.allocation
        self.B_inv = np.linalg.inv(self.B)
        self.w2_max = params.omega_max ** 2


        # omega_loop define Kp = Inertia * wc
        I = np.diag(params.inertia)
        self.kp_rate = np.array([0.71,1.41, 0.5])
        self.ki_rate = np.array([0.0, 0.0, 0.0])
        self.int_rate = np.zeros(3)

        # altitude_loop expected_acc to lift_force
        self.kp_z, self.ki_z, self.kd_z = 2.5, 0.0, 3.2
        self.int_z = 0.0
        self.z_des = None
        self.w_des = np.zeros(3)

    def reset(self, state: State) -> None:
        self.int_rate[:] = 0.0
        self.int_z = 0.0
        self.z_des = state.position[2]

    def step(self, state: State, dt: float) -> np.ndarray:
        # error in altitude -> total thrust
        e_z = self.z_des - state.position[2]
        v_z = state.velocity[2] # speed in z direction in the world frame
        self.int_z = np.clip(self.int_z + e_z * dt, -1.0, 1.0) # integral
        acc_z = self.kp_z * e_z + self.ki_z * self.int_z - self.kd_z * v_z
        F = self.params.mass * (self.g + acc_z)

        # omega-loop: w_des -> tau
        e_w = self.w_des - state.omega
        self.int_rate = np.clip(self.int_rate + e_w * dt, -1.0, 1.0) # integral
        tau = self.kp_rate * e_w + self.ki_rate * self.int_rate

        # mixed: [F, tau_x, tau_y, tau_z] = B @ w^2
        w2 = self.B_inv @ np.array([F, *tau])
        # scaling...
        if w2.max() > self.w2_max or w2.min() < 0.0:
            w2 = self._scale(w2)
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


#class PositionController(Controller):
