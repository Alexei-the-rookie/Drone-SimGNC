# SPDX-License-Identifier: GPL-3.0-or-later

"""Self-contained 6-DOF quadrotor with first-order motor lag.

This is the backend to develop controllers against: no simulator, no
dependency beyond numpy, deterministic, and fast enough to run a 1000-step
sweep without noticing.

Rigid-body equations (body frame, FLU):

    m dv/dt = R [0, 0, F] - m g e_z - c v
    I dw/dt = tau - w x (I w)
    dq/dt   = 0.5 q (x) [w, 0]
    d wm/dt = (cmd - wm) / tau_motor        (tau_up or tau_down per motor)

with [F, tau] = B @ wm^2, B being `QuadParams.allocation`.
"""

from __future__ import annotations

import numpy as np

from quadfc.params import G, QuadParams
from quadfc.plant.base import Plant
from quadfc.state import State, quat_mul, quat_normalize, quat_to_rotmat


class AnalyticPlant(Plant):
    def __init__(self, params: QuadParams, substep: float = 1e-3, altitude: float = 0.0):
        super().__init__(params)
        self.substep = substep
        self.altitude = altitude
        self._B = params.allocation
        self._I = params.inertia
        self._I_inv = np.linalg.inv(self._I)
        self._n = params.n_rotors
        self._x = self._equilibrium_state(0.0)
        self._t = 0.0

    # ---- state handling --------------------------------------------------

    def _equilibrium_state(self, z0: float) -> np.ndarray:
        return np.concatenate(
            [
                np.array([0.0, 0.0, z0]),  # position
                np.zeros(3),  # velocity
                np.array([0.0, 0.0, 0.0, 1.0]),  # quaternion
                np.zeros(3),  # body rates
                np.full(self._n, self.params.hover_omega),  # motor speeds
            ]
        )

    @property
    def state(self) -> State:
        x = self._x
        return State(
            t=self._t,
            position=x[0:3].copy(),
            velocity=x[3:6].copy(),
            quat=x[6:10].copy(),
            omega=x[10:13].copy(),
        )

    def start(self) -> None:
        pass

    def reset(self) -> State:
        self._x = self._equilibrium_state(self.altitude)
        self._t = 0.0
        return self.state

    # ---- dynamics --------------------------------------------------------

    def _derivative(self, x: np.ndarray, cmd: np.ndarray) -> np.ndarray:
        p = self.params
        pos_dot = x[3:6]
        quat = x[6:10]
        omega = x[10:13]
        motor = x[13 : 13 + self._n]

        w2 = motor * motor
        wrench = self._B @ w2
        thrust_b = np.array([0.0, 0.0, wrench[0]])

        rot = quat_to_rotmat(quat)
        acc = rot @ thrust_b / p.mass
        acc[2] -= G
        if p.drag:
            acc -= p.drag * x[3:6] / p.mass

        torque = wrench[1:4]
        omega_dot = self._I_inv @ (torque - np.cross(omega, self._I @ omega))

        quat_dot = 0.5 * quat_mul(quat, np.array([omega[0], omega[1], omega[2], 0.0]))

        tau = np.where(cmd > motor, p.tau_up, p.tau_down)
        motor_dot = (cmd - motor) / tau

        return np.concatenate([pos_dot, acc, quat_dot, omega_dot, motor_dot])

    def _integrate(self, cmd: np.ndarray, h: float) -> None:
        x = self._x
        k1 = self._derivative(x, cmd)
        k2 = self._derivative(x + 0.5 * h * k1, cmd)
        k3 = self._derivative(x + 0.5 * h * k2, cmd)
        k4 = self._derivative(x + h * k3, cmd)
        self._x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._x[6:10] = quat_normalize(self._x[6:10])

    def step(self, motor_cmd, dt: float) -> State:
        cmd = np.clip(np.asarray(motor_cmd, dtype=float), 0.0, self.params.omega_max)
        n = max(1, int(np.ceil(dt / self.substep)))
        h = dt / n
        for _ in range(n):
            self._integrate(cmd, h)
            self._t += h
        return self.state
