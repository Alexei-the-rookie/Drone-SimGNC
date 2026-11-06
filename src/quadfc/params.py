# SPDX-License-Identifier: GPL-3.0-or-later

"""Quadrotor physical model, loaded from YAML.

Everything here is "the plant".  A controller must not read these except for
sanity (e.g. hover speed), because on a real aircraft you would not have them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

G = 9.81


@dataclass(frozen=True)
class Rotor:
    """One rotor, in the body frame (x forward, y left, z up)."""

    x: float
    y: float
    z: float
    spin: int  # +1: CCW seen from above (+z), -1: CW

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass(frozen=True)
class QuadParams:
    name: str
    body_mass: float
    rotor_mass: float
    body_inertia: tuple[float, float, float]  # diagonal of the body inertia
    kf: float  # thrust coefficient, N / (rad/s)^2
    km_ratio: float  # kM / kF, dimensionless
    omega_max: float  # rad/s
    tau_up: float  # motor spin-up time constant, s
    tau_down: float  # motor spin-down time constant, s
    rotors: tuple[Rotor, ...]
    drag: float = 0.0  # linear damping force per unit velocity, N/(m/s)

    # ---- derived ---------------------------------------------------------

    @property
    def n_rotors(self) -> int:
        return len(self.rotors)

    @property
    def mass(self) -> float:
        return self.body_mass + self.n_rotors * self.rotor_mass

    @property
    def inertia(self) -> np.ndarray:
        """Total inertia tensor: body + rotor point masses (parallel axis)."""
        diag = np.array(self.body_inertia, dtype=float)
        for r in self.rotors:
            diag += self.rotor_mass * np.array(
                [r.y**2 + r.z**2, r.x**2 + r.z**2, r.x**2 + r.y**2]
            )
        return np.diag(diag)

    @property
    def weight(self) -> float:
        return self.mass * G

    @property
    def hover_omega(self) -> float:
        """Motor speed at which the aircraft is in equilibrium (level, no drag)."""
        return float(np.sqrt(self.weight / (self.n_rotors * self.kf)))

    @property
    def allocation(self) -> np.ndarray:
        """Mixer matrix B, shape (4, n_rotors).

        Maps squared motor speeds to wrench: u = [F, tau_x, tau_y, tau_z] = B @ omega^2.
        This is *the* matrix to check when the aircraft rolls the wrong way.
        """
        b = np.zeros((4, self.n_rotors))
        for i, r in enumerate(self.rotors):
            b[0, i] = self.kf
            b[1, i] = self.kf * r.y  # tau_x = sum(y_i * f_i)
            b[2, i] = -self.kf * r.x  # tau_y = -sum(x_i * f_i)
            b[3, i] = -self.kf * self.km_ratio * r.spin
        return b

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QuadParams":
        raw = yaml.safe_load(Path(path).read_text())
        rotors = tuple(
            Rotor(
                x=float(r["x"]),
                y=float(r["y"]),
                z=float(r.get("z", 0.0)),
                spin=int(r["spin"]),
            )
            for r in raw["rotors"]
        )
        return cls(
            name=raw.get("name", "quad"),
            body_mass=float(raw["body_mass"]),
            rotor_mass=float(raw["rotor_mass"]),
            body_inertia=tuple(float(v) for v in raw["body_inertia"]),
            kf=float(raw["kf"]),
            km_ratio=float(raw["km_ratio"]),
            omega_max=float(raw["omega_max"]),
            tau_up=float(raw["tau_up"]),
            tau_down=float(raw["tau_down"]),
            rotors=rotors,
            drag=float(raw.get("drag", 0.0)),
        )
