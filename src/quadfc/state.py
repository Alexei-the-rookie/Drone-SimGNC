# SPDX-License-Identifier: GPL-3.0-or-later

"""State container and quaternion helpers.

Conventions, fixed once so every backend agrees:
  * world frame is ENU (x east, y north, z up), gravity is -z
  * body frame is FLU (x forward, y left, z up)
  * quaternion is (x, y, z, w) and rotates body -> world
  * angular velocity is expressed in the body frame
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotmat_to_euler(r: np.ndarray) -> np.ndarray:
    """ZYX intrinsic (yaw-pitch-roll), returns (roll, pitch, yaw)."""
    pitch = np.arcsin(np.clip(-r[2, 0], -1.0, 1.0))
    roll = np.arctan2(r[2, 1], r[2, 2])
    yaw = np.arctan2(r[1, 0], r[0, 0])
    return np.array([roll, pitch, yaw])


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )


@dataclass
class State:
    """Full rigid-body state of the aircraft at one instant."""

    t: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # world frame
    quat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))  # body rates

    def rotation(self) -> np.ndarray:
        """Body -> world rotation matrix."""
        return quat_to_rotmat(self.quat)

    def euler(self) -> np.ndarray:
        """(roll, pitch, yaw) in radians."""
        return rotmat_to_euler(self.rotation())

    def body_velocity(self) -> np.ndarray:
        return self.rotation().T @ self.velocity

    def copy(self) -> "State":
        return State(
            t=float(self.t),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quat=self.quat.copy(),
            omega=self.omega.copy(),
        )

    def __str__(self) -> str:
        p, rpy = self.position, self.euler()
        return (
            f"t={self.t:6.3f}  p=[{p[0]:6.3f} {p[1]:6.3f} {p[2]:6.3f}]"
            f"  rpy=[{np.degrees(rpy[0]):6.1f} {np.degrees(rpy[1]):6.1f}"
            f" {np.degrees(rpy[2]):6.1f}]deg"
        )
