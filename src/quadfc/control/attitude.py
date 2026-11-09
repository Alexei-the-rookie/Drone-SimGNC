import numpy as np


class AttitudeController:
    """
    Thrust-vector attitude loop (B-plan).

    ``step(R, f_des, yaw_des)`` returns the desired body rate in the **body
    frame** [rad/s].  The desired attitude is not commanded directly: it is
    whatever orientation puts the body z axis along ``f_des``, with ``yaw_des``
    fixing the heading.  Stateless, so there is no ``reset()``.
    """
    name = "attitude"
    def __init__(self):
        self.kp = np.array([8.0, 8.0, 1.0])

    def step(self, R, f_des, yaw_des) -> np.ndarray:
        """
        Compute the desired angular velocity w_des for the next interval.
        """
        R_des = yaw_des_rotation(f_des, yaw_des)
        return -self.kp * attitude_error(R, R_des)

def _Rz(yaw: float) -> np.ndarray:
    """Level attitude with the given heading."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def yaw_des_rotation(f_des, yaw_des=0.0) -> np.ndarray:
    """
    Compute the desired rotation matrix R_des from the desired force vector f_des
    and desired yaw angle yaw_des.

    Degenerate inputs fall back to level-at-``yaw_des`` rather than dividing by
    a near-zero norm: a zero force vector has no thrust direction, and a force
    vector parallel to the yaw reference needs 90 deg of tilt (unreachable with
    X3 authority, which tops out near 47 deg).
    """
    norm_f = float(np.linalg.norm(f_des))
    if norm_f < 1e-6:
        return _Rz(yaw_des)
    z_b = f_des / norm_f
    x_c = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
    y_b = np.cross(z_b, x_c)
    norm_y = float(np.linalg.norm(y_b))
    if norm_y < 1e-6:
        return _Rz(yaw_des)
    y_b = y_b / norm_y
    x_b = np.cross(y_b, z_b)
    R_des = np.column_stack([x_b, y_b, z_b])
    return R_des

def attitude_error(R, R_des) -> np.ndarray:
    """
    Compute the attitude error between the current rotation matrix R and the desired
    rotation matrix R_des.

    Returns the rotation vector (body frame) that takes R_des to R, so the
    correction is ``w_des = -kp * e_R``.  Uses transposes instead of inverses so
    that M stays exactly skew-symmetric even if R carries numerical drift.
    """
    M = R_des.T @ R - R.T @ R_des
    e_R = 0.5 * np.array([M[2, 1], M[0, 2], M[1, 0]])
    return e_R