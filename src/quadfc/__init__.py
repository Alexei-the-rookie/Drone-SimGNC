# SPDX-License-Identifier: GPL-3.0-or-later

"""quadfc -- controller / plant / simulation skeleton for a hand-written quadrotor
flight controller.

Layers
------
params    : the physical model (mass, inertia, kf, motor lag, rotor layout)
state     : State dataclass + quaternion helpers (one convention, everywhere)
control/  : Controller ABC; drop your PID / LQR / MPC in here
plant/    : Plant ABC; AnalyticPlant (numpy 6-DOF) or GzPlant (Gazebo Harmonic)
runner    : fixed-rate closed loop + logging
cli       : python3 -m quadfc
"""

__version__ = "0.1.0"
