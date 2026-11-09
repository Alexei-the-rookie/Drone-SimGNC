# SPDX-License-Identifier: GPL-3.0-or-later

"""Control layer.

A controller sees a `State` and a time step, and returns four motor speeds in
rad/s.  Nothing else.  Mixer, saturation and motor lag belong to the plant.

Plug-in points for the control laws you actually care about:
  * PID            -> new file control/pid.py
  * LQR            -> new file control/lqr.py   (needs a linearised model)
  * MPC            -> new file control/mpc.py   (needs a QP solver)
Each is a subclass of `Controller` and needs no change to the runner.
"""

from quadfc.control.base import Controller
from quadfc.control.cascade import CascadeController
from quadfc.control.hover import HoverController

__all__ = ["Controller", "CascadeController", "HoverController"]
