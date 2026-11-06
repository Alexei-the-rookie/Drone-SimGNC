# SPDX-License-Identifier: GPL-3.0-or-later

"""Trivial controller: constant hover motor speed.

Its only job is to prove the controller/plant/runner wiring is correct.  If the
aircraft holds altitude with this, the mixer signs, the thrust coefficient and
the mass are all consistent.
"""

from __future__ import annotations

import numpy as np

from quadfc.control.base import Controller
from quadfc.params import QuadParams
from quadfc.state import State


class HoverController(Controller):
    name = "hover"

    def __init__(self, params: QuadParams, scale: float = 1.0):
        """`scale` multiplies the equilibrium speed, e.g. 1.02 to climb slowly."""
        self.name = f"hover(x{scale:g})"
        self.cmd = np.full(params.n_rotors, params.hover_omega * scale)

    def reset(self, state: State) -> None:
        pass

    def step(self, state: State, dt: float) -> np.ndarray:
        return self.cmd.copy()
