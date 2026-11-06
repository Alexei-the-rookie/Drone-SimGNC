# SPDX-License-Identifier: GPL-3.0-or-later

"""Controller interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from quadfc.state import State


class Controller(ABC):
    """Motor-speed level controller.

    Contract
    --------
    ``step`` is called at a fixed rate with the latest state and the nominal
    period, and must return a length-4 array of **motor speeds in rad/s**.

    Do not clip to [0, omega_max] here: that is the mixer's job.  Returning an
    unclipped command lets the runner report how badly you saturated, which is
    the fastest way to find a badly scaled gain.
    """

    name: str = "controller"

    @abstractmethod
    def reset(self, state: State) -> None:
        """Called once before the loop starts."""

    @abstractmethod
    def step(self, state: State, dt: float) -> np.ndarray:
        """Return the motor speed command (rad/s) for the next interval."""
