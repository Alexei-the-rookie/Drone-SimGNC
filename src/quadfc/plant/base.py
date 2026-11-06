# SPDX-License-Identifier: GPL-3.0-or-later

"""Plant interface.

A plant takes motor-speed commands and advances time.  Both backends implement
the same three methods, so the runner and the controller never know which one
they are talking to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from quadfc.params import QuadParams
from quadfc.state import State


class Plant(ABC):
    def __init__(self, params: QuadParams):
        self.params = params

    @property
    @abstractmethod
    def state(self) -> State:
        """Latest state. Must not block."""

    @abstractmethod
    def start(self) -> None:
        """Allocate resources / launch the simulator."""

    @abstractmethod
    def reset(self) -> State:
        """Return the state the loop should start from."""

    @abstractmethod
    def step(self, motor_cmd, dt: float) -> State:
        """Apply a motor command and advance time by `dt`. Returns the new state."""

    def close(self) -> None:
        """Release resources. Safe to call twice."""

    def __enter__(self) -> "Plant":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
