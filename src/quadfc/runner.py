# SPDX-License-Identifier: GPL-3.0-or-later

"""Fixed-rate closed loop plus logging."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from quadfc.control.base import Controller
from quadfc.plant.base import Plant
from quadfc.state import State


@dataclass
class Log:
    t: list = field(default_factory=list)
    position: list = field(default_factory=list)
    velocity: list = field(default_factory=list)
    rpy: list = field(default_factory=list)
    omega: list = field(default_factory=list)
    command: list = field(default_factory=list)
    clipped: list = field(default_factory=list)

    def append(self, t: float, state: State, cmd: np.ndarray, clipped: np.ndarray) -> None:
        self.t.append(t)
        self.position.append(state.position.copy())
        self.velocity.append(state.velocity.copy())
        self.rpy.append(state.euler())
        self.omega.append(state.omega.copy())
        self.command.append(np.asarray(cmd, dtype=float).copy())
        self.clipped.append(clipped.copy())

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "t": np.asarray(self.t),
            "position": np.asarray(self.position),
            "velocity": np.asarray(self.velocity),
            "rpy": np.asarray(self.rpy),
            "omega": np.asarray(self.omega),
            "command": np.asarray(self.command),
            "clipped": np.asarray(self.clipped),
        }

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        a = self.arrays()
        header = ["t"]
        cols = [a["t"]]
        for i in range(3):
            header.append(f"p{i}")
            cols.append(a["position"][:, i])
        for i in range(3):
            header.append(f"v{i}")
            cols.append(a["velocity"][:, i])
        for i, name in enumerate(["roll", "pitch", "yaw"]):
            header.append(name)
            cols.append(a["rpy"][:, i])
        for i, name in enumerate(["wx", "wy", "wz"]):
            header.append(name)
            cols.append(a["omega"][:, i])
        for i in range(a["command"].shape[1]):
            header.append(f"cmd{i}")
            cols.append(a["command"][:, i])
            header.append(f"clip{i}")
            cols.append(a["clipped"][:, i])

        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(np.column_stack(cols))
        return path


@dataclass
class Report:
    steps: int
    duration: float
    final_position: np.ndarray
    altitude_error: float
    max_tilt_deg: float
    saturation_fraction: float
    wall_time: float

    def __str__(self) -> str:
        return (
            f"steps={self.steps}  sim_time={self.duration:.2f}s  "
            f"wall={self.wall_time:.2f}s\n"
            f"final position = [{self.final_position[0]:7.3f} "
            f"{self.final_position[1]:7.3f} {self.final_position[2]:7.3f}]\n"
            f"altitude error = {self.altitude_error:+.3f} m   "
            f"max tilt = {self.max_tilt_deg:5.2f} deg   "
            f"saturated = {100 * self.saturation_fraction:5.1f}% of samples"
        )


def run(
    plant: Plant,
    controller: Controller,
    duration: float,
    rate: float,
    *,
    realtime: bool = True,
    log: Log | None = None,
    verbose: bool = True,
) -> tuple[Log, Report]:
    """Run one closed-loop episode.

    `realtime=False` only makes sense for the analytic plant; the gz plant is
    paced by the simulator regardless.
    """
    log = log if log is not None else Log()
    dt = 1.0 / rate
    omega_max = plant.params.omega_max

    state = plant.reset()
    controller.reset(state)
    start_pos = state.position.copy()
    start_wall = time.time()

    t = 0.0
    max_tilt = 0.0
    n_saturated = 0
    steps = 0
    next_tick = time.time()

    while t < duration - 1e-9:
        cmd = np.asarray(controller.step(state, dt), dtype=float)
        clipped = np.clip(cmd, 0.0, omega_max)
        if np.any(np.abs(cmd - clipped) > 1e-9):
            n_saturated += 1

        log.append(state.t, state, cmd, clipped)
        state = plant.step(clipped, dt)
        steps += 1
        t += dt

        max_tilt = max(max_tilt, np.degrees(np.linalg.norm(state.euler()[:2])))

        if verbose and steps % max(1, int(rate)) == 0:
            print(f"  {state}")

        if realtime and plant.__class__.__name__ == "AnalyticPlant":
            next_tick += dt
            sleep = next_tick - time.time()
            if sleep > 0:
                time.sleep(sleep)

    report = Report(
        steps=steps,
        duration=t,
        final_position=state.position.copy(),
        altitude_error=state.position[2] - start_pos[2],
        max_tilt_deg=max_tilt,
        saturation_fraction=n_saturated / max(steps, 1),
        wall_time=time.time() - start_wall,
    )
    return log, report
