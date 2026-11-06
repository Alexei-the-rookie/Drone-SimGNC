# SPDX-License-Identifier: GPL-3.0-or-later

"""Open-loop cross-check: analytic model vs Gazebo.

The point of this script is to answer one question -- *does the numpy plant
behave like the simulator?* -- before you trust either of them with a
controller.  Both backends get the same open-loop motor schedule; if the
trajectories diverge you have a wrong sign, a wrong unit or a wrong inertia,
and you want to find that now, not while debugging an MPC.

    python3 tools/compare_backends.py --duration 1.5 --altitude 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quadfc.params import G, QuadParams  # noqa: E402
from quadfc.plant.analytic import AnalyticPlant  # noqa: E402
from quadfc.plant.gzsim import GzPlant  # noqa: E402

PKG_ROOT = Path(__file__).resolve().parent.parent


def schedule(params: QuadParams, t: float) -> np.ndarray:
    """Open-loop motor command as a function of time: hover, then excite."""
    cmd = np.full(params.n_rotors, params.hover_omega)
    if 0.20 <= t < 0.60:
        ramp = np.sin(np.pi * (t - 0.20) / 0.4)
        cmd += np.array([25.0, -25.0, 25.0, -25.0]) * ramp  # pitch
        cmd += np.array([8.0, 8.0, -8.0, -8.0]) * ramp  # yaw
    if 0.75 <= t < 0.90:
        cmd -= 15.0
    return np.clip(cmd, 0.0, params.omega_max)


def trim(plant, params: QuadParams, altitude: float, timeout: float = 5.0) -> None:
    """Settle on `altitude` with a vertical-only P-D on the common throttle.

    Needed because the simulator starts on the ground (and loses a few cm of
    altitude in the latency between teleport and the first command), while the
    analytic plant starts exactly at equilibrium.  Comparing an aircraft that is
    still falling with one that is not tells you nothing.

    The loop works in acceleration space and maps back through
    omega = omega_hover * sqrt(1 + a/g), which keeps the authority sane far
    from the target (a raw gain on the motor speed does not).
    """
    s = plant.state
    t_end = s.t + timeout
    while s.t < t_end:
        a_cmd = 2.5 * (altitude - s.position[2]) - 3.2 * s.velocity[2]
        a_cmd = float(np.clip(a_cmd, -0.6 * G, 4.0 * G))
        throttle = params.hover_omega * np.sqrt(max(0.05, 1.0 + a_cmd / G))
        cmd = np.full(params.n_rotors, min(throttle, params.omega_max))
        s = plant.step(cmd, 0.01)
        if abs(altitude - s.position[2]) < 0.01 and abs(s.velocity[2]) < 0.02:
            break


def rollout(plant, params: QuadParams, duration: float, rate: float):
    dt = 1.0 / rate
    s = plant.state
    t0 = s.t
    ts, ps, rpys = [0.0], [s.position.copy()], [s.euler()]
    while s.t - t0 < duration:
        s = plant.step(schedule(params, s.t - t0), dt)
        ts.append(s.t - t0)
        ps.append(s.position.copy())
        rpys.append(s.euler())
    return np.array(ts), np.array(ps), np.array(rpys)


def interp(t_src, y_src, t_dst):
    return np.column_stack(
        [np.interp(t_dst, t_src, y_src[:, i]) for i in range(y_src.shape[1])]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", type=Path, default=PKG_ROOT / "config" / "x3.yaml")
    ap.add_argument("--world", type=Path, default=PKG_ROOT / "worlds" / "x3_quadcopter.sdf")
    ap.add_argument("--duration", type=float, default=1.5)
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--altitude", type=float, default=0.5)
    ap.add_argument("--attach", action="store_true", help="use an already running gz sim")
    args = ap.parse_args()

    params = QuadParams.from_yaml(args.params)

    analytic = AnalyticPlant(params, substep=1e-4, altitude=args.altitude)
    gz = GzPlant(params, args.world, spawn=not args.attach, altitude=args.altitude)

    with gz:
        trim(gz, params, args.altitude)
        print(f"gz  trimmed: {gz.state}")
        t_gz, p_gz, rpy_gz = rollout(gz, params, args.duration, args.rate)

    analytic.reset()
    trim(analytic, params, args.altitude)
    print(f"ana trimmed: {analytic.state}")
    t_an, p_an, rpy_an = rollout(analytic, params, args.duration, args.rate)

    grid = np.linspace(0.0, min(t_gz[-1], t_an[-1]), 200)
    p_gz_i = interp(t_gz, p_gz, grid)
    p_an_i = interp(t_an, p_an, grid)
    rpy_gz_i = np.degrees(interp(t_gz, rpy_gz, grid))
    rpy_an_i = np.degrees(interp(t_an, rpy_an, grid))

    dp = np.abs(p_gz_i - p_an_i)
    drpy = np.abs(rpy_gz_i - rpy_an_i)
    excursion = max(np.abs(p_an_i - p_an_i[0]).max(), 1e-6)

    print("\n           analytic         gz        |diff|")
    for i, name in enumerate("xyz"):
        print(
            f"{name} [m]   {p_an_i[-1, i]:9.4f}   {p_gz_i[-1, i]:9.4f}   {dp[-1, i]:8.4f}"
        )
    for i, name in enumerate(["roll ", "pitch", "yaw  "]):
        print(
            f"{name}[deg] {rpy_an_i[-1, i]:9.3f}   {rpy_gz_i[-1, i]:9.3f}   "
            f"{drpy[-1, i]:8.3f}"
        )

    print(
        f"\nmax |dp|   = {dp.max():.4f} m  "
        f"({100 * dp.max() / excursion:.1f}% of the trajectory excursion)\n"
        f"max |drpy| = {drpy.max():.3f} deg"
    )
    ok = dp.max() < 0.10 and drpy.max() < 2.0
    print("VERDICT:", "models agree" if ok else "MODELS DIVERGE -- check signs/units")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
