# SPDX-License-Identifier: GPL-3.0-or-later

"""Command line entry point: python3 -m quadfc --help"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from quadfc.control import Controller, HoverController
from quadfc.control.rate_alt import RateAltController
from quadfc.params import QuadParams
from quadfc.plant.analytic import AnalyticPlant
from quadfc.plant.base import Plant
from quadfc.runner import Log, run

PKG_ROOT = Path(__file__).resolve().parent.parent  # .../src
DEFAULT_PARAMS = PKG_ROOT / "config" / "x3.yaml"
DEFAULT_WORLD = PKG_ROOT / "worlds" / "x3_quadcopter.sdf"


def build_controller(name: str, params: QuadParams, args) -> Controller:
    if name == "hover":
        return HoverController(params, scale=args.hover_scale)
    if name == "rate_alt":
        return RateAltController(params)
    raise SystemExit(f"unknown controller '{name}'")


def build_plant(backend: str, params: QuadParams, args) -> Plant:
    if backend == "analytic":
        return AnalyticPlant(params, substep=args.substep, altitude=args.altitude)
    if backend == "gz":
        from quadfc.plant.gzsim import GzPlant

        return GzPlant(params, args.world, spawn=not args.attach, gui=args.gui,
                       altitude=args.altitude if args.altitude > 0 else None)
    raise SystemExit(f"unknown backend '{backend}'")


def plot_log(log: Log, path: Path | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = log.arrays()
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    for i, name in enumerate(["x", "y", "z"]):
        axes[0].plot(a["t"], a["position"][:, i], label=name)
    axes[0].set_ylabel("position [m]")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    for i, name in enumerate(["roll", "pitch", "yaw"]):
        axes[1].plot(a["t"], np.degrees(a["rpy"][:, i]), label=name)
    axes[1].set_ylabel("attitude [deg]")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    for i in range(a["command"].shape[1]):
        axes[2].plot(a["t"], a["command"][:, i], label=f"motor {i}")
    axes[2].set_ylabel("motor speed [rad/s]")
    axes[2].set_xlabel("sim time [s]")
    axes[2].legend(loc="upper right", ncol=4, fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    out = path or (Path.cwd() / "quadfc_plot.png")
    fig.savefig(out, dpi=120)
    print(f"plot  -> {out}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="quadfc",
        description="Quadrotor controller / plant / simulation skeleton.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--backend", choices=["analytic", "gz"], default="analytic")
    p.add_argument("--controller", default="hover", choices=["hover", "rate_alt"])
    p.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    p.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--rate", type=float, default=100.0, help="control loop rate [Hz]")
    p.add_argument("--altitude", type=float, default=0.0, help="initial z (analytic backend)")
    p.add_argument("--substep", type=float, default=1e-3, help="analytic integration step [s]")
    p.add_argument("--hover-scale", type=float, default=1.0, help="hover throttle multiplier")
    p.add_argument("--attach", action="store_true", help="use an already running gz sim")
    p.add_argument("--gui", action="store_true", help="open the gz GUI")
    p.add_argument("--no-realtime", action="store_true", help="run analytic backend flat out")
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    params = QuadParams.from_yaml(args.params)
    plant = build_plant(args.backend, params, args)
    controller = build_controller(args.controller, params, args)

    print(
        f"{params.name}: mass={params.mass:.3f} kg  "
        f"I=diag({np.diag(params.inertia).round(6)})\n"
        f"hover speed={params.hover_omega:.1f} rad/s  "
        f"(limit {params.omega_max:.0f}, margin {100 * params.hover_omega / params.omega_max:.0f}%)\n"
        f"backend={args.backend}  controller={controller.name}  "
        f"rate={args.rate:.0f} Hz  duration={args.duration:g}s"
    )

    try:
        with plant:
            log, report = run(
                plant,
                controller,
                args.duration,
                args.rate,
                realtime=not args.no_realtime,
                verbose=not args.quiet,
            )
    except RuntimeError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1

    print("\n" + str(report))

    if args.csv:
        print(f"csv   -> {log.to_csv(args.csv)}")
    if args.plot:
        plot_log(log, None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
