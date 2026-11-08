# Step Sine Rate Alt Controller Test
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # .../src

from quadfc.control.rate_alt import RateAltController
from quadfc.params import QuadParams
from quadfc.plant.analytic import AnalyticPlant

ROOT = Path(__file__).resolve().parent.parent.parent  # .../src

@dataclass
class Run:
    t:np.ndarray
    z:np.ndarray
    vz:np.ndarray
    thrust:np.ndarray
    saturated:np.ndarray

def simulate(params, z0, ref, duration, rate, kp, kd):
    plant = AnalyticPlant(params, substep=1e-3, altitude=z0)
    ctrl = RateAltController(params)
    ctrl.kp_z = kp
    ctrl.kd_z = kd
    s = plant.reset()
    ctrl.reset(s)

    dt = 1.0 / rate
    n = int(np.ceil(duration / dt))
    t, z, vz, thrust = (np.empty(n) for _ in range(4))
    sat = np.empty(n, dtype=bool)

    for i in range(n):
        ctrl.z_des = float(ref(s.t))  # update reference
        u = np.asarray(ctrl.step(s, dt), dtype=float)  # compute control input
        c = np.clip(u, 0.0, params.omega_max)  # clip to max/min
        sat[i] = ctrl.saturated  # set inside step(), before the internal _scale

        t[i], z[i], vz[i] = s.t, s.position[2], s.velocity[2]
        thrust[i] = params.allocation[0] @ (c ** 2)  # compute thrust from motor speeds
        s = plant.step(c, dt)  # step the plant

    return Run(t, z, vz, thrust, sat)

def step_metrics(r, z0, z1, t_step):
    sel = r.t >= t_step
    tt, zz = r.t[sel] - t_step, r.z[sel]
    dz = z1 - z0

    def first_cross(frac):
        target = z0 + frac * dz
        hit = np.flatnonzero(zz >= target) if dz > 0 else np.flatnonzero(zz <= target)
        return float(tt[hit[0]]) if hit.size else float("nan")

    t10, t90 = first_cross(0.1), first_cross(0.9)
    peak = float(zz.max() if dz > 0 else zz.min())
    overshoot = 100.0 * (peak - z1) / dz if dz > 0 else 100.0 * (z1 - peak) / -dz

    outside = np.flatnonzero(np.abs(zz - z1) > 0.02 * abs(dz))
    settle = float(tt[outside[-1] + 1]) if outside.size and outside[-1] + 1 < tt.size else float("nan")
    tail = zz[tt >= tt[-1] - 0.5]

    return {
        "t_rise_10_90 [s]": t90 - t10,
        "overshoot [%]": max(0.0, overshoot),
        "t_settle_2% [s]": settle,
        "steady_err [m]": float(tail.mean() - z1),
        "peak_z [m]": peak,
        "sat [%]": 100.0 * r.saturated[sel].mean(),
    }

def sine_fit(r, f, t_start):
    """z ≈ c + a·sin(wt) + b·cos(wt) 的最小二乘拟合。"""
    sel = r.t >= t_start
    tt, zz = r.t[sel], r.z[sel]
    w = 2.0 * np.pi * f
    X = np.column_stack([np.ones_like(tt), np.sin(w * tt), np.cos(w * tt)])
    coef, *_ = np.linalg.lstsq(X, zz, rcond=None)
    a, b = float(coef[1]), float(coef[2])
    gain = float(np.hypot(a, b))
    lag = float(-np.degrees(np.arctan2(b, a)))   # positive = output lags input
    return gain, lag

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode", choices=["step", "sine", "both"], default="both")
    ap.add_argument("--params", type=Path, default=ROOT / "config" / "x3.yaml")
    ap.add_argument("--altitude", type=float, default=1.0, help="起始高度 [m]")
    ap.add_argument("--step", type=float, default=1.0, help="阶跃幅值 [m]")
    ap.add_argument("--step-at", type=float, default=0.5, help="阶跃时刻 [s]")
    ap.add_argument("--hold", type=float, default=10.0, help="阶跃后仿真时长 [s]")
    ap.add_argument("--amplitude", type=float, default=0.05, help="正弦幅值 [m]")
    ap.add_argument("--freqs", default="0.05,0.08,0.10,0.13,0.16,0.20,0.30")
    ap.add_argument("--rate", type=float, default=100.0, help="控制频率 [Hz]")
    ap.add_argument("--kp", type=float, default=None, help="临时覆盖 kp_z")
    ap.add_argument("--kd", type=float, default=None, help="临时覆盖 kd_z")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    if abs(args.step) < 1e-9:
        raise SystemExit("--step must be non-zero")

    params = QuadParams.from_yaml(args.params)
    kp = 1.0 if args.kp is None else args.kp
    kd = 0.0 if args.kd is None else args.kd
    thrust_max = 4.0 * params.kf * params.omega_max**2

    print(f"hover_omega = {params.hover_omega:7.1f} rad/s")
    print(f"thrust      = {params.weight:7.3f} N hover / {thrust_max:7.3f} N max "
          f"(+{(thrust_max - params.weight) / params.mass:.2f} / -9.81 m/s^2)")
    print(f"altitude loop: kp_z={kp:g}  ki_z=0  kd_z={kd:g}")

    step_info = None
    if args.mode in ("step", "both"):
        z0, z1 = args.altitude, args.altitude + args.step
        ref = lambda t: z0 if t < args.step_at else z1
        r = simulate(params, z0, ref, args.step_at + args.hold, args.rate, kp, kd)
        step_info = (r, z0, z1, args.step_at)

        print(f"\n=== step {z0:.2f} -> {z1:.2f} m at t={args.step_at:.2f} s ===")
        for key, val in step_metrics(r, z0, z1, args.step_at).items():
            print(f"  {key:18s} {val:9.3f}")

        if args.csv:
            np.savetxt(args.csv,
                       np.column_stack([r.t, r.z, r.vz, r.thrust, r.saturated.astype(float)]),
                       delimiter=",", header="t,z,vz,thrust,saturated", comments="")
            print(f"  csv -> {args.csv}")

    sine_results = []
    if args.mode in ("sine", "both"):
        print(f"\n=== sine sweep, amplitude {args.amplitude:g} m ===")
        print(f"  {'f [Hz]':>8} {'gain':>8} {'gain [dB]':>10} {'lag [deg]':>10} {'sat [%]':>8}")
        for f in (float(x) for x in args.freqs.split(",")):
            period = 1.0 / f
            duration = max(20.0, 6.0 * period)
            t_start = max(5.0, 2.0 * period)
            ref = lambda t, f=f: args.altitude + args.amplitude * np.sin(2.0 * np.pi * f * t)
            r = simulate(params, args.altitude, ref, duration, args.rate, kp, kd)
            mag, lag = sine_fit(r, f, t_start)
            gain = mag / args.amplitude
            sat = 100.0 * r.saturated[r.t >= t_start].mean()
            sine_results.append((f, gain, lag))
            print(f"  {f:8.3f} {gain:8.2f} {20 * np.log10(max(gain, 1e-9)):10.2f} "
                  f"{lag:10.1f} {sat:8.1f}")

        fs = np.array([x[0] for x in sine_results])
        db = np.array([20 * np.log10(max(x[1], 1e-9)) for x in sine_results])
        if db[0] >= -3.0:
            below = np.flatnonzero(db <= -3.0)
            if below.size and below[0] > 0:
                i = below[0]
                bw = float(np.interp(-3.0, [db[i - 1], db[i]], [fs[i - 1], fs[i]]))
                print(f"  -3 dB bandwidth ≈ {bw:.3f} Hz")

    if args.plot and (step_info or sine_results):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ncol = (1 if step_info else 0) + (2 if sine_results else 0)
        fig, axes = plt.subplots(1, ncol, figsize=(5 * ncol, 4), squeeze=False)
        ax, col = axes[0], 0

        if step_info:
            r, z0, z1, t_step = step_info
            ax[col].plot(r.t, r.z)
            ax[col].axhline(z1, ls="--", c="k", lw=0.8)
            ax[col].axvline(t_step, ls=":", c="k", lw=0.8)
            ax[col].set(xlabel="t [s]", ylabel="z [m]", title="step response")
            ax[col].grid(alpha=0.3)
            col += 1

        if sine_results:
            ax[col].semilogx(fs, db, "o-")
            ax[col].axhline(-3, ls="--", c="k", lw=0.8)
            ax[col].set(xlabel="f [Hz]", ylabel="gain [dB]", title="sine gain")
            ax[col].grid(alpha=0.3, which="both")
            col += 1
            ax[col].semilogx(fs, [x[2] for x in sine_results], "o-")
            ax[col].set(xlabel="f [Hz]", ylabel="phase lag [deg]", title="sine phase")
            ax[col].grid(alpha=0.3, which="both")

        fig.tight_layout()
        fig.savefig("rate_alt_step_sine.png", dpi=120)
        print(f"\nplot -> {(Path.cwd() / 'rate_alt_step_sine.png').resolve()}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())