# Step / Sine Rate Loop Test
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise the rate loop + mixer in isolation.

This tool deliberately wires only the two layers it claims to test:

    w_des --> RateController --> tau --> MixController --> motor speed

The collective thrust is pinned at F = m·g, so the mixer linearisation point is
constant and the only thing left in the loop is

    w_des --> kp --> tau --> motor lag --> I w' = tau --> w

Using `CascadeController` here would drag the altitude layer into a test that
has nothing to do with height.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # .../src

from quadfc.control.mixer import MixerController
from quadfc.control.rate import RateController
from quadfc.params import G, QuadParams
from quadfc.plant.analytic import AnalyticPlant

ROOT = Path(__file__).resolve().parent.parent.parent  # .../src

AXES = {"roll": 0, "pitch": 1, "yaw": 2}


@dataclass
class Run:
    t: np.ndarray
    w: np.ndarray        # measured body rate on the excited axis
    w_des: np.ndarray    # reference fed to the controller
    tau: np.ndarray      # axis torque actually applied by the mixer
    thrust: np.ndarray   # collective thrust [N]
    saturated: np.ndarray


def simulate(params, z0, axis, ref, duration, rate, kp, ki=0.0):
    plant = AnalyticPlant(params, substep=1e-3, altitude=z0)
    rate_ctl = RateController(params)
    mix = MixerController(params)

    rate_ctl.kp[axis] = kp
    rate_ctl.ki[axis] = ki

    s = plant.reset()
    rate_ctl.reset()

    f = params.mass * G          # altitude layer disabled: hover thrust only
    w_des = np.zeros(3)

    dt = 1.0 / rate
    n = int(np.ceil(duration / dt))
    t = np.empty(n)
    w = np.empty(n)
    r = np.empty(n)
    tau = np.empty(n)
    thrust = np.empty(n)
    sat = np.empty(n, dtype=bool)

    for i in range(n):
        w_des[:] = 0.0
        w_des[axis] = float(ref(s.t))
        u = np.asarray(mix.step(f, rate_ctl.step(s.omega, w_des, dt)), dtype=float)
        c = np.clip(u, 0.0, params.omega_max)
        sat[i] = mix.saturated

        wrench = params.allocation @ (c**2)
        t[i], w[i], r[i] = s.t, s.omega[axis], w_des[axis]
        thrust[i], tau[i] = wrench[0], wrench[1 + axis]
        s = plant.step(c, dt)

    return Run(t, w, r, tau, thrust, sat)


def step_metrics(r, w1, t_step):
    sel = r.t >= t_step
    tt, ww = r.t[sel] - t_step, r.w[sel]
    dw = w1

    def first_cross(frac):
        target = frac * dw
        hit = np.flatnonzero(ww >= target) if dw > 0 else np.flatnonzero(ww <= target)
        return float(tt[hit[0]]) if hit.size else float("nan")

    t10, t90 = first_cross(0.1), first_cross(0.9)
    peak = float(ww.max() if dw > 0 else ww.min())
    overshoot = 100.0 * (peak - w1) / dw if dw > 0 else 100.0 * (w1 - peak) / -dw

    outside = np.flatnonzero(np.abs(ww - w1) > 0.02 * abs(dw))
    settle = float(tt[outside[-1] + 1]) if outside.size and outside[-1] + 1 < tt.size else float("nan")
    tail = ww[tt >= tt[-1] - 0.2]

    return {
        "t_rise_10_90 [s]": t90 - t10,
        "overshoot [%]": max(0.0, overshoot),
        "t_settle_2% [s]": settle,
        "steady_err [rad/s]": float(tail.mean() - w1),
        "peak_w [rad/s]": peak,
        "tau_peak [N.m]": float(np.abs(r.tau[sel]).max()),
        "sat [%]": 100.0 * r.saturated[sel].mean(),
    }


def sine_fit(r, f, t_start):
    """w ≈ c + a·sin(wt) + b·cos(wt) 的最小二乘拟合。"""
    sel = r.t >= t_start
    tt, ww = r.t[sel], r.w[sel]
    w = 2.0 * np.pi * f
    X = np.column_stack([np.ones_like(tt), np.sin(w * tt), np.cos(w * tt)])
    coef, *_ = np.linalg.lstsq(X, ww, rcond=None)
    a, b = float(coef[1]), float(coef[2])
    gain = float(np.hypot(a, b))
    lag = float(-np.degrees(np.arctan2(b, a)))   # positive = output lags input
    return gain, lag


def theory(kp, inertia, tau_m, freqs):
    """Closed loop with the motor lag kept.

    tau_cmd --(1/g)--> w_cmd --1/(1+tau_m s)--> w --g--> tau --1/(I s)--> w_out

    The mixer gain `g` cancels, leaving

        H(s) = kp / (I·tau_m·s² + I·s + kp)
             => wn = sqrt(kp/(I·tau_m)),  zeta = sqrt(I/(kp·tau_m))/2
    """
    wn = float(np.sqrt(kp / (inertia * tau_m)))
    zeta = float(0.5 * np.sqrt(inertia / (kp * tau_m)))
    gains, lags = [], []
    for f in freqs:
        s = 1j * 2.0 * np.pi * f
        h = kp / (inertia * tau_m * s**2 + inertia * s + kp)
        gains.append(float(abs(h)))
        lags.append(float(-np.degrees(np.angle(h))))
    return wn, zeta, np.asarray(gains), np.asarray(lags)


def bandwidth(fs, gains):
    """-3 dB point, interpolated on a decreasing-in-dB grid (first crossing)."""
    db = 20.0 * np.log10(np.maximum(gains, 1e-12))
    below = np.flatnonzero(db <= -3.0)
    if not below.size or below[0] == 0:
        return float("nan")
    i = below[0]
    return float(np.interp(-3.0, [db[i], db[i - 1]], [fs[i], fs[i - 1]]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode", choices=["step", "sine", "both"], default="both")
    ap.add_argument("--params", type=Path, default=ROOT / "config" / "x3.yaml")
    ap.add_argument("--axis", choices=[*AXES, "all"], default="all")
    ap.add_argument("--altitude", type=float, default=1.0, help="起始高度 [m]")
    ap.add_argument("--step", type=float, default=0.1, help="阶跃幅值 [rad/s]")
    ap.add_argument("--step-at", type=float, default=0.2, help="阶跃时刻 [s]")
    ap.add_argument("--hold", type=float, default=1.0, help="阶跃后仿真时长 [s]")
    ap.add_argument("--amplitude", type=float, default=0.02, help="正弦幅值 [rad/s]")
    ap.add_argument("--freqs", default="0.2,0.5,1,1.5,2,3,4,6,8")
    ap.add_argument("--rate", type=float, default=100.0, help="控制频率 [Hz]")
    ap.add_argument("--kp", type=float, default=None, help="临时覆盖该轴 kp")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    if abs(args.step) < 1e-9:
        raise SystemExit("--step must be non-zero")

    params = QuadParams.from_yaml(args.params)
    inertia = np.diag(params.inertia)
    # 默认值从控制器读，免得这里和 rate.py 各写一份而悄悄失配
    kp = RateController(params).kp.astype(float).copy()
    names = list(AXES) if args.axis == "all" else [args.axis]
    if args.kp is not None:
        for name in names:
            kp[AXES[name]] = args.kp
    thrust_max = 4.0 * params.kf * params.omega_max**2

    print(f"hover_omega = {params.hover_omega:7.1f} rad/s   "
          f"thrust = {params.weight:6.3f} N hover / {thrust_max:6.3f} N max")
    print("altitude layer not used: F = m*g constant\n")
    print(f"  {'axis':>5} {'I [kg m^2]':>10} {'kp':>6} {'kp/I [rad/s]':>13} "
          f"{'wn [rad/s]':>10} {'zeta':>6}")
    for name in names:
        i = AXES[name]
        k = kp[i]
        wn, zeta, _, _ = theory(k, inertia[i], params.tau_up, [])
        print(f"  {name:>5} {inertia[i]:10.4f} {k:6.2f} {k / inertia[i]:13.1f} "
              f"{wn:10.1f} {zeta:6.2f}")
    print(f"  (wn/zeta = 2nd order incl. motor lag, tau_up = {params.tau_up:g} s)")

    freqs = np.array([float(x) for x in args.freqs.split(",")])

    # ---- step ----------------------------------------------------------
    if args.mode in ("step", "both"):
        print(f"\n=== rate step 0 -> {args.step:+.3f} rad/s at t={args.step_at:.2f} s ===")
        for name in names:
            i = AXES[name]
            k = kp[i]
            ref = lambda t, st=args.step: 0.0 if t < args.step_at else st
            r = simulate(params, args.altitude, i, ref,
                         args.step_at + args.hold, args.rate, k)
            wn, zeta, _, _ = theory(k, inertia[i], params.tau_up, [])
            print(f"\n  [{name}]  kp={k:g}")
            for key, val in step_metrics(r, args.step, args.step_at).items():
                print(f"    {key:20s} {val:9.4f}")
            if zeta < 1.0:
                print(f"    theory wn={wn:.1f} rad/s zeta={zeta:.2f} "
                      f"-> tr(10-90)≈{2.2 / wn:.3f} s, OS≈{100 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2)):.0f}%")
            else:
                print(f"    theory wn={wn:.1f} rad/s zeta={zeta:.2f} (over-damped, no overshoot)")

            if args.csv:
                out = args.csv.with_name(f"{args.csv.stem}_{name}{args.csv.suffix}")
                np.savetxt(out, np.column_stack(
                    [r.t, r.w, r.w_des, r.tau, r.thrust, r.saturated.astype(float)]),
                    delimiter=",", header="t,w,w_des,tau,thrust,saturated", comments="")
                print(f"    csv -> {out}")

    # ---- sine ----------------------------------------------------------
    sine_results = {}
    if args.mode in ("sine", "both"):
        print(f"\n=== rate sine sweep, amplitude {args.amplitude:g} rad/s ===")
        for name in names:
            i = AXES[name]
            k = kp[i]
            rows = []
            print(f"\n  [{name}]  kp={k:g}")
            print(f"    {'f [Hz]':>7} {'gain':>7} {'gain [dB]':>10} "
                  f"{'lag [deg]':>10} {'sat [%]':>8}")
            for f in freqs:
                period = 1.0 / f
                duration = max(10.0, 6.0 * period)
                t_start = max(4.0, 3.0 * period)
                ref = lambda t, f=f: args.amplitude * np.sin(2.0 * np.pi * f * t)
                r = simulate(params, args.altitude, i, ref, duration, args.rate, k)
                mag, lag = sine_fit(r, f, t_start)
                gain = mag / args.amplitude
                sat = 100.0 * r.saturated[r.t >= t_start].mean()
                rows.append((f, gain, lag, sat))
                print(f"    {f:7.3f} {gain:7.3f} "
                      f"{20 * np.log10(max(gain, 1e-9)):10.2f} {lag:10.1f} {sat:8.1f}")

            fs = np.array([x[0] for x in rows])
            gains = np.array([x[1] for x in rows])
            sine_results[name] = rows
            bw = bandwidth(fs, gains)

            print(f"\n    measured  -3 dB bandwidth: {bw:.3f} Hz "
                  f"({2 * np.pi * bw:.1f} rad/s)")
            for label, tau_m in (("tau_up", params.tau_up),
                                 ("mean", 0.5 * (params.tau_up + params.tau_down)),
                                 ("tau_down", params.tau_down)):
                wn, zeta, g_th, _ = theory(k, inertia[i], tau_m, fs)
                g_th = np.abs(g_th)
                print(f"    theory[{label:8s}] bw = {bandwidth(fs, g_th):.3f} Hz   "
                      f"wn = {wn:5.1f} rad/s  zeta = {zeta:4.2f}")

            # 与二阶模型对比（只在模型可信的频段，避开 100 Hz 离散化误差）
            wn, zeta, g_th, lag_th = theory(k, inertia[i],
                                            0.5 * (params.tau_up + params.tau_down), fs)
            keep = fs <= 0.4 * args.rate
            if keep.any():
                eg = np.abs(gains[keep] - g_th[keep]).max()
                el = np.abs(np.array([x[2] for x in rows])[keep] - lag_th[keep]).max()
                print(f"    |measured - theory| (f <= {0.4 * args.rate:.0f} Hz): "
                      f"gain {eg:.3f},  lag {el:.1f} deg")

    # ---- plot ----------------------------------------------------------
    if args.plot and (args.mode in ("step", "both") or sine_results):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cols = {"roll": "tab:red", "pitch": "tab:green", "yaw": "tab:blue"}
        marks = {"roll": "o", "pitch": "x", "yaw": "^"}
        # roll and pitch share kp/I == 20 rad/s, so they overlay exactly; dash
        # pitch to keep both visible.
        styles = {"roll": "-", "pitch": "--", "yaw": "-"}

        panels = ["step"] if args.mode in ("step", "both") else []
        panels += ["gain", "phase"] if sine_results else []
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4),
                                 squeeze=False)
        ax = {name: axes[0][j] for j, name in enumerate(panels)}

        if "step" in ax:
            for name in names:
                i = AXES[name]
                k = kp[i]
                ref = lambda t, st=args.step: 0.0 if t < args.step_at else st
                r = simulate(params, args.altitude, i, ref,
                             args.step_at + args.hold, args.rate, k)
                ax["step"].plot(r.t - args.step_at, r.w, styles[name],
                                color=cols[name], label=name)
            ax["step"].axhline(args.step, ls="--", c="k", lw=0.8)
            ax["step"].set(xlabel="t - t_step [s]", ylabel="rate [rad/s]",
                           title="rate step response")
            ax["step"].legend()
            ax["step"].grid(alpha=0.3)

        for name, rows in sine_results.items():
            i = AXES[name]
            k = kp[i]
            fs = np.array([x[0] for x in rows])
            db = np.array([20 * np.log10(max(x[1], 1e-9)) for x in rows])
            lag = np.array([x[2] for x in rows])
            ax["gain"].semilogx(fs, db, marks[name], color=cols[name], label=f"{name} meas")
            ax["phase"].semilogx(fs, lag, marks[name], color=cols[name], label=f"{name} meas")

            ft = np.logspace(np.log10(fs[0]), np.log10(fs[-1]), 100)
            _, _, g_th, l_th = theory(k, inertia[i],
                                      0.5 * (params.tau_up + params.tau_down), ft)
            ax["gain"].semilogx(ft, 20 * np.log10(np.abs(g_th)), "--",
                                color=cols[name], lw=0.9)
            ax["phase"].semilogx(ft, l_th, "--", color=cols[name], lw=0.9)

        if "gain" in ax:
            ax["gain"].axhline(-3, ls=":", c="k", lw=0.8)
            ax["gain"].set(xlabel="f [Hz]", ylabel="gain [dB]",
                           title="rate loop gain (meas / theory --)")
            ax["phase"].set(xlabel="f [Hz]", ylabel="phase lag [deg]",
                            title="rate loop phase (meas / theory --)")
            for key in ("gain", "phase"):
                ax[key].grid(alpha=0.3, which="both")
                ax[key].legend(fontsize=8)

        fig.tight_layout()
        out = Path.cwd() / "rate_loop_step_sine.png"
        fig.savefig(out, dpi=120)
        print(f"\nplot -> {out.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
