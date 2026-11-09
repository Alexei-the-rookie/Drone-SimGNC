# Step / Sine Attitude Loop Test
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise the thrust-vector attitude loop (B-plan) end to end.

The attitude layer has no setpoint of its own: tilt is a slave of the desired
force.  So this tool injects a lateral acceleration and measures the tilt, not
an angle command.

Wiring mirrors `CascadeController` exactly, except the altitude loop is muted
(kp = ki = kd = 0, so a_des[2] == 0).  Without that, a tilt change makes the
aircraft sink, the altitude loop raises the collective thrust, and the mixer
saturates -- which masks the attitude loop's own limits (see docs §6.9).

    a_des --> f_des = m(a_des + g e_z) --> R_des --> e_R --> w_des
           --> RateController --> MixerController --> motor speed

Modes
-----
step     lateral acceleration step, measure the tilt response
sine     acceleration sweep, measure tilt gain / phase against the 3rd-order model
recover  start tilted with a level force command, measure time to level
yaw      yaw setpoint step (saturation-limited: keep the step <= 10 deg)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # .../src

from quadfc.control.attitude import AttitudeController
from quadfc.control.mixer import MixerController
from quadfc.control.rate import RateController
from quadfc.params import G, QuadParams
from quadfc.plant.analytic import AnalyticPlant
from quadfc.state import euler_to_quat, rotmat_to_euler

ROOT = Path(__file__).resolve().parent.parent.parent  # .../src

# name -> (injected a_des index, measured euler index, applied torque index, sign)
# a_des x tilts about +y (pitch, tau_y); a_des y tilts about -x, so the same
# positive acceleration shows up as a NEGATIVE roll angle (ENU + FLU + ZYX).
AXES = {
    "pitch": (0, 1, 2, +1.0),
    "roll": (1, 0, 1, -1.0),
    "yaw": (None, 2, 3, +1.0),
}


@dataclass
class Run:
    t: np.ndarray
    ang: np.ndarray        # measured tilt / yaw [deg]
    ref: np.ndarray        # injected setpoint (ax [m/s^2], or yaw [deg])
    thrust: np.ndarray     # collective thrust [N]
    tau: np.ndarray        # axis torque actually applied [N.m]
    saturated: np.ndarray


class Loop:
    """The three inner layers, wired the same way `CascadeController` does."""

    def __init__(self, params, *, with_altitude: bool = False):
        self.params = params
        self.att = AttitudeController()
        self.rate = RateController(params)
        self.mix = MixerController(params)
        self.a_des = np.zeros(3)
        self.yaw_des = 0.0
        self.with_altitude = with_altitude
        if with_altitude:
            from quadfc.control.altitude import AltitudeController

            self.alt = AltitudeController()

    def reset(self, z0: float) -> None:
        self.rate.reset()
        if self.with_altitude:
            self.alt.reset(z0)

    def step(self, state, dt: float) -> np.ndarray:
        if self.with_altitude:
            self.a_des[2] = self.alt.step(state.position[2], state.velocity[2], dt)
        f_des = self.params.mass * (self.a_des + np.array([0.0, 0.0, G]))
        w_des = self.att.step(state.rotation(), f_des, self.yaw_des)
        tau = self.rate.step(state.omega, w_des, dt)
        self.f_des = f_des
        return self.mix.step(float(np.linalg.norm(f_des)), tau)


def simulate(params, axis, ref, duration, rate, kp, z0=2.0, rpy0=(0.0, 0.0, 0.0),
             with_altitude=False):
    plant = AnalyticPlant(params, substep=1e-3, altitude=z0)
    if any(rpy0):
        plant._x[6:10] = euler_to_quat(*rpy0)
    loop = Loop(params, with_altitude=with_altitude)
    if kp is not None:
        loop.att.kp[:] = kp

    s = plant.reset()
    if any(rpy0):
        plant._x[6:10] = euler_to_quat(*rpy0)
        s = plant.state
    loop.reset(z0)

    a_idx, e_idx, tau_idx, _ = AXES[axis]
    dt = 1.0 / rate
    n = int(np.ceil(duration / dt))
    t = np.empty(n)
    ang = np.empty(n)
    r = np.empty(n)
    thrust = np.empty(n)
    tau = np.empty(n)
    sat = np.empty(n, dtype=bool)

    for i in range(n):
        if a_idx is None:
            loop.yaw_des = np.radians(float(ref(s.t)))   # ref is in degrees
        else:
            loop.a_des[:] = 0.0
            loop.a_des[a_idx] = float(ref(s.t))
        u = np.asarray(loop.step(s, dt), dtype=float)
        c = np.clip(u, 0.0, params.omega_max)

        wrench = params.allocation @ (c**2)
        euler = rotmat_to_euler(s.rotation())
        t[i] = s.t
        ang[i] = np.degrees(euler[e_idx])
        r[i] = float(ref(s.t)) if a_idx is None else float(ref(s.t))
        thrust[i] = wrench[0]
        tau[i] = wrench[tau_idx]
        sat[i] = loop.mix.saturated
        s = plant.step(c, dt)

    return Run(t, ang, r, thrust, tau, sat)


# ---- theory --------------------------------------------------------------


def inner_loop(params, kp_rate, tau_m, axis_idx):
    """Rate loop seen as a 2nd-order system (docs §6.6)."""
    I = float(np.diag(params.inertia)[axis_idx])
    wn = float(np.sqrt(kp_rate / (I * tau_m)))
    zeta = float(0.5 * np.sqrt(I / (kp_rate * tau_m)))
    return wn, zeta


def attitude_tf(K, wn, zeta, freqs=None):
    """Attitude closed loop: s^3 + 2*zeta*wn s^2 + wn^2 s + K*wn^2.

    Cheaper than modelling the cascade as two first-order lags and matches the
    measured rate loop (docs §6.6), so the same numbers are reused here.
    """
    gains, lags, poles = [], [], np.roots([1.0, 2 * zeta * wn, wn**2, K * wn**2])
    for f in np.atleast_1d(freqs if freqs is not None else []):
        s = 1j * 2 * np.pi * f
        h = K * wn**2 / (s**3 + 2 * zeta * wn * s**2 + wn**2 * s + K * wn**2)
        gains.append(float(abs(h)))
        lags.append(float(-np.degrees(np.angle(h))))
    return poles, np.asarray(gains), np.asarray(lags)


def theory_step(K, wn, zeta, duration=4.0, npts=20000):
    """Denominator-normalised step response of the 3rd-order attitude loop."""
    coef = np.array([1.0, 2 * zeta * wn, wn**2, K * wn**2])
    poles = np.roots(coef)
    dcoef = np.polyder(coef)
    t = np.linspace(0.0, duration, npts)
    y = np.ones_like(t, dtype=complex)          # DC gain is 1
    for p in poles:
        y += (K * wn**2 / (p * np.polyval(dcoef, p))) * np.exp(p * t)
    return t, y.real


def theory_step_metrics(K, wn, zeta):
    t, y = theory_step(K, wn, zeta)
    i10, i90 = np.flatnonzero(y >= 0.1), np.flatnonzero(y >= 0.9)
    rise = float(t[i90[0]] - t[i10[0]]) if i10.size and i90.size else np.nan
    out = np.flatnonzero(np.abs(y - 1.0) > 0.02)
    settle = float(t[out[-1] + 1]) if out.size and out[-1] + 1 < t.size else np.nan
    return {"rise": rise, "overshoot": 100 * max(0.0, y.max() - 1.0), "settle": settle}


# ---- metrics -------------------------------------------------------------


def step_metrics(r, target, t_step):
    sel = r.t >= t_step
    tt, yy = r.t[sel] - t_step, r.ang[sel]
    if target == 0.0:
        raise SystemExit("target must be non-zero")

    def first_cross(frac):
        hit = np.flatnonzero(yy >= frac * target) if target > 0 else \
              np.flatnonzero(yy <= frac * target)
        return float(tt[hit[0]]) if hit.size else float("nan")

    t10, t90 = first_cross(0.1), first_cross(0.9)
    peak = float(yy.max() if target > 0 else yy.min())
    overshoot = 100.0 * (peak - target) / target
    outside = np.flatnonzero(np.abs(yy - target) > 0.02 * abs(target))
    settle = float(tt[outside[-1] + 1]) if outside.size and outside[-1] + 1 < tt.size else np.nan
    tail = yy[tt >= tt[-1] - 0.1]

    return {
        "t_rise_10_90 [s]": t90 - t10,
        "overshoot [%]": max(0.0, overshoot),
        "t_settle_2% [s]": settle,
        "steady_err [deg]": float(tail.mean() - target),
        "peak [deg]": peak,
        "tau_peak [N.m]": float(np.abs(r.tau[sel]).max()),
        "sat [%]": 100.0 * r.saturated[sel].mean(),
    }


def sine_fit(r, f, t_start):
    sel = r.t >= t_start
    tt, yy = r.t[sel], r.ang[sel]
    w = 2.0 * np.pi * f
    X = np.column_stack([np.ones_like(tt), np.sin(w * tt), np.cos(w * tt)])
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
    a, b = float(coef[1]), float(coef[2])
    return float(np.hypot(a, b)), float(-np.degrees(np.arctan2(b, a)))


def bandwidth(fs, gains):
    db = 20.0 * np.log10(np.maximum(gains, 1e-12))
    below = np.flatnonzero(db <= -3.0)
    if not below.size or below[0] == 0:
        return float("nan")
    i = below[0]
    return float(np.interp(-3.0, [db[i], db[i - 1]], [fs[i], fs[i - 1]]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode", default="all",
                    choices=["step", "sine", "recover", "yaw", "all"])
    ap.add_argument("--params", type=Path, default=ROOT / "config" / "x3.yaml")
    ap.add_argument("--axis", default="pitch", choices=sorted(AXES))
    ap.add_argument("--altitude", type=float, default=2.0, help="起始高度 [m]")
    ap.add_argument("--step", type=float, default=1.0, help="横向加速度阶跃 [m/s^2]")
    ap.add_argument("--step-at", type=float, default=0.3, help="阶跃时刻 [s]")
    ap.add_argument("--hold", type=float, default=1.0, help="阶跃后仿真时长 [s]")
    ap.add_argument("--amplitude", type=float, default=0.3, help="正弦幅值 [m/s^2]")
    ap.add_argument("--freqs", default="0.5,1,2,3,4,6,8")
    ap.add_argument("--yaw-step", type=float, default=10.0, help="yaw 阶跃 [deg]")
    ap.add_argument("--tilt", type=float, default=10.0, help="recover 模式初始倾角 [deg]")
    ap.add_argument("--rate", type=float, default=100.0, help="控制频率 [Hz]")
    ap.add_argument("--kp", type=float, default=None, help="临时覆盖三轴姿态 kp（标量）")
    ap.add_argument("--with-altitude", action="store_true",
                    help="同时开高度环（默认关闭以隔离姿态环）")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    if abs(args.step) < 1e-9:
        raise SystemExit("--step must be non-zero")

    params = QuadParams.from_yaml(args.params)
    att = AttitudeController()
    kp = att.kp.copy() if args.kp is None else np.full(3, args.kp)
    rate_kp = RateController(params).kp
    a_idx, e_idx, _, sign = AXES[args.axis]
    thrust_max = 4.0 * params.kf * params.omega_max**2

    print(f"hover_omega = {params.hover_omega:7.1f} rad/s   "
          f"thrust = {params.weight:6.3f} N hover / {thrust_max:6.3f} N max")
    print(f"altitude loop {'ON' if args.with_altitude else 'OFF'} "
          f"(a_des[2] = {'alt.step' if args.with_altitude else '0'})")
    print(f"axis = {args.axis}   attitude kp = {np.round(kp, 3)}")

    if a_idx is not None:
        wn, zeta = inner_loop(params, rate_kp[a_idx], params.tau_up, a_idx)
        print(f"inner rate loop (tau_up={params.tau_up:g} s): wn = {wn:5.1f} rad/s, zeta = {zeta:4.2f}")
        th = theory_step_metrics(kp[a_idx], wn, zeta)
        print(f"theory attitude loop @ K={kp[a_idx]:g}: rise {th['rise']:.3f} s  "
              f"overshoot {th['overshoot']:.2f} %  settle {th['settle']:.3f} s")
    else:
        wn, zeta = inner_loop(params, rate_kp[2], params.tau_up, 2)
        print(f"inner rate loop (tau_up={params.tau_up:g} s): wn = {wn:5.1f} rad/s, zeta = {zeta:4.2f}")

    # ---- step ----------------------------------------------------------
    if args.mode in ("step", "all") and a_idx is not None:
        target = sign * float(np.degrees(np.arctan(args.step / G)))
        ref = lambda t: 0.0 if t < args.step_at else args.step
        r = simulate(params, args.axis, ref, args.step_at + args.hold, args.rate, kp,
                     z0=args.altitude, with_altitude=args.with_altitude)
        print(f"\n=== {args.axis}: a = 0 -> {args.step:+.2f} m/s^2 at "
              f"t={args.step_at:.2f} s (target tilt {target:.3f} deg) ===")
        for key, val in step_metrics(r, target, args.step_at).items():
            print(f"  {key:20s} {val:9.4f}")

        if args.csv:
            np.savetxt(args.csv, np.column_stack(
                [r.t, r.ang, r.thrust, r.tau, r.saturated.astype(float)]),
                delimiter=",", header="t,ang_deg,thrust,tau,saturated", comments="")
            print(f"  csv -> {args.csv}")

    # ---- sine ----------------------------------------------------------
    sine_rows = []
    if args.mode in ("sine", "all") and a_idx is not None:
        wn, zeta = inner_loop(params, rate_kp[a_idx], params.tau_up, a_idx)
        print(f"\n=== {args.axis}: acceleration sweep, amplitude {args.amplitude:g} m/s^2 ===")
        print(f"  {'f [Hz]':>7} {'gain':>7} {'gain [dB]':>10} {'lag [deg]':>10} {'sat [%]':>8}")
        freqs = np.array([float(x) for x in args.freqs.split(",")])
        ref_tilt = np.degrees(np.arctan(args.amplitude / G))   # 线性化：atan(a/g) ≈ a/g
        for f in freqs:
            period = 1.0 / f
            duration = max(4.0, 6.0 * period)
            t_start = max(1.5, 3.0 * period)
            ref = lambda t, f=f: args.amplitude * np.sin(2.0 * np.pi * f * t)
            r = simulate(params, args.axis, ref, duration, args.rate, kp,
                         z0=args.altitude, with_altitude=args.with_altitude)
            mag, lag = sine_fit(r, f, t_start)
            gain = mag / ref_tilt
            sat = 100.0 * r.saturated[r.t >= t_start].mean()
            sine_rows.append((f, gain, lag, sat))
            print(f"  {f:7.3f} {gain:7.3f} "
                  f"{20 * np.log10(max(gain, 1e-9)):10.2f} {lag:10.1f} {sat:8.1f}")

        fs = np.array([x[0] for x in sine_rows])
        gs = np.array([x[1] for x in sine_rows])
        print(f"\n  measured -3 dB bandwidth: {bandwidth(fs, gs):.3f} Hz")
        _, g_th, _ = attitude_tf(kp[a_idx], wn, zeta, fs)
        print(f"  theory   -3 dB bandwidth: {bandwidth(fs, g_th):.3f} Hz")

    # ---- recover -------------------------------------------------------
    if args.mode in ("recover", "all") and a_idx is not None:
        rpy0 = [0.0, 0.0, 0.0]
        rpy0[e_idx] = np.radians(args.tilt)
        r = simulate(params, args.axis, lambda t: 0.0, 1.5, args.rate, kp,
                     z0=args.altitude, rpy0=tuple(rpy0), with_altitude=args.with_altitude)
        tol = 0.5
        inside = np.flatnonzero(np.abs(r.ang) < tol)
        print(f"\n=== recover: start at {args.axis} = {args.tilt:+.1f} deg, "
              f"level force command ===")
        print(f"  |angle| converges below {tol} deg at t = "
              f"{r.t[inside[0]]:.3f} s" if inside.size else "  did NOT converge")
        print(f"  final angle = {r.ang[-1]:+.4f} deg    peak |angle| = {np.abs(r.ang).max():.3f} deg")
        print(f"  sat = {100 * r.saturated.mean():.1f} %")

    # ---- yaw -----------------------------------------------------------
    if args.mode in ("yaw", "all"):
        r = simulate(params, "yaw", lambda t: args.yaw_step, 5.0, args.rate, kp,
                     z0=args.altitude, with_altitude=args.with_altitude)
        wn, zeta = inner_loop(params, rate_kp[2], params.tau_up, 2)
        print(f"\n=== yaw: setpoint 0 -> {args.yaw_step:+.1f} deg ===")
        print(f"  initial w_des = {kp[2] * np.radians(args.yaw_step):.4f} rad/s   "
              f"(yaw authority ~0.223 rad/s)")
        print(f"  final yaw = {r.ang[-1]:+.3f} deg    peak = {r.ang.max():+.3f} deg")
        for key, val in step_metrics(r, args.yaw_step, 0.0).items():
            print(f"  {key:20s} {val:9.4f}")
        print(f"  theory @ K={kp[2]:g}: wn={wn:.1f}, zeta={zeta:.2f} "
              f"-> {theory_step_metrics(kp[2], wn, zeta)}")

    # ---- plot ----------------------------------------------------------
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        panels = []
        if args.mode in ("step", "all") and a_idx is not None:
            panels.append("step")
        if sine_rows:
            panels += ["gain", "phase"]
        if args.mode in ("recover", "all") and a_idx is not None:
            panels.append("recover")
        if args.mode in ("yaw", "all"):
            panels.append("yaw")
        if panels:
            fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4),
                                     squeeze=False)
            ax = {n: axes[0][j] for j, n in enumerate(panels)}

            if "step" in ax:
                ref = lambda t: 0.0 if t < args.step_at else args.step
                r = simulate(params, args.axis, ref, args.step_at + args.hold, args.rate,
                             kp, z0=args.altitude, with_altitude=args.with_altitude)
                ax["step"].plot(r.t - args.step_at, r.ang, color="tab:blue")
                ax["step"].axhline(sign * np.degrees(np.arctan(args.step / G)),
                                    ls="--", c="k", lw=0.8)
                ax["step"].set(xlabel="t - t_step [s]", ylabel=f"{args.axis} [deg]",
                               title="tilt step response")
                ax["step"].grid(alpha=0.3)

            if "gain" in ax:
                wn, zeta = inner_loop(params, rate_kp[a_idx], params.tau_up, a_idx)
                fs = np.array([x[0] for x in sine_rows])
                db = np.array([20 * np.log10(max(x[1], 1e-9)) for x in sine_rows])
                ax["gain"].semilogx(fs, db, "o", color="tab:blue", label="meas")
                ft = np.logspace(np.log10(fs[0]), np.log10(fs[-1]), 100)
                _, g_th, l_th = attitude_tf(kp[a_idx], wn, zeta, ft)
                ax["gain"].semilogx(ft, 20 * np.log10(np.abs(g_th)), "--", lw=0.9, label="theory")
                ax["gain"].axhline(-3, ls=":", c="k", lw=0.8)
                ax["gain"].set(xlabel="f [Hz]", ylabel="gain [dB]", title="tilt gain")
                ax["gain"].grid(alpha=0.3, which="both")
                ax["gain"].legend(fontsize=8)
                lag_meas = np.unwrap(np.radians([x[2] for x in sine_rows]))
                ax["phase"].semilogx(fs, np.degrees(lag_meas), "o", color="tab:blue")
                ax["phase"].semilogx(ft, np.degrees(np.unwrap(np.radians(l_th))), "--", lw=0.9)
                ax["phase"].set(xlabel="f [Hz]", ylabel="phase lag [deg]", title="tilt phase")
                ax["phase"].grid(alpha=0.3, which="both")

            if "recover" in ax:
                rpy0 = [0.0, 0.0, 0.0]
                rpy0[e_idx] = np.radians(args.tilt)
                r = simulate(params, args.axis, lambda t: 0.0, 1.5, args.rate, kp,
                             z0=args.altitude, rpy0=tuple(rpy0),
                             with_altitude=args.with_altitude)
                ax["recover"].plot(r.t, r.ang, color="tab:red")
                ax["recover"].axhline(0, ls="--", c="k", lw=0.8)
                ax["recover"].set(xlabel="t [s]", ylabel=f"{args.axis} [deg]",
                                  title="tilt recovery")
                ax["recover"].grid(alpha=0.3)

            if "yaw" in ax:
                r = simulate(params, "yaw", lambda t: args.yaw_step, 5.0, args.rate, kp,
                             z0=args.altitude, with_altitude=args.with_altitude)
                ax["yaw"].plot(r.t, r.ang, color="tab:green")
                ax["yaw"].axhline(args.yaw_step, ls="--", c="k", lw=0.8)
                ax["yaw"].set(xlabel="t [s]", ylabel="yaw [deg]", title="yaw step")
                ax["yaw"].grid(alpha=0.3)

            fig.tight_layout()
            out = Path.cwd() / "attitude_loop_topics.png"
            fig.savefig(out, dpi=120)
            print(f"\nplot -> {out.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
