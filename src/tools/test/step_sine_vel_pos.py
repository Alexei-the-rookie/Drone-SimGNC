# Step / Sine Velocity and Position Loop Test
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise the horizontal velocity loop and the position loop built on it.

Wiring, only the layers this tool claims to test:

    pos mode:  p_des --> PositionController --> v_des
              v_des --> VelocityController --> a_des[0:2]
              a_des --> AttitudeController --> RateController --> Mixer

    vel mode:  v_des ----------------------> VelocityController --> ...

    cascade:   the real `CascadeController`, so the wiring in
               quadfc/control/cascade.py is covered as well.  Needs
               `cascade.p_des` (shape (2,), world frame x/y).

The altitude loop is muted by default (`a_des[2] == 0`) for the same reason as
`step_sine_att.py`: tilting drops the vertical thrust component, the altitude
loop raises the collective, and the mixer saturates -- which hides the outer
loop's own limits.  `--with-altitude` re-enables it for the joint test.

Theory
------
The inner stack (attitude + rate + mixer) is the third-order loop derived in
docs/altitude_loop.md §6.6:

    G_i(s) = a0 / (s^3 + a2 s^2 + a1 s + a0),   a0 = K wn^2, a1 = wn^2, a2 = 2 zeta wn
    wn = sqrt(kp_rate / (I tau_up)),  zeta = 0.5 sqrt(I / (kp_rate tau_up))

Closing the outer loops around it gives, exactly (no further approximation):

    velocity:  v/v_des = kv a0 / (s^4 + a2 s^3 + a1 s^2 + a0 s + kv a0)
    position:  p/p_des = kv kp a0 / (s^5 + a2 s^4 + a1 s^3 + a0 s^2 + kv a0 s + kv kp a0)

Both are evaluated here by residues, so the printed theory is what the plant
should do, not a thumb estimate.

Design rule (first-order approximation of G_i, good to ~8 %):
with G_i ~= 1/(1 + tau_i s), tau_i = tan(-angle G_i)/w at 0.1 Hz,

    velocity:  wn = sqrt(kv / tau_i),   zeta = 1 / (2 sqrt(kv tau_i))

so the velocity loop is *faster* than the naive 1/kv first-order estimate,
because the inner lag rotates kv into the negative imaginary direction and
partly cancels jw.  kv = 2 lands at zeta ~= 1.  Ignoring tau_i (i.e. using
1/kv) overestimates the rise time by ~40 %, which is why this loop looks
"faster than it should" the first time it is measured.

Both loops are pure P.  Expect a steady-state error only if the plant has drag;
there is none here, so `steady_err` must stay at zero -- a non-zero value means
an integrator leaked in somewhere.
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

ROOT = Path(__file__).resolve().parent.parent.parent  # .../src

# name -> (a_des index, euler index, sign of the resulting tilt)
# a_des x tilts about +y (pitch > 0); a_des y tilts about -x, so the same
# positive acceleration shows up as a NEGATIVE roll (ENU + FLU + ZYX).
AXES = {"x": (0, 1, +1.0), "y": (1, 0, -1.0)}

TILT_MAX_DEG = 30.0  # default lateral authority clamp, see CascadeController


def _load(name: str):
    """Import an outer loop lazily so --help works before the code lands."""
    if name == "velocity":
        try:
            from quadfc.control.velocity import VelocityController

            return VelocityController
        except ImportError as exc:  # pragma: no cover - only before the code lands
            raise SystemExit(
                f"missing quadfc/control/velocity.py: {exc}\n"
                "see docs/velocity_position_loop.md §2."
            )
    try:
        from quadfc.control.position import PositionController

        return PositionController
    except ImportError as exc:  # pragma: no cover - only before the code lands
        raise SystemExit(
            f"missing quadfc/control/position.py: {exc}\n"
            "see docs/velocity_position_loop.md §2."
        )


# ---- theory --------------------------------------------------------------


def _construct(name: str, gain: float | None):
    """Instantiate an outer loop, optionally passing an explicit gain.

    Falling back to a bare `Cls()` when no gain is given is what lets the tool
    read a loop's default gain from the loop itself instead of duplicating it.
    """
    cls = _load(name)
    try:
        return cls() if gain is None else cls(gain)
    except TypeError as exc:
        raise SystemExit(
            f"{cls.__name__}({'kp' if gain is None else '...'}) is not constructible "
            f"this way: {exc}\n"
            f"give `kp` a default (docs/velocity_position_loop.md §2) or pass "
            f"--kv/--kp explicitly."
        )


def _builtin_gain(name: str) -> float:
    """A loop's own default gain, so the tool cannot drift from the controller."""
    inst = _construct(name, None)
    return float(np.ravel(inst.kp)[0])


def inner_tf(params: QuadParams):
    """Attitude+rate+mixer as a0/(s^3 + a2 s^2 + a1 s + a0), pitch axis.

    Same chain as `step_sine_att.inner_loop` + `attitude_tf`, so the numbers
    here cannot drift from the attitude loop's own theory.
    """
    I = float(np.diag(params.inertia)[1])  # pitch, the x-acceleration axis
    kp_rate = RateController(params).kp[1]
    wn = float(np.sqrt(kp_rate / (I * params.tau_up)))
    zeta = float(0.5 * np.sqrt(I / (kp_rate * params.tau_up)))
    K = float(AttitudeController().kp[1])
    return K * wn**2, wn**2, 2 * zeta * wn


def inner_lag(params: QuadParams) -> float:
    """Equivalent first-order lag of the inner stack [s]; diagnostic only."""
    a0, a1, a2 = inner_tf(params)
    w = 2.0 * np.pi * 0.1
    s = 1j * w
    return float(np.tan(-np.angle(a0 / (s**3 + a2 * s**2 + a1 * s + a0))) / w)


def tf_step(num, den, duration=20.0, npts=40001):
    """Step response of H(s) = num(s)/den(s), coefficients highest power first."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    dden = np.polyder(den)
    h0 = np.polyval(num, 0.0) / np.polyval(den, 0.0)
    t = np.linspace(0.0, duration, npts)
    y = h0 * np.ones_like(t, dtype=complex)
    for p in np.roots(den):
        # the step contributes a pole at s = 0, hence the extra /p
        y += (np.polyval(num, p) / (p * np.polyval(dden, p))) * np.exp(p * t)
    return t, y.real


def tf_metrics(num, den, duration=20.0):
    t, y = tf_step(num, den, duration)
    i10, i90 = np.flatnonzero(y >= 0.1), np.flatnonzero(y >= 0.9)
    rise = float(t[i90[0]] - t[i10[0]]) if i10.size and i90.size else np.nan
    out = np.flatnonzero(np.abs(y - 1.0) > 0.02)
    settle = float(t[out[-1] + 1]) if out.size and out[-1] + 1 < t.size else np.nan
    return {"rise": rise, "overshoot": 100 * max(0.0, y.max() - 1.0),
            "settle": settle}


def vel_shape(kv, tau):
    """Design pair from the first-order approximation tau s^2 + s + kv."""
    return float(np.sqrt(kv / tau)), float(0.5 / np.sqrt(kv * tau))


def pos_shape(kv, kp):
    """Design pair of s^2 + kv s + kv kp (the tau s^3 term only detunes it)."""
    wn = float(np.sqrt(kv * kp))
    return wn, float(kv / (2 * wn))


def vel_tf(kv, inner):
    """v/v_des = kv a0 / (s^4 + a2 s^3 + a1 s^2 + a0 s + kv a0)."""
    a0, a1, a2 = inner
    return [kv * a0], [1.0, a2, a1, a0, kv * a0]


def pos_tf(kv, kp, inner):
    """p/p_des = kv kp a0 / (s^5 + a2 s^4 + a1 s^3 + a0 s^2 + kv a0 s + kv kp a0)."""
    a0, a1, a2 = inner
    return [kv * kp * a0], [1.0, a2, a1, a0, kv * a0, kv * kp * a0]


def tf_gain(num, den, f):
    s = 1j * 2 * np.pi * f
    h = np.polyval(num, s) / np.polyval(den, s)
    return float(abs(h)), float(-np.degrees(np.angle(h)))


def bandwidth(fs, gains):
    db = 20.0 * np.log10(np.maximum(gains, 1e-12))
    below = np.flatnonzero(db <= -3.0)
    if not below.size or below[0] == 0:
        return float("nan")
    i = below[0]
    return float(np.interp(-3.0, [db[i], db[i - 1]], [fs[i], fs[i - 1]]))


# ---- simulation ----------------------------------------------------------


@dataclass
class Run:
    t: np.ndarray
    ref: np.ndarray        # position [m] or velocity [m/s] setpoint
    y: np.ndarray          # measured position [m] or velocity [m/s]
    a_cmd: np.ndarray      # a_des on the excited axis, after the clamp
    tilt: np.ndarray       # measured tilt on the corresponding axis [deg]
    thrust: np.ndarray     # collective thrust [N]
    saturated: np.ndarray


class Loop:
    """Position -> velocity -> attitude -> rate -> mixer, wired like the cascade."""

    def __init__(self, params, loop: str, tilt_max_deg: float,
                 with_altitude: bool = False, kv=None, kp=None):
        self.params = params
        self.loop = loop
        self.vel = _construct("velocity", kv)
        self.pos = _construct("position", kp) if loop == "pos" else None
        self.att = AttitudeController()
        self.rate = RateController(params)
        self.mix = MixerController(params)
        self.a_max = G * np.tan(np.radians(tilt_max_deg))
        self.a_des = np.zeros(3)
        self.v_des = np.zeros(2)
        self.p_des = np.zeros(2)
        self.with_altitude = with_altitude
        if with_altitude:
            from quadfc.control.altitude import AltitudeController

            self.alt = AltitudeController()

    def reset(self, state) -> None:
        self.rate.reset()
        self.vel.reset()
        if self.pos is not None:
            self.pos.reset()
        if self.with_altitude:
            self.alt.reset(state.position[2])

    def step(self, state, dt: float) -> np.ndarray:
        if self.pos is not None:
            self.v_des[:] = self.pos.step(state.position[:2], self.p_des, dt)
        a_xy = np.asarray(self.vel.step(state.velocity[:2], self.v_des, dt), float)
        # Clamp before the force is formed: past this point the mixer can only
        # report saturation, it cannot refuse the command (docs §9.5-1).
        self.a_des[:2] = np.clip(a_xy, -self.a_max, self.a_max)
        if self.with_altitude:
            self.a_des[2] = self.alt.step(state.position[2], state.velocity[2], dt)
        f_des = self.params.mass * (self.a_des + np.array([0.0, 0.0, G]))
        w_des = self.att.step(state.rotation(), f_des, 0.0)
        tau = self.rate.step(state.omega, w_des, dt)
        return self.mix.step(float(np.linalg.norm(f_des)), tau)


def simulate(params, loop, axis, ref_pos, ref_vel, duration, rate, *,
             tilt_max_deg=TILT_MAX_DEG, with_altitude=False, kv=None, kp=None,
             z0=2.0):
    """Run one episode.  `loop` is 'vel', 'pos' or 'cascade'."""
    ai, ei, _ = AXES[axis]
    dt = 1.0 / rate
    n = int(np.ceil(duration / dt))

    if loop == "cascade":
        from quadfc.control.cascade import CascadeController

        ctrl = CascadeController(params)
        if not hasattr(ctrl, "p_des"):
            raise SystemExit(
                "CascadeController has no `p_des`; add it (shape (2,)) or use "
                "--loop vel/pos."
            )
        if kv is not None or kp is not None:
            print("note: --kv/--kp are ignored by --loop cascade")
        plant = AnalyticPlant(params, substep=1e-3, altitude=z0)
        s = plant.reset()
        ctrl.reset(s)
    else:
        plant = AnalyticPlant(params, substep=1e-3, altitude=z0)
        ctrl = Loop(params, loop, tilt_max_deg, with_altitude=with_altitude,
                    kv=kv, kp=kp)
        if kv is not None:
            ctrl.vel.kp[:] = kv
        if kp is not None and ctrl.pos is not None:
            ctrl.pos.kp[:] = kp
        s = plant.reset()
        ctrl.reset(s)
    t = np.empty(n)
    ref = np.empty(n)
    y = np.empty(n)
    a_cmd = np.empty(n)
    tilt = np.empty(n)
    thrust = np.empty(n)
    sat = np.empty(n, dtype=bool)

    for i in range(n):
        if loop == "vel":
            v_des = np.array([ref_vel(s.t), s.velocity[1]])
            ctrl.v_des[:] = v_des
        else:
            p_des = np.array(ref_pos(s.t), float)
            if loop == "cascade":
                ctrl.p_des[:] = p_des
            else:
                ctrl.p_des[:] = p_des

        u = np.asarray(ctrl.step(s, dt), dtype=float)
        c = np.clip(u, 0.0, params.omega_max)
        wrench = params.allocation @ (c ** 2)

        t[i] = s.t
        ref[i] = ref_vel(s.t) if loop == "vel" else float(np.asarray(ref_pos(s.t))[ai])
        y[i] = s.velocity[ai] if loop == "vel" else s.position[ai]
        a_cmd[i] = _read_a_cmd(ctrl, loop, ai)
        tilt[i] = np.degrees(s.euler()[ei])
        thrust[i] = wrench[0]
        sat[i] = _read_saturated(ctrl)
        s = plant.step(c, dt)

    return Run(t, ref, y, a_cmd, tilt, thrust, sat)


def _read_a_cmd(ctrl, loop, ai):
    if loop == "cascade":
        return float(getattr(ctrl, "a_des", np.zeros(3))[ai])
    return float(ctrl.a_des[ai])


def _read_saturated(ctrl):
    mix = ctrl.mix if hasattr(ctrl, "mix") else getattr(ctrl, "mixer", None)
    return bool(getattr(mix, "saturated", False))


# ---- metrics -------------------------------------------------------------


def step_metrics(r, target, t_step, *, tail=0.5):
    sel = r.t >= t_step
    tt, yy = r.t[sel] - t_step, r.y[sel]

    def first_cross(frac):
        hit = np.flatnonzero(yy >= frac * target) if target > 0 else \
              np.flatnonzero(yy <= frac * target)
        return float(tt[hit[0]]) if hit.size else float("nan")

    t10, t90 = first_cross(0.1), first_cross(0.9)
    peak = float(yy.max() if target > 0 else yy.min())
    outside = np.flatnonzero(np.abs(yy - target) > 0.02 * abs(target))
    settle = float(tt[outside[-1] + 1]) if outside.size and outside[-1] + 1 < tt.size else np.nan
    tail_sel = tt >= tt[-1] - tail
    return {
        "t_rise_10_90 [s]": t90 - t10,
        "overshoot [%]": 100.0 * (peak - target) / target,
        "t_settle_2% [s]": settle,
        "steady_err": float(yy[tail_sel].mean() - target),
        "peak": peak,
        "sat [%]": 100.0 * r.saturated[sel].mean(),
    }


def sine_fit(r, f, t_start):
    sel = r.t >= t_start
    tt, yy = r.t[sel], r.y[sel]
    w = 2.0 * np.pi * f
    X = np.column_stack([np.ones_like(tt), np.sin(w * tt), np.cos(w * tt)])
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
    return float(np.hypot(coef[1], coef[2])), float(-np.degrees(np.arctan2(coef[2], coef[1])))


# ---- reporting -----------------------------------------------------------

CHECKS: list[tuple[str, float, float, float]] = []  # name, measured, theory, tol(frac)


def check(name, measured, theory, tol, *, unit=""):
    ok = np.isfinite(measured) and np.isfinite(theory) and theory != 0.0 and \
        abs(measured - theory) <= tol * abs(theory)
    CHECKS.append((name, measured, theory, ok))
    flag = "ok  " if ok else "FAIL"
    print(f"  [{flag}] {name:26s} meas={measured:9.4f}{unit}  "
          f"theory={theory:9.4f}{unit}  tol=±{100 * tol:.0f}%")


def check_abs(name, measured, limit, *, unit=""):
    ok = bool(np.isfinite(measured)) and abs(measured) <= limit
    CHECKS.append((name, measured, limit, ok))
    flag = "ok  " if ok else "FAIL"
    print(f"  [{flag}] {name:26s} meas={measured:9.4f}{unit}  |x|<={limit:g}{unit}")


def report_footer() -> int:
    bad = [c for c in CHECKS if not c[3]]
    print(f"\n{len(CHECKS) - len(bad)}/{len(CHECKS)} checks passed")
    if bad:
        print("failed: " + ", ".join(c[0] for c in bad))
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--loop", default="pos", help="逗号分隔: vel,pos,cascade")
    ap.add_argument("--mode", choices=["step", "sine", "both"], default="both")
    ap.add_argument("--params", type=Path, default=ROOT / "config" / "x3.yaml")
    ap.add_argument("--axis", default="x", choices=sorted(AXES))
    ap.add_argument("--altitude", type=float, default=2.0, help="起始高度 [m]")
    ap.add_argument("--step-v", type=float, default=0.5, help="速度阶跃 [m/s]")
    ap.add_argument("--step-p", type=float, default=1.0, help="位置阶跃 [m]")
    ap.add_argument("--step-at", type=float, default=1.0, help="阶跃时刻 [s]")
    ap.add_argument("--hold", type=float, default=8.0, help="阶跃后仿真时长 [s]")
    ap.add_argument("--amp-v", type=float, default=0.2, help="速度正弦幅值 [m/s]")
    ap.add_argument("--amp-p", type=float, default=0.2, help="位置正弦幅值 [m]")
    ap.add_argument("--freqs", default="0.05,0.08,0.10,0.15,0.20,0.30,0.50")
    ap.add_argument("--rate", type=float, default=100.0, help="控制频率 [Hz]")
    ap.add_argument("--kv", type=float, default=None, help="临时覆盖速度环 kp")
    ap.add_argument("--kp", type=float, default=None, help="临时覆盖位置环 kp")
    ap.add_argument("--tilt-max", type=float, default=TILT_MAX_DEG,
                    help="横向加速度限幅对应的最大倾角 [deg]")
    ap.add_argument("--with-altitude", action="store_true",
                    help="同时开高度环（默认关闭以隔离外环）")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--no-check", action="store_true", help="只打印，不判定 PASS/FAIL")
    args = ap.parse_args(argv)

    params = QuadParams.from_yaml(args.params)
    inner = inner_tf(params)
    tau = inner_lag(params)
    loops = [s.strip() for s in args.loop.split(",") if s.strip()]
    for name in loops:
        if name not in ("vel", "pos", "cascade"):
            raise SystemExit(f"unknown --loop '{name}'")

    print(f"hover_omega = {params.hover_omega:7.1f} rad/s   "
          f"mass = {params.mass:.3f} kg")
    print(f"inner stack: 3rd order, equivalent 1st-order lag tau_i = {tau:.4f} s "
          f"(design rule only)")
    print(f"lateral clamp: {args.tilt_max:g} deg = "
          f"{G * np.tan(np.radians(args.tilt_max)):.2f} m/s^2")
    if "cascade" in loops:
        print("altitude loop: always live in --loop cascade, "
              f"{'ON' if args.with_altitude else 'OFF'} in vel/pos")
    else:
        print(f"altitude loop {'ON' if args.with_altitude else 'OFF'}")

    rc = 0
    for name in loops:
        if name != "cascade":
            kv_def = args.kv if args.kv is not None else _builtin_gain("velocity")
            kp_def = (args.kp if args.kp is not None
                      else (_builtin_gain("position") if name == "pos" else float("nan")))
        else:
            from quadfc.control.cascade import CascadeController

            c = CascadeController(params)
            missing = [a for a in ("p_des", "vel", "pos", "a_des")
                       if not hasattr(c, a)]
            if missing:
                raise SystemExit(
                    "CascadeController is missing "
                    + ", ".join(f"`{a}`" for a in missing)
                    + "\nsee docs/velocity_position_loop.md §4 (p_des/vel/pos are "
                    "the wiring, a_des is only read for logging), or use "
                    "--loop vel/pos." 
                )
            kv_def = float(c.vel.kp[0])
            kp_def = float(c.pos.kp[0])
        kv, kp = kv_def, kp_def

        if name == "vel":
            num, den = vel_tf(kv, inner)
            th = tf_metrics(num, den)
            wn_th, zeta_th = vel_shape(kv, tau)
            print(f"\n=== velocity loop: kv={kv:g} -> "
                  f"wn={wn_th:.2f} rad/s zeta={zeta_th:.2f} ===")
            target, stepv, amp = args.step_v, args.step_v, args.amp_v
        else:
            num, den = pos_tf(kv, kp, inner)
            th = tf_metrics(num, den)
            wn_th, zeta_th = pos_shape(kv, kp)
            print(f"\n=== {'cascade' if name == 'cascade' else 'position loop'}: "
                  f"kv={kv:g} kp={kp:g} -> wn={wn_th:.2f} rad/s "
                  f"zeta={zeta_th:.2f} (design pair) ===")
            target, stepv, amp = args.step_p, args.step_p, args.amp_p

        print(f"  theory: rise={th['rise']:.3f} s  overshoot={th['overshoot']:.2f} %  "
              f"settle={th['settle']:.3f} s")

        # ---- step ----------------------------------------------------------
        if args.mode in ("step", "both"):
            t_end = args.step_at + args.hold
            if name == "vel":
                r = simulate(params, "vel", args.axis, None,
                             lambda t, v=stepv: 0.0 if t < args.step_at else v,
                             t_end, args.rate, tilt_max_deg=args.tilt_max,
                             with_altitude=args.with_altitude, kv=kv)
            else:
                ai = AXES[args.axis][0]
                r = simulate(
                    params, name, args.axis,
                    lambda t, ai=ai, v=stepv: (
                        np.zeros(2) if t < args.step_at
                        else np.eye(2)[ai] * v),
                    None, t_end, args.rate, tilt_max_deg=args.tilt_max,
                    with_altitude=args.with_altitude,
                    z0=args.altitude, kv=kv, kp=kp)
            m = step_metrics(r, target, args.step_at)
            print(f"  --- step 0 -> {target:g} at t={args.step_at:g}s "
                  f"({'m/s' if name == 'vel' else 'm'}) ---")
            print(f"    measured rise={m['t_rise_10_90 [s]']:.3f} s  "
                  f"overshoot={m['overshoot [%]']:.2f} %  "
                  f"settle={m['t_settle_2% [s]']:.3f} s  "
                  f"peak tilt={np.abs(r.tilt).max():.2f} deg  "
                  f"sat={m['sat [%]']:.2f} %")
            a_max = G * np.tan(np.radians(args.tilt_max))
            clamped = bool(np.any(np.abs(r.a_cmd) >= a_max * 0.999))
            if clamped:
                print(f"    note: lateral clamp ({a_max:.2f} m/s^2) was active, so the "
                      f"linear theory no longer applies -- checks skipped")
            if not args.no_check and not clamped:
                check("rise 10-90", m["t_rise_10_90 [s]"], th["rise"], 0.15, unit=" s")
                check("settle 2%", m["t_settle_2% [s]"], th["settle"], 0.20, unit=" s")
                check_abs("overshoot", m["overshoot [%]"], max(8.0, 3 * th["overshoot"]),
                          unit=" %")
                check_abs("steady state error",
                          m["steady_err"] / target, 0.02)
                # sign convention: +x must pitch nose-up, +y must roll negative
                _, _, sign = AXES[args.axis]
                peak_tilt = float(r.tilt[int(np.argmax(np.abs(r.tilt)))])
                check_abs("tilt sign", np.sign(peak_tilt) - sign, 1.5)

        # ---- sine ----------------------------------------------------------
        if args.mode in ("sine", "both"):
            freqs = np.array([float(x) for x in args.freqs.split(",")])
            print(f"  --- sweep, amplitude {amp:g} "
                  f"({'m/s' if name == 'vel' else 'm'}) ---")
            print(f"    {'f [Hz]':>8} {'gain':>8} {'th':>8} {'lag':>9} {'th':>9} {'sat':>7}")
            gs, gt = [], []
            for f in freqs:
                period = 1.0 / f
                dur = max(5.0 * period, 8.0)
                t_start = max(2.0, 3.0 * period)
                if name == "vel":
                    r = simulate(params, "vel", args.axis, None,
                                 lambda t, f=f: amp * np.sin(2 * np.pi * f * t),
                                 dur, args.rate, tilt_max_deg=args.tilt_max,
                                 with_altitude=args.with_altitude, kv=kv)
                else:
                    ai = AXES[args.axis][0]
                    r = simulate(
                        params, name, args.axis,
                        lambda t, f=f, ai=ai: np.eye(2)[ai] * amp * np.sin(2 * np.pi * f * t),
                        None, dur, args.rate, tilt_max_deg=args.tilt_max,
                        with_altitude=args.with_altitude,
                        z0=args.altitude, kv=kv, kp=kp)
                mag, lag = sine_fit(r, f, t_start)
                g = mag / amp
                g_th, lag_th = tf_gain(num, den, f)
                gs.append(g)
                gt.append(g_th)
                sat = 100.0 * r.saturated[r.t >= t_start].mean()
                print(f"    {f:8.3f} {g:8.3f} {g_th:8.3f} {lag:9.1f} {lag_th:9.1f} "
                      f"{sat:7.1f}")

            bw_m, bw_t = bandwidth(freqs, np.array(gs)), bandwidth(freqs, np.array(gt))
            print(f"    -3 dB: measured {bw_m:.3f} Hz   theory {bw_t:.3f} Hz")
            if not args.no_check:
                check("-3 dB bandwidth", bw_m, bw_t, 0.12, unit=" Hz")
                g_mid = gs[len(gs) // 2]
                g_th_mid = gt[len(gt) // 2]
                check("gain @ mid sweep", g_mid, g_th_mid, 0.08)

        if args.csv:
            np.savetxt(args.csv, np.column_stack(
                [r.t, r.ref, r.y, r.a_cmd, r.tilt, r.thrust, r.saturated.astype(float)]),
                delimiter=",", header="t,ref,y,a_cmd,tilt_deg,thrust,saturated", comments="")
            print(f"    csv -> {args.csv}")

        # ---- plot ----------------------------------------------------------
        if args.plot:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
            axes[0].plot(r.t, r.ref, "--", c="k", lw=0.9, label="ref")
            axes[0].plot(r.t, r.y, c="tab:blue", label="meas")
            axes[0].set(xlabel="t [s]",
                        ylabel="v [m/s]" if name == "vel" else "p [m]",
                        title=f"{name}: step")
            axes[0].legend(fontsize=8)
            axes[1].plot(r.t, r.a_cmd, c="tab:orange")
            axes[1].set(xlabel="t [s]", ylabel="a_cmd [m/s^2]", title="commanded accel")
            axes[2].plot(r.t, r.tilt, c="tab:red")
            axes[2].set(xlabel="t [s]", ylabel="tilt [deg]", title="tilt")
            for ax in axes:
                ax.grid(alpha=0.3)
            fig.tight_layout()
            out = Path.cwd() / f"{name}_loop_step.png"
            fig.savefig(out, dpi=120)
            print(f"    plot -> {out.resolve()}")

    rc = report_footer() if not args.no_check else 0
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
