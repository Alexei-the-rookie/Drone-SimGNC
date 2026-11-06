# SPDX-License-Identifier: GPL-3.0-or-later

"""Plant backed by Gazebo Harmonic (gz-sim 8).

Talks to the simulator over gz-transport, no ROS involved:

    out  /X3/gazebo/command/motor_speed   gz.msgs.Actuators      (4 speeds, rad/s)
    in   /model/x3/odometry               gz.msgs.Odometry       100 Hz, pose + twist

`step()` publishes a command and blocks until the next odometry sample, so the
control loop self-synchronises to the simulator instead of free-running.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from gz.msgs10.actuators_pb2 import Actuators
from gz.msgs10.odometry_pb2 import Odometry
from gz.transport13 import Node

from quadfc.params import QuadParams
from quadfc.plant.base import Plant
from quadfc.state import State


class GzPlant(Plant):
    def __init__(
        self,
        params: QuadParams,
        world: str | Path,
        *,
        spawn: bool = True,
        gui: bool = False,
        motor_topic: str = "/X3/gazebo/command/motor_speed",
        odom_topic: str = "/model/x3/odometry",
        model_name: str = "x3",
        world_name: str = "quadcopter",
        altitude: float | None = None,
        startup_timeout: float = 90.0,
    ):
        super().__init__(params)
        self.world = Path(world)
        self.spawn = spawn
        self.gui = gui
        self.motor_topic = motor_topic
        self.odom_topic = odom_topic
        self.model_name = model_name
        self.world_name = world_name
        self.altitude = altitude
        self.startup_timeout = startup_timeout

        self._node: Node | None = None
        self._pub = None
        self._proc: subprocess.Popen | None = None
        self._cond = threading.Condition()
        self._latest: State | None = None
        self._seq = 0

    # ---- transport -------------------------------------------------------

    def _on_odometry(self, msg: Odometry) -> None:
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        quat = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
        )
        lin = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        ang = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])

        # gz's OdometryPublisher reports twist in the child (base) frame.
        from quadfc.state import quat_to_rotmat

        state = State(
            t=msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9,
            position=pos,
            velocity=quat_to_rotmat(quat) @ lin,
            quat=quat,
            omega=ang,
        )
        with self._cond:
            self._latest = state
            self._seq += 1
            self._cond.notify_all()

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._node is not None:
            return

        self._node = Node()
        self._pub = self._node.advertise(self.motor_topic, Actuators)
        self._node.subscribe(Odometry, self.odom_topic, self._on_odometry)

        if self.spawn:
            cmd = ["gz", "sim"]
            if not self.gui:
                cmd.append("-s")  # server only, no GUI
            cmd += ["-r", str(self.world)]
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

        deadline = threading.Event()
        timer = threading.Timer(self.startup_timeout, deadline.set)
        timer.start()
        try:
            with self._cond:
                while self._seq == 0:
                    if deadline.is_set():
                        raise RuntimeError(
                            f"no odometry on {self.odom_topic} after "
                            f"{self.startup_timeout:.0f}s.\n"
                            f"Is the world '{self.world}' running with the "
                            f"OdometryPublisher plugin?"
                        )
                    self._cond.wait(0.2)
        finally:
            timer.cancel()

    def close(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        self._node = None

    @property
    def state(self) -> State:
        with self._cond:
            if self._latest is None:
                raise RuntimeError("plant not started")
            return self._latest.copy()

    def reset(self) -> State:
        """Start an episode, teleporting to `self.altitude` if one was given."""
        if self.altitude is None:
            return self.state

        hover = np.full(self.params.n_rotors, self.params.hover_omega)
        # Spin the motors up *before* moving: with the rotors idle the aircraft
        # free-falls during the teleport round trip and starts every episode
        # with ~0.2 m/s of unwanted descent.
        for _ in range(20):
            self.step(hover, 0.01)

        self.set_pose((0.0, 0.0, self.altitude))
        # The first odometry sample after a teleport reports a bogus velocity
        # (it differentiates the jump), so throw it away.
        for _ in range(3):
            s = self.step(hover, 0.01)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            s = self.step(hover, 0.01)
            if abs(s.position[2] - self.altitude) < 0.05:
                break
        return s

    def set_pose(self, position, rpy=(0.0, 0.0, 0.0)) -> bool:
        """Teleport the model. Useful to start an episode airborne.

        Note: this does not clear linear/angular velocity; call it while the
        aircraft is at rest if you need a clean episode start.
        """
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.pose_pb2 import Pose

        from quadfc.state import euler_to_quat

        q = euler_to_quat(*rpy)
        req = Pose()
        req.name = self.model_name
        req.position.x, req.position.y, req.position.z = (float(v) for v in position)
        req.orientation.x, req.orientation.y, req.orientation.z, req.orientation.w = (
            float(v) for v in q
        )
        # NOTE: the "/blocking" variant of this service does not work through
        # the python transport binding (it always answers False), so use the
        # plain one and poll the state yourself if you need synchronisation.
        ok, _response = self._node.request(
            f"/world/{self.world_name}/set_pose", req, Pose, Boolean, 3000
        )
        return bool(ok)

    # ---- stepping --------------------------------------------------------

    def _wait_for(self, seq0: int, timeout: float) -> State:
        with self._cond:
            self._cond.wait_for(lambda: self._seq > seq0, timeout=timeout)
            if self._latest is None:
                raise RuntimeError("no odometry received")
            return self._latest.copy()

    def step(self, motor_cmd, dt: float) -> State:
        cmd = np.clip(np.asarray(motor_cmd, dtype=float), 0.0, self.params.omega_max)
        with self._cond:
            seq0 = self._seq

        msg = Actuators()
        msg.velocity.extend([float(v) for v in cmd])
        self._pub.publish(msg)

        return self._wait_for(seq0, timeout=max(1.0, 10.0 * dt))
