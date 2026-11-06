# quadfc

[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](./LICENSE)

手搓四旋翼飞控的**骨架**：把「控制器 / 被控对象 / 仿真环境」三层拆开，控制律留空由你填。

内置两个可互换的被控对象：

- **`analytic`** —— 纯 numpy 的 6-DOF 刚体 + 电机一阶惯性 + RK4，无外部依赖，比实时快一个数量级，用来快速迭代控制律；
- **`gz`** —— Gazebo Harmonic 的 X3 四旋翼，**电机转速级**接口（你直接发 4 路 ω，不是姿态指令），带真实的电机惯性、桨叶阻力、DART 刚体动力学。

两者共用同一套参数和同一套接口，可以互相验证：同一串开环指令下，1.2 s 机动后位置差 **2 cm**、姿态差 **0.46°**。这保证了「解析模型上调好的增益，搬到 Gazebo 里还成立」。

> 控制律本身（PID / LQR / MPC）**不在**这个仓库里，这里只提供它们该长什么样的插槽。

---

## 目录

- [为什么是这套架构](#为什么是这套架构)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
- [数据流](#数据流)
- [接口约定](#接口约定)
- [如何接入你的控制器](#如何接入你的控制器)
- [机体参数（X3）](#机体参数x3)
- [Gazebo 接口明细](#gazebo-接口明细)
- [双后端一致性验证](#双后端一致性验证)
- [已知的坑](#已知的坑)
- [路线图](#路线图)
- [许可与致谢](#许可与致谢)

---

## 为什么是这套架构

自己写飞控最容易陷进去的地方，是把「控制律」和「仿真器」搅在一起：改一次增益要重启一次 Gazebo，换个被控对象整个工程重写。

所以这里做了三条硬性切分：

1. **控制器只吐电机转速**。`step(state, dt) -> ndarray(4)`，单位 rad/s。混控、饱和、电机惯性全部属于被控对象，不在控制器里。
2. **被控对象是一个接口**。`state / reset / step`，Gazebo 和 numpy 模型都可以实现，runner 和控制器都不知道背后是谁。
3. **控制器不做 clip**。饱和统计由 runner 记录——增益离谱时会直接看到「30% 的样本饱和」，而不是被 clip 悄悄掩盖成「响应慢」。

这样得到一个实用性质：**毫秒级的 `analytic` 后端调参，Gazebo 后端只用来做最终验证**，而两者的一致性有脚本可查、有数字可对。

---

## 环境依赖

| 用途 | 依赖 | 说明 |
|---|---|---|
| 必需 | Python ≥ 3.10、numpy、PyYAML | `analytic` 后端只需要这些 |
| 可选 | Gazebo Harmonic (gz-sim 8) + Python 绑定 | `gz` 后端需要 `import gz.transport13`、`gz.msgs10.*` |
| 可选 | matplotlib | `--plot` 出图 |

gz 的 Python 绑定不走 pip。两条常见来源：

```bash
# A. 装了 ROS 2 Jazzy（自带 gz vendor 包，最省事）
source /opt/ros/jazzy/setup.bash        # 会把 gz vendor 的 python 路径加进 PYTHONPATH
python3 -c "import gz.transport13; print('ok')"
```

```bash
# B. 系统装的 Gazebo Harmonic
sudo apt install python3-gz-transport13 python3-gz-msgs10
```

装好后：

```bash
pip install -r requirements.txt
```

---

## 快速开始

不需要 colcon，也不需要建 catkin/ament 工作空间，直接跑：

```bash
cd src

# 1) 纯 numpy，0.3 秒跑完 2 秒仿真
python3 -m quadfc --backend analytic --duration 2 --altitude 3

# 2) Gazebo（首次运行会从 Fuel 下载 X3 模型，约 21 MB，缓存在 ~/.gz/fuel）
python3 -m quadfc --backend gz --duration 5 --altitude 2 --gui

# 3) 录制轨迹 + 出图
python3 -m quadfc --backend analytic --duration 5 --altitude 2 \
    --hover-scale 1.03 --csv run.csv --plot

# 4) 验证两个后端一致
python3 tools/compare_backends.py --duration 1.2 --altitude 1.0
```

常用参数：

| 参数 | 默认 | 作用 |
|---|---|---|
| `--backend {analytic,gz}` | `analytic` | 选被控对象 |
| `--controller NAME` | `hover` | 选控制律（目前只有 `hover`，见下文如何加） |
| `--duration` | 5.0 | 仿真时长 |
| `--rate` | 100 | 控制环频率 Hz |
| `--altitude` | 0 | 起始高度（`gz` 后端会 teleport 到该高度） |
| `--hover-scale` | 1.0 | 悬停油门倍率，>1 会缓慢爬升 |
| `--attach` | — | 接一个已经在跑的 `gz sim`，不自己起 |
| `--gui` | — | 开 Gazebo 图形界面 |
| `--no-realtime` | — | `analytic` 后端全速跑，不等实时 |
| `--csv` / `--plot` | — | 落盘轨迹 / 出三视图 |

运行结束会打印报告，饱和比例是最有用的一行：

```
steps=200  sim_time=4.00s  wall=2.00s
final position = [ -0.000  -0.000   2.073]
altitude error = +0.020 m   max tilt =  0.00 deg   saturated =   0.0% of samples
```

---

## 数据流

```
                       State
   ┌──────────┐  ──────────────────►  ┌──────────────┐
   │  Runner  │                       │  Controller  │   ← 你写这里
   │ 固定频率 │  ◄──────────────────  │  PID/LQR/MPC │
   └──────────┘   4×ω [rad/s]         └──────────────┘
        │  step(cmd, dt)
        ▼
   ┌──────────┐   gz:   gz-transport ──► gz.msgs.Actuators
   │  Plant   │   ───►  Gazebo Harmonic (DART, 1 ms 步长)
   │          │   ◄───  /model/x3/odometry (100 Hz)
   ├──────────┤
   │ analytic │   同一套方程，RK4，1 ms 子步
   └──────────┘
```

`Runner` 的循环（`quadfc/runner.py`）：

```python
state = plant.reset()
controller.reset(state)
while t < duration:
    cmd = controller.step(state, dt)          # 不做 clip
    log.append(state, cmd)
    state = plant.step(np.clip(cmd, 0, omega_max), dt)
```

`analytic` 的 `step()` 就是积分 `dt`；`gz` 的 `step()` 会**发布指令后阻塞等下一帧 odometry**，所以控制环自动跟仿真同步，不会自由狂奔（也让 `--rate` 在 gz 后端变成「上限」而非「定时器」）。

---

## 接口约定

全项目统一，写在 `quadfc/state.py` 里，别改：

| 项 | 约定 |
|---|---|
| 世界系 | ENU（x 东、y 北、z 上），重力 `-z` |
| 机体系 | FLU（x 前、y 左、z 上） |
| 四元数 | `(x, y, z, w)`，表示 **body → world** |
| 角速度 | 机体系 |
| 欧拉角 | ZYX 内旋，`State.euler()` 返回 `(roll, pitch, yaw)` |
| 电机指令 | `ndarray(4)`，单位 **rad/s**，顺序 `rotor_0..3`（见参数表） |

`State` 提供 `rotation()`、`euler()`、`body_velocity()`、`copy()`。

**控制器**（`quadfc/control/base.py`）：

```python
class Controller(ABC):
    name: str
    def reset(self, state: State) -> None: ...
    def step(self, state: State, dt: float) -> np.ndarray: ...   # (4,) rad/s
```

**被控对象**（`quadfc/plant/base.py`）：

```python
class Plant(ABC):
    state: State                     # 最新状态，不阻塞
    def start(self) -> None: ...
    def reset(self) -> State: ...
    def step(self, motor_cmd, dt) -> State: ...
    def close(self) -> None: ...
```

---

## 如何接入你的控制器

1. 新建 `src/quadfc/control/pid.py`（或 `lqr.py` / `mpc.py`），继承 `Controller`；
2. 在 `src/quadfc/cli.py` 的 `build_controller()` 里挂个名字；
3. `--controller pid` 跑起来。runner 和 plant 一行都不用改。

骨架长这样：

```python
import numpy as np
from quadfc.control.base import Controller
from quadfc.params import QuadParams
from quadfc.state import State


class CascadePID(Controller):
    name = "pid"

    def __init__(self, params: QuadParams):
        self.B = params.allocation          # u = [F, tx, ty, tz] = B @ w^2
        self.B_inv = np.linalg.inv(self.B)  # X 构型下 4x4 可逆
        self.omega_max = params.omega_max

    def reset(self, state: State) -> None:
        pass

    def step(self, state: State, dt: float) -> np.ndarray:
        # 1) 位置/速度环 -> 期望加速度 a_des
        # 2) 由 a_des 与期望偏航推出期望姿态 + 总推力 F
        #    （差分平坦：z_B = (a_des + g*e_z)/||.||，解析式三行）
        # 3) 姿态/角速率环 -> 期望力矩 tau
        u = np.array([F, *tau])
        w2 = self.B_inv @ u                 # -> 各电机 w^2
        return np.sqrt(np.clip(w2, 0.0, self.omega_max**2))
```

三个提醒：

- **`B` 就是混控矩阵**，`params.allocation` 的四行分别是 `[F, tx, ty, tz]` 对 `ω²` 的系数。符号写反了飞机会朝反方向滚，这是最常见的 bug。
- **反解出负的 `ω²` 说明指令不可达**（比如想滚转但已经贴到推力上限）。生产做法是在 QP/混控里做等比例缩放，而不是逐通道 `clip`——逐通道 clip 会破坏力矩方向，出现「想滚转却掉高」。
- **饱和要能被看见**。返回未裁剪的指令，runner 会统计；一旦 `saturated` 超过几个百分点，先改增益，别急着怪仿真。

---

## 机体参数（X3）

全部来自 Gazebo 官方模型的 SDF（`model.sdf` + `worlds/quadcopter.sdf` 的电机插件），落在 `src/config/x3.yaml`。

| 项 | 值 | 备注 |
|---|---|---|
| 机体质量 | 1.5 kg + 4 × 0.005 kg 旋翼 = **1.52 kg** | |
| 机体惯量 | diag(0.0347563, 0.07, 0.0977) | 含旋翼质点后 → diag(0.035651, 0.070349, 0.098922) |
| 推力系数 kf | **8.54858e-6 N/(rad/s)²** | SDF 里的 `motorConstant` |
| 反扭矩系数 kM/kF | **0.016** | SDF 里的 `momentConstant`，是**比值** |
| 电机最大转速 | 800 rad/s | |
| 电机时间常数 | τ_up = 12.5 ms / τ_down = 25 ms | 一阶，上升下降不对称 |
| 悬停转速 | **660.4 rad/s** | 占上限 83%，机动余量不大 |
| 旋翼布局 | x = ±0.13，y = ±0.22 / ±0.20，z = 0.023 | **长方形**，不是正方形 |

旋翼编号与旋向（`spin = +1` 表示从上往下看逆时针）：

| 编号 | x | y | 旋向 |
|---|---|---|---|
| rotor_0 | +0.13 | −0.22 | CCW |
| rotor_1 | −0.13 | +0.20 | CCW |
| rotor_2 | +0.13 | +0.22 | CW |
| rotor_3 | −0.13 | −0.20 | CW |

> 因为 y 是 ±0.22 / ±0.20 而不是对称的，等推力时净力矩为零，但**非对称指令会耦合出额外的滚转**。这不是 bug，是模型如此；做 MPC 时如果忽略它，预测会跟实际对不上。

---

## Gazebo 接口明细

`src/worlds/x3_quadcopter.sdf` 在官方 world 基础上做了两处改动：加了 `OdometryPublisher`（官方 world 只广播 ~60 Hz 的位姿，没有速度，MPC 没法用），并去掉了传感器（需要 IMU/相机时把 `gz-sim-sensors-system` 加回来）。

| 方向 | 话题 / 服务 | 类型 | 说明 |
|---|---|---|---|
| 出 | `/X3/gazebo/command/motor_speed` | `gz.msgs.Actuators` | 填 `velocity`，4 个元素，单位 rad/s |
| 入 | `/model/x3/odometry` | `gz.msgs.Odometry` | 100 Hz，`pose` + `twist` |
| 服务 | `/world/quadcopter/set_pose` | `gz.msgs.Pose` → `Boolean` | teleport，用于重置到指定高度 |

`twist` 在**机体系**，转到世界系需要 `R(q) @ twist.linear`（已用「数值微分位姿」对照核实过）；`twist.angular` 本身就在机体系，直接用。

---

## 双后端一致性验证

```bash
python3 tools/compare_backends.py --duration 1.2 --altitude 1.0
```

给两个后端发同一串开环指令（悬停 → 0.4 s 正弦俯仰+偏航激励 → 0.15 s 掉油门），比 1.2 s 后的轨迹：

```
           analytic         gz        |diff|
x [m]     -0.4146     -0.4015     0.0131
y [m]      0.0547      0.0513     0.0034
z [m]      0.9461      0.9662     0.0202
roll [deg]    -3.090      -2.635      0.455
pitch[deg]   -23.646     -23.311      0.334
yaw  [deg]    -0.234      -0.251      0.017

max |dp|   = 0.0202 m  (4.9% of the trajectory excursion)
max |drpy| = 0.455 deg
VERDICT: models agree
```

这一张表同时验证了四件最容易写错的事：**混控矩阵符号、惯量、推力系数、电机惯性**。

> **改任何参数后先跑这个脚本**，再去调控制器。否则你会在一个错的模型上把增益调到「刚好抵消错误」。

脚本内部还包含一段 `trim()`：仿真器是从地面起步的，而在「teleport → 第一条指令」的十几毫秒里飞机会掉一点高度；不先把两个后端配平到同一个悬停状态，比出来的差异全是测量噪声。

---

## 已知的坑

这几条都是实际踩过、并且已经写进代码注释的：

1. **`momentConstant` 是 kM/kF 比值，不是 kM 本身。** gz 源码里是
   `torque_z = -turningDirection * thrust * momentConstant`，其中 `thrust = kf·ω²`。
   当成绝对力矩系数会让偏航力矩差几个数量级。
2. **teleport 之后的第一个 odometry 样本速度是假的。** `OdometryPublisher` 对位姿做微分，
   跳变 2 m 会得到 **190 m/s** 的瞬时速度。`GzPlant.reset()` 会丢弃该样本。
3. **teleport 之前要先把电机转起来。** 否则在「teleport → 位姿同步」的十几毫秒里飞机会自由落体，
   每局开局白送 0.2 m/s 的下落速度（看起来像控制器有稳态误差）。
4. **`/world/<w>/set_pose/blocking` 在 Python 绑定里永远返回 `False`**，服务实际没生效。
   只能用非 blocking 版本，再自己轮询状态。
5. **单位陷阱**：这里 kf 是 `N/(rad/s)²`。`gym-pybullet-drones` 那类用 `N/rpm²`，
   同一个物理量数值差约 **300 倍**，抄参数必炸。
6. **`analytic` 后端没有地面。** 高度降到 0 以下它不会停。需要地面接触时用 `gz` 后端。

---

## 路线图

- [x] 三层骨架 + 双后端 + 一致性验证
- [ ] 级联 PID（位置 → 速度 → 姿态 → 角速率 → 混控）
- [ ] 线性化模型导出：悬停点数值线性化，输出 `A`、`B`，并用 gz 有限差分校验
- [ ] LQR：`scipy.linalg.solve_continuous_are`，离散化用 ZOH（`expm`）而不是前向欧拉
- [ ] 差分平坦 + 外环 MPC + 内环 PID（约束走 QP：`A_alloc⁻¹·u ∈ [0, ω_max²]`）
- [ ] 把控制器包成 ROS 2 节点，对外只暴露 `/cmd_pose`，接规划层

---

## 许可与致谢

本项目源码采用 **GPL-3.0-or-later**，全文见 [LICENSE](./LICENSE)；每个源文件头部带
`SPDX-License-Identifier: GPL-3.0-or-later`。

选它的两个背景，动手改之前值得看一眼：

- **为什么兼容**：gz-sim 是 Apache-2.0，与 GPL-3.0 **单向兼容**；而且本项目通过
  `gz-transport` 跨进程与仿真通信，不会被算作「链接进同一个二进制」。
- **代价**：强 copyleft——接入闭源 SDK 或要做商业产品时，整个衍生作品都得开源；
  GPLv3 §6 的反 Tivoization 条款还要求，向用户**交付硬件**时提供「安装信息」。
  如果这两点对你是障碍，换成 `LGPL-3.0-or-later` 或 `MPL-2.0` 即可：
  版权在你手上随时能改，但一旦收了外部贡献就难了。

第三方内容：

| 内容 | 许可 | 说明 |
|---|---|---|
| `src/worlds/x3_quadcopter.sdf` | GPL-3.0-or-later，派生自 gz-sim 示例（Apache-2.0） | 文件头保留了 Apache 归属与逐条修改说明 |
| X3 机体模型 | **CC-BY 4.0**（OpenRobotics） | **不随仓库分发**，首次运行从 Fuel 下载；二次分发需署名 |
| gz-sim / Gazebo Harmonic | Apache-2.0 | 运行时依赖 |

参数组织与饱和处理思路参考了：

- Lee, Leok, McClamroch, *Geometric Tracking Control of a Quadrotor UAV on SE(3)*, CDC 2010
- Mellinger & Kumar, *Minimum Snap Trajectory Generation and Control for Quadrotors*, ICRA 2011（差分平坦）
- `gym-pybullet-drones`（UTIAS DSL）、`PyFlyt`
