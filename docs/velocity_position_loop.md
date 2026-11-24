# 横向速度 / 位置环（外环）

> 代码：[`velocity.py`](../src/quadfc/control/velocity.py)、
> [`position.py`](../src/quadfc/control/position.py)
> 已实现的内环：[`altitude.py`](../src/quadfc/control/altitude.py)、
> [`attitude.py`](../src/quadfc/control/attitude.py)、
> [`rate.py`](../src/quadfc/control/rate.py)、
> [`mixer.py`](../src/quadfc/control/mixer.py)
> 被控对象：[`../src/quadfc/plant/analytic.py`](../src/quadfc/plant/analytic.py)
> 参数：[`../src/config/x3.yaml`](../src/config/x3.yaml)
> 测试工具：[`../src/tools/test/step_sine_vel_pos.py`](../src/tools/test/step_sine_vel_pos.py)
> 内环结构与实测见 [`altitude_loop.md`](./altitude_loop.md)。

本文是**规格 + 理论 + 测试口径**。§6 的实测数字取自本仓库的实现（纯 P、`kv=2`、`kp=0.5`），
`--loop cascade` 与 `--loop pos` 逐位一致，所以它同时是控制律和接线的记录。改任何一层之后，
按 §7 重跑；数字落在区间外就是回归。

---

## 目录

- [1. 分层与范围](#1-分层与范围)
- [2. 接口约定（要写的代码）](#2-接口约定要写的代码)
- [3. 理论模型与增益整定](#3-理论模型与增益整定)
- [4. 接入 `CascadeController` 与 CLI](#4-接入-cascadecontroller-与-cli)
- [5. 测试工具](#5-测试工具)
- [6. 实测记录（analytic、100 Hz、2026-11）](#6-实测记录analytic100-hz2026-11)
- [7. 回归指纹](#7-回归指纹)
- [8. 已知问题与 TODO](#8-已知问题与-todo)
- [9. 复现命令](#9-复现命令)

---

## 1. 分层与范围

高度环是**标量**的（`z`），速度和位置环是**水平二维**的（`x`、`y`），这个不对称是刻意的：

- `z` 已经被 `AltitudeController` 闭环（§3.1），再在位置环里写一遍 `z` 会出现**两个积分器争夺同一个通道**，
  表现为爬升慢慢漂或两个环互相对抗；
- 水平方向目前 `a_des[0:2]` 恒为 0（§9.5），所以这两个分量正好空着，不会和任何东西抢。

| 层 | 接口 | 输出 | 状态 |
|---|---|---|---|
| 位置 | `PositionController.step(p, p_des, dt) -> v_des` | `(2,)` m/s | 无（纯 P） |
| 速度 | `VelocityController.step(v, v_des, dt) -> a_xy` | `(2,)` m/s² | 无（纯 P） |
| 高度 | `AltitudeController.step(z, vz, dt) -> acc_z` | 标量 | `int` |
| 姿态 | `AttitudeController.step(R, f_des, yaw_des) -> w_des` | `(3,)` rad/s | 无 |
| 角速度 | `RateController.step(omega, w_des, dt) -> tau` | `(3,)` N·m | `int` |
| 混控 | `MixerController.step(f, tau) -> omega_cmd` | `(4,)` rad/s | 无（`saturated` 是输出） |
| 串联 | `CascadeController.step(state, dt)` | `(4,)` rad/s | 转发 |

### 1.1 带宽预算

外环带宽受**内环**限制，经验上限是内环的 1/3～1/5。内环（姿态层）实测 −3 dB = **2.292 Hz**（§6.7）：

| 环 | 预算上限 | 选定增益 | 实测 −3 dB | 占比 |
|---|---|---|---|---|
| 姿态（内环） | — | `kp = 8` | 2.292 Hz | 1/1 |
| 速度 | ≤ 0.46～0.76 Hz | `kv = 2` | **0.435 Hz** | 1/5.3 |
| 位置 | ≤ 0.09～0.15 Hz | `kp = 0.5` | **0.106 Hz** | 1/21.6 |

速度环取内环的 1/5 是保守但有理由的：这个环的相位裕度只有约 74°，再快一点阶跃响应就会明显超调
（`kv=4` 时超调 4.6 %，见 §3.4）。

---

## 2. 接口约定（要写的代码）

### 2.1 `VelocityController`

```python
class VelocityController:
    name = "velocity"

    def __init__(self, kp: float = 2.0):
        self.kp = np.array([kp, kp])

    def reset(self) -> None:      # 纯 P 无状态，保留是为了和其他层对称
        pass

    def step(self, v, v_des, dt) -> np.ndarray:   # (2,) m/s²，机体系无关，世界系
        return self.kp * (np.asarray(v_des, float) - np.asarray(v, float))
```

### 2.2 `PositionController`

```python
class PositionController:
    name = "position"

    def __init__(self, kp: float = 0.5):
        self.kp = np.array([kp, kp])

    def reset(self) -> None:
        pass

    def step(self, p, p_des, dt) -> np.ndarray:   # (2,) m/s
        return self.kp * (np.asarray(p_des, float) - np.asarray(p, float))
```

四条约定，和 `base.py` 的 `Controller` 契约一致：

1. **形状是 `(2,)`，只有 x/y。** 传 `(3,)` 会诱使别人把 `z` 也接进来（§1）。`z` 由 `AltitudeController` 管。
2. **不做限幅。** 限幅属于串联层（§4.3），理由同 `base.py`：「饱和要能被观察到，不能被 clip 掩盖」。
3. **不要加 `ki`。** `analytic` 后端没有阻力，纯 P 已经是零稳态误差（§6.3 实测 `steady_err < 1e-3 m`）。
   真要加，必须先有抗饱和（§7-5）；在 `a_des` 上积分而不管 `mix.saturated`，只会把阶跃变成过冲。
4. **单位与世界系**：`p`/`v` 是 `State.position` / `State.velocity`，ENU、米、米每秒。不要在这里做机体系变换
   ——`AttitudeController` 已经从 `f_des` 里解出该有的姿态。

---

## 3. 理论模型与增益整定

### 3.1 内环是**三阶**的，这一点决定了外环的形状

内环（姿态 + 角速度 + 混控 + 电机滞后）在 §6.6 已经导出为

```
G_i(s) = a0 / (s³ + a2·s² + a1·s + a0)
a0 = K·ωn²,  a1 = ωn²,  a2 = 2ζωn
ωn = √(kp_rate / (I·τ_up)) = 40.04 rad/s,  ζ = 0.5·√(I / (kp_rate·τ_up)) = 0.999,  K = 8
```

**内环 DC 增益是 1**（`a_des = 0.5 m/s²` 稳定后实测 `a_act = 0.5000 m/s²`、`pitch = 2.918° = atan(a/g)`），
所以外环只需要关心它的**相位**。

### 3.2 速度环：精确传递函数

`a = G_i · kv · (v_des − v)`，于是

```
v/v_des = kv·a0 / (s⁴ + a2·s³ + a1·s² + a0·s + kv·a0)
```

这不是一阶系统。若把内环近似成一阶滞后 `G_i ≈ 1/(1 + τ_i·s)`（`τ_i = 0.1253 s`，由 0.1 Hz 处相位反推），
简化成二阶：

```
ωn = √(kv / τ_i),      ζ = 1 / (2·√(kv·τ_i))
```

| `kv` | `ωn` [rad/s] | `ζ` | 三阶理论 rise | 三阶理论 settle | 一阶近似 `2.197/kv` |
|---|---|---|---|---|---|
| 1 | 2.83 | 1.41 | 1.902 | 3.517 | 2.197（高估 16 %） |
| **2** | **4.00** | **1.00** | **0.794** | **1.511** | 1.099（高估 38 %） |
| 3 | 4.89 | 0.82 | 0.427 | 0.750 | 0.732（高估 71 %） |
| 4 | 5.65 | 0.71 | 0.291 | 0.849 | 0.549（高估 89 %） |

> **这是本文最容易被误判的一条**：用 `rise ≈ 2.197/kv`（把内环当理想）去核对，会发现实测比理论快 30～40 %，
> 看起来像「增益写错了」或「混控多给了力」。其实这是内环相位把 `kv` 转到负虚轴、部分抵消了 `jw`，
> 让闭环极点比 `kv` 更靠左。**用 §3.2 的精确式核对，误差 < 1 %。**
>
> 反向的坑：`kv` 从 3 加到 4，rise 只从 0.427 降到 0.291（−32 %），settle 反而从 0.750 涨到 0.849（+13 %），
> 因为 `ζ` 掉到 0.71、超调 4.6 %。**`kv = 3` 是实用上限，`kv = 2` 是推荐默认值。**

### 3.3 位置环：精确传递函数与设计对位

`v_des = kp·(p_des − p)` 串在速度环外：

```
p/p_des = kv·kp·a0 / (s⁵ + a2·s⁴ + a1·s³ + a0·s² + kv·a0·s + kv·kp·a0)
```

同一阶近似下退化成「`τ_i·s³ + s² + kv·s + kv·kp`」，与高度环（§4.1）**同构**，因此 §4.2 的 Routh 条件
可以直接搬过来：`kv > τ_i·kv·kp`，即 **`kp < 1/τ_i ≈ 8`**。这不是紧约束，真正的约束是带宽预算（§1.1）。

| `kp`（`kv=2`） | 设计 `ωn` | 设计 `ζ` | 三阶理论 rise | 三阶理论 settle |
|---|---|---|---|---|
| 0.20 | 0.63 | 1.58 | 9.847 | 17.991 |
| 0.25 | 0.71 | 1.41 | 7.647 | 14.043 |
| **0.50** | **1.00** | **1.00** | **3.244** | **5.957** |
| 1.00 | 1.41 | 0.71 | 1.299 | 3.749 |

`kp = 0.5` 恰好临界阻尼、带宽 0.106 Hz，落在 §1.1 的预算内，取作默认值。
`kp = 1.0` 的 `ωn = 1.41 rad/s = 0.22 Hz` 已经**超过** §1.1 给位置环的上限（0.09～0.15 Hz），
阶跃虽然更快，但内环相位会开始主导，超调不可控——不要因为「看起来更快」就选它。

### 3.4 横向加速度权限

倾斜会损失垂直推力，所以水平加速度上限取决于「允许多大倾角」而不是「电机多强」：

| 约束 | 值 |
|---|---|
| 全推力下的物理上限（保持高度） | `a_xy ≤ 10.54 m/s²`，倾角 47° |
| 推荐限幅（`--tilt-max 30`） | `a_xy ≤ g·tan(30°) = 5.66 m/s²` |
| 1 m 位置阶跃的峰值倾角 | 5.3° |
| 3 m 位置阶跃的峰值倾角 | 15.6° |
| 触发 30° 限幅所需阶跃 | 约 8 m |

`a_max = g·tan(tilt_max)` 这一条要写在**算 `f_des` 之前**：混控只会老实报 `saturated`，它没有能力拒绝指令（§9.5-1）。

> **限幅限的是指令，不是实测倾角。** 6 m 阶跃、`--tilt-max 30` 时实测峰值倾角是 **34.1°**，
> 因为姿态环对大阶跃还有约 4° 的动态超调；`--tilt-max 8` 时峰值 8.17°。
> 这不是限幅失效，而是「期望加速度」和「实际姿态」之间的差——要压住实际倾角，得把余量留在限幅里
> （比如机动里用 25°，而不是压着 47° 的物理极限用 40°）。

---

## 4. 接入 `CascadeController` 与 CLI

### 4.1 串联层要加的东西

| 成员 | 形状 | 作用 |
|---|---|---|
| `self.pos` | `PositionController` | 新层 |
| `self.vel` | `VelocityController` | 新层 |
| `self.p_des` | `(2,)` 或 `None` | 水平目标；`None` 作哨兵，`reset()` 时取当前位姿 |
| `self.a_des` | `(3,)` | 上一次的期望加速度，**仅供记录/绘图** |
| `self.a_max` | 标量 | `g·tan(tilt_max)` |

`p_des` 必须用 `None` 哨兵，和 `z_des`（§7-2）、`yaw_des`（§7-12）是同一类 bug：
无条件归零会让 `gz` 后端 teleport 到 (3, 4) 之后**自己飞回原点**。

`step()` 里的顺序：

```
acc_z = alt.step(z, vz, dt)
v_des = pos.step(p[:2], p_des, dt)
a_xy  = clip(vel.step(v[:2], v_des, dt), -a_max, a_max)   # 限幅在算 f_des 之前
a_des = [a_xy[0], a_xy[1], acc_z]
f_des = m · (a_des + [0, 0, g])
```

**不要把 `p_des` 存成 `(3,)` 再把 `z` 也写进去**（§1）。`a_des` 之所以留成属性，是因为测试工具的
`--csv`/`--plot` 要读它；这跟 `self.yaw_des` 被保留的理由一样。

### 4.2 CLI

| 参数 | 默认 | 作用 |
|---|---|---|
| `--target-x` / `--target-y` | `None` | 水平目标 [m]，写到 `cascade.p_des`；`None` = 原地悬停 |
| `--target-z` | `None` | 高度目标 [m]，写到 `cascade.alt.z_des`；`None` = 当前高度 |
| `--tilt-max` | 30 | 横向加速度限幅对应的最大倾角 [deg] |

`--target-*` 全为 `None` 时必须**与原来的行为逐位一致**（原地悬停、`sat` 不变）——已实测：
`--altitude 2 --duration 5` 仍是 `[0.000 0.000 2.000]`、`sat 0.0%`。不传新参数就退化成老版本，这是硬要求。

两条容易漏的实现细节：

1. **只给一个轴怎么办。** 两个后端的 `reset()` 都从 x = y = 0 起步（`gz` 会 teleport 到 `(0, 0, altitude)`），
   所以省略的分量按 0 处理就等于「留在原地」。命令行会把解析后的目标打印出来（`target  = x/y=[2.000 1.000] m`），
   免得这个隐含假设变成谜。
2. **`hover` 控制器没有 `p_des` / `alt`。** 给了 `--target-*` 就报错退出，而不是静默忽略：

   ```
   --target-x/--target-y need a position-tracking controller (current: 'hover(x1)'); try --controller cascade
   ```

### 4.3 runner 的报告口径

`run()` 增加关键字参数 `z_target`：给了目标，`Report.altitude_error` 是**终点 − 目标**；
没给，仍是**终点 − 起点**（这样老命令逐位不变）。`--csv` 的列不变，那个文件里没有误差列。

---

## 5. 测试工具

[`../src/tools/test/step_sine_vel_pos.py`](../src/tools/test/step_sine_vel_pos.py)

三种接法，**每种只串它声明要测的层**：

| `--loop` | 被测对象 | 隔离手法 |
|---|---|---|
| `vel` | `VelocityController` | 位置环旁路，`v_des` 直接注入 |
| `pos` | `PositionController` + 速度环 | 手动串两层，高度环默认静音 |
| `cascade` | 整个 `CascadeController` | 用真实串联层，验证接线（需要 §4.1 的四个成员） |

`cascade` 是唯一的**集成**测试：`vel`/`pos` 测的是控制律对不对，`cascade` 测的是线接得对不对——
两者都过才算完成。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--loop` | `pos` | 逗号分隔：`vel`、`pos`、`cascade` |
| `--mode step\|sine\|both` | both | 阶跃 / 扫频 / 都做 |
| `--axis x\|y` | x | 注入哪个水平轴 |
| `--step-v` / `--step-p` | 0.5 / 1.0 | 速度阶跃 [m/s] / 位置阶跃 [m] |
| `--step-at` / `--hold` | 1.0 / 8.0 | 阶跃时刻 [s] / 之后仿真时长 [s] |
| `--rate` | 100 | 控制频率 [Hz] |
| `--altitude` | 2.0 | 起始高度 [m] |
| `--amp-v` / `--amp-p` | 0.2 / 0.2 | 正弦幅值 |
| `--freqs` | 0.05…0.50 | 7 个频率点，跨越两个环的带宽 |
| `--kv` / `--kp` | 取控制器默认 | 临时覆盖，不改源码 |
| `--tilt-max` | 30 | 与 §4.1 的 `a_max` 一致 |
| `--with-altitude` | 关 | 打开高度环（联调用） |
| `--csv` / `--plot` | — | 落盘 / 出图 |
| `--no-check` | 关 | 只打印，不判定 |

工具内部把 §3.1 的三阶内环当作理论，用留数法直接算阶跃响应（不用一阶近似），所以
「theory」列就是 plant 应该给出的值。判定规则：

| 指标 | 容差 |
|---|---|
| rise 10–90 | ±15 % |
| settle 2% | ±20 % |
| 超调 | ≤ max(8 %, 3×理论) |
| 稳态误差 | ≤ 2 % 阶跃幅值 |
| 倾角符号 | `+x → +pitch`、`+y → −roll` |
| −3 dB 带宽 | ±12 % |
| 扫频中段增益 | ±8 % |

任何一项不过 → 退出码 1，可以直接当回归门用。**限幅生效时线性理论不成立，工具会自动跳过判定并打印原因**。

---

## 6. 实测记录（analytic、100 Hz、2026-11）

本节全部是 `--loop vel` / `--loop pos`（工具自己接线）。`--loop cascade`（用真实的
`CascadeController`）在同一批命令上给出**逐位相同**的结果——1 m 阶跃 3.240 s / 5.960 s / `sat` 0.38 %，
−3 dB 0.106 Hz，7/7 通过。两者不一致就说明接线错了（§7）。

### 6.1 速度环阶跃：`kv = 2`，0 → 0.5 m/s

| 指标 | 实测 | 三阶理论 | 一阶近似 |
|---|---|---|---|
| rise 10–90 | **0.790 s** | 0.794 s | 1.099 s |
| 超调 | 0.00 % | 0.00 % | 0 % |
| settle 2% | **1.510 s** | 1.511 s | 1.956 s |
| 稳态误差 | −0.0000 m/s | 0 | 0 |
| 峰值倾角 | 5.31° | — | — |
| `sat` | 0.38 % | — | — |

**幅值线性**（同一个 `kv`，只改阶跃幅值）：

| 阶跃 [m/s] | 0.05 | 0.20 | 0.50 | 1.00 |
|---|---|---|---|---|
| rise | 0.790 | 0.790 | 0.790 | 0.800 |
| `sat` | 0 % | 0 % | 0.5 % | 1.3 % |

rise 与幅值无关 → 这个环在 0.05～1.0 m/s 内是线性的，没有死区、没有饱和拐点。

### 6.2 速度环扫频（A = 0.2 m/s）

| f [Hz] | gain | 理论 | lag | 理论 | sat |
|---|---|---|---|---|---|
| 0.05 | 0.994 | 0.994 | 9.0° | 9.0° | 0 % |
| 0.08 | 0.985 | 0.985 | 14.3° | 14.3° | 0 % |
| 0.10 | 0.976 | 0.976 | 17.8° | 17.8° | 0 % |
| 0.15 | 0.948 | 0.948 | 26.4° | 26.4° | 0 % |
| 0.20 | 0.913 | 0.913 | 34.6° | 34.6° | 0 % |
| 0.30 | 0.829 | 0.829 | 49.8° | 49.8° | 0 % |
| 0.50 | 0.656 | 0.656 | 75.2° | 75.3° | 0 % |

−3 dB：实测 **0.435 Hz**，理论 0.435 Hz。

### 6.3 位置环阶跃

| 工况 | rise | 超调 | settle | 峰值倾角 | `sat` |
|---|---|---|---|---|---|
| 1 m，x，高度环关 | 3.240 s | −0.42 % | 5.960 s | 5.30° | 0.38 % |
| 1 m，x，高度环开 | 3.240 s | −0.42 % | 5.960 s | 5.38° | 0.38 % |
| 1 m，y，高度环开 | 3.250 s | −0.42 % | 5.950 s | −5.37°（roll） | 0.00 % |
| 3 m，x，高度环关 | 3.240 s | −0.02 % | 6.010 s | 15.64° | 2.08 % |
| 3 m，x，高度环开 | 3.240 s | −0.02 % | 6.010 s | 15.81° | 1.83 % |

理论（1 m）：rise 3.244 s / settle 5.957 s。**位置环的理论吻合度是 0.1 %**——因为它足够慢，
内环的一阶近似误差被积分掉了。

两条值得注意的对称性：

1. **x 与 y 完全对称**（3.240 vs 3.250 s）。`y` 轴走的是 roll，符号为负（§6.7 的约定），幅值一致。
2. **高度环开不开几乎无差别**（5.960 vs 5.960，`sat` 都是 0.38 %）。这与姿态环的结论（§6.7
   「联调时饱和会出现」）**相反**：1 m 阶跃只倾斜 5.3°，`cos θ` 只损失 0.4 %，高度环还没反应过来，
   机动已经结束了。到 3 m（15.8°）才开始看到差别，但 `sat` 也只从 2.08 % 变到 1.83 %。

### 6.4 位置环扫频（A = 0.2 m）

| f [Hz] | gain | 理论 | lag | 理论 | sat |
|---|---|---|---|---|---|
| 0.05 | 0.912 | 0.912 | 34.7° | 34.7° | 0 % |
| 0.08 | 0.806 | 0.806 | 52.9° | 52.9° | 0 % |
| 0.10 | 0.731 | 0.731 | 63.7° | 63.7° | 0 % |
| 0.15 | 0.560 | 0.560 | 86.3° | 86.3° | 0 % |
| 0.20 | 0.428 | 0.428 | 104.0° | 104.0° | 0 % |
| 0.30 | 0.260 | 0.261 | 130.1° | 130.1° | 0 % |
| 0.50 | 0.115 | 0.116 | 163.5° | 163.6° | 0 % |

−3 dB：实测 **0.106 Hz**，理论 0.106 Hz。

### 6.5 `kv` 扫描（速度环，0 → 0.5 m/s）

| `kv` | rise 实测 | rise 理论 | settle 实测 | settle 理论 | 超调 |
|---|---|---|---|---|---|
| 1 | 1.910 | 1.902 | 3.510 | 3.517 | −0.01 % |
| 2 | 0.790 | 0.794 | 1.510 | 1.511 | 0.00 % |
| 3 | 0.430 | 0.427 | 0.780 | 0.750 | 0.10 % |
| 4 | 0.290 | 0.291 | 0.870 | 0.849 | 4.61 % |

### 6.6 Gazebo 交叉验证

同一套增益（`kv=2` / `kp=0.5` / `tilt-max 30`）直接搬到 gz 后端（Harmonic 8.15，真实电机惯性 +
桨叶阻力 + DART，`-s` 无 GUI），不改一行代码：

| 工况 | analytic | gz | 差 |
|---|---|---|---|
| 悬停 5 s，`--altitude 2` | `[0.000 0.000 2.000]`，`sat` 0 % | `[0.000 0.000 2.004]`，`sat` 0 % | z 4 mm |
| 航点 (1, 0.5)，25 s | `[1.000 0.500 2.000]`，峰值倾角 **5.99°**，`sat` 0 % | `[1.000 0.500 2.004]`，峰值倾角 **5.97°**，`sat` 0 % | 倾角 0.02° |

两件事值得记：

1. **峰值倾角差 0.02°**，说明 §3 的三阶理论（用 `τ_up` 建的内环模型）在 gz 上同样成立——
   外环的两个增益不需要为 gz 重新整定。
2. **那 4 mm 不是控制误差。** `GzPlant.reset()` 的收敛判据是 `|z − altitude| < 0.05`，所以起点本身就
   允许几毫米偏差，而 `altitude_error` 是相对目标算的。想消掉就把那个 5 cm 阈值收紧（代价是 reset 慢一点）。

> 工具的 `--loop vel|pos|cascade` 只接 `AnalyticPlant`，所以 gz 只能用 CLI 的端到端数字核对（§9）。
> 在 gz 上重跑工具需要给 `simulate()` 加一个 plant 工厂，属 TODO。

---

## 7. 回归指纹

改任何一层之后，下面这些数字必须落在括号内的区间：

```bash
cd src

# 速度环：0 -> 0.5 m/s
python3 tools/test/step_sine_vel_pos.py --mode step --loop vel
#   rise 0.790 s (±15%)   settle 1.510 s (±20%)   超调 < 8%

# 位置环：0 -> 1 m
python3 tools/test/step_sine_vel_pos.py --mode step --loop pos
#   rise 3.240 s (±15%)   settle 5.960 s (±20%)   超调 < 8%

# 两个环的频响
python3 tools/test/step_sine_vel_pos.py --mode sine --loop vel,pos
#   vel -3 dB 0.435 Hz (±12%)   pos -3 dB 0.106 Hz (±12%)

# 集成：真实 CascadeController（需要 §4.1 的四个成员）
python3 tools/test/step_sine_vel_pos.py --mode step --loop cascade
```

`cascade` 与 `pos` 的数字应当**逐位一致**（同一套增益、同一套接线）。
不一致就说明 `CascadeController` 的接线和控制律本身不一致——这正是这个模式存在的理由。

端到端（`cli.py` + `runner.py` 的接线，不在工具覆盖范围内，只能按数字核对）：

```bash
python3 -m quadfc --backend analytic --controller cascade --altitude 2 \
    --target-x 2 --target-y 1 --duration 20 --no-realtime --quiet
#   final position = [2.000 1.000 2.000]   altitude error = +0.000 m
#   max tilt = 11.71 deg                   saturated = 0.0%

python3 -m quadfc --backend analytic --controller cascade --altitude 2 \
    --target-z 3 --duration 15 --no-realtime --quiet
#   final position = [0.000 0.000 3.000]   altitude error = ±0.000 m

python3 -m quadfc --backend analytic --controller cascade --altitude 2 \
    --duration 5 --no-realtime --quiet
#   不传目标 = 老行为： final position = [0.000 0.000 2.000]   saturated = 0.0%

python3 -m quadfc --backend analytic --controller cascade --altitude 2 \
    --target-x 6 --tilt-max 8 --duration 25 --no-realtime --quiet
#   限幅生效： max tilt = 8.17 deg（默认 30° 时同一个目标到 34.12 deg，见 §3.4）
```

---

## 8. 已知问题与 TODO

| # | 问题 | 影响 | 建议 |
|---|---|---|---|
| 1 | 两个环都只有 P | `analytic` 后端无阻力，所以没有静差；换到有阻力的模型或真机会有 | 加 `ki` 前先做抗饱和，复用 `mix.saturated` |
| 2 | 没有前馈（`z̈_des`、`v̇_des`） | 正弦跟踪有相位滞后（0.2 Hz 已 35°） | 位置环输出 `v_des` 时顺带算 `v̇_des`，加到 `a_xy` |
| 3 | `p_des` 是**阶跃**输入 | 3 m 阶跃峰值倾角 15.6°，8 m 就会撞 30° 限幅 | 需要航点跟踪时换成带速度规划的参考（S 曲线 / 最小 snap） |
| 4 | ~~只在 `analytic` 后端验证过~~ | 已在 gz 8.15 上交叉验证（§6.6）：航点峰值倾角 5.97° vs 5.99°，增益无需重调 | 工具仍只支持 analytic；给 `simulate()` 加 plant 工厂后才能自动化 |
| 5 | 横向限幅固定 30° | 大机动时推力余量没有被利用 | 需要时按当前 `acc_z` 动态算 `a_max`，而不是常数 |
| 6 | 位置环 `kp` 上限（`1/τ_i ≈ 8`）只是 Routh 必要条件 | 过了这个值会先出现振荡，不会立刻发散 | 不要把 Routh 条件当设计上限，用 §1.1 的带宽预算 |
| 7 | 没有 yaw 规划 | `yaw_des` 保持起飞时的朝向 | 需要「机头朝速度方向」时，在 `p_des` 变化时更新 `yaw_des` |

---

## 9. 复现命令

> Windows 上这些命令都能跑（`analytic` 后端），把 `python3` 换成 `python` 即可；
> Windows 写法与坑见 README 的「在 Windows 上开发」一节。gz 那几条只能在 Linux / WSL 上跑。

```bash
cd /home/nuaawzh/detect_design/src

# 速度环：阶跃 + 扫频
python3 tools/test/step_sine_vel_pos.py --loop vel --mode both --plot

# 位置环：阶跃 + 扫频
python3 tools/test/step_sine_vel_pos.py --loop pos --mode both --plot

# y 轴（roll 通道）与高度环联调
python3 tools/test/step_sine_vel_pos.py --loop pos --axis y --with-altitude

# 大阶跃（看倾角与饱和）
python3 tools/test/step_sine_vel_pos.py --loop pos --step-p 3 --hold 12

# 限幅分支（8° 时线性理论不成立，工具自动跳过判定）
python3 tools/test/step_sine_vel_pos.py --loop pos --step-p 3 --tilt-max 8

# 换增益（不改源码）
python3 tools/test/step_sine_vel_pos.py --loop vel --kv 3
python3 tools/test/step_sine_vel_pos.py --loop pos --kp 1.0

# 集成测试（CascadeController）
python3 tools/test/step_sine_vel_pos.py --loop cascade --mode step

# 命令行端到端
python3 -m quadfc --backend analytic --controller cascade \
    --altitude 2 --target-x 2 --target-y 1 --duration 15 --no-realtime

# Gazebo（真实物理，实时跑；首次启动约 20 s）
python3 -m quadfc --backend gz --controller cascade --altitude 2 --duration 5 --quiet
python3 -m quadfc --backend gz --controller cascade --altitude 2 \
    --target-x 1 --target-y 0.5 --duration 25 --quiet
python3 -m quadfc --backend gz --controller cascade --altitude 2 --duration 8 --gui
```
