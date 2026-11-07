# 高度环（`RateAltController`）结构与测试

> 代码：[`../src/quadfc/control/rate_alt.py`](../src/quadfc/control/rate_alt.py)
> 被控对象：[`../src/quadfc/plant/analytic.py`](../src/quadfc/plant/analytic.py)
> 参数：[`../src/config/x3.yaml`](../src/config/x3.yaml)
> 测试工具：[`../src/tools/test/step_sine_rate_alt.py`](../src/tools/test/step_sine_rate_alt.py)

---

## 1. 定位与范围

`RateAltController` 是一个 **单环高度 + 角速度阻尼** 的控制器，输入 `State`，输出 4 个电机转速（rad/s）。

| 包含 | 不包含 |
|---|---|
| 高度 PD（可选 I） | 水平位置 / 速度环 |
| 角速度 P（可选 I） | 姿态角环、期望姿态生成 |
| 混控 `B⁻¹` + 推力饱和缩放 | 轨迹前馈、倾角补偿 |
| — | 电机动力学（属于 plant） |

因此它只能验证「升到指定高度并停住」和「角速度接近零」，**不能验证航点跟踪或姿态保持**（见 §7）。

---

## 2. 数据流

```
                 ┌── 高度环 ─────────────────────────────────────────┐
 z_des ──► e_z ──► kp_z / ki_z / kd_z ──► acc_z [m/s²] ──► F = m(g + acc_z) [N] ─┐
                 └─────────────────────────────────────────────────────────────────┤
                                                                                   │
                 ┌── 角速度环 ────────────────────────────────────────────────────┤
 w_des ──► e_ω ──► kp_rate / ki_rate ──► τ [N·m] ─────────────────────────────────┤
                 └─────────────────────────────────────────────────────────────────┤
                                                                                   ▼
                                                       u = [F, τx, τy, τz] = B · w²
                                                                                   │
                                                                    w² = B⁻¹ · u   │
                                                                                   │
                                                          _scale() → [0, w²_max]  │
                                                                                   │
                                                       ω_cmd = √w²  → (4,) rad/s ◄─┘
```

关键点：控制器输出的是**电机转速**，不是推力/力矩。推力到转速之间是 `w = √(F/(4kf))` 的非线性（X 型等权分配时）。

---

## 3. 逐块说明

### 3.1 高度环（`step()` 前半段）

```python
e_z   = self.z_des - state.position[2]          # [m]，世界系 ENU 的 z
v_z   = state.velocity[2]                       # [m/s]，世界系
self.int_z = clip(self.int_z + e_z*dt, -1, 1)   # 单位 m·s，限幅 ±1
acc_z = kp_z*e_z + ki_z*self.int_z - kd_z*v_z   # [m/s²]
F     = mass * (g + acc_z)                      # [N]，悬停前馈 + 校正
```

| 项 | 含义 | 当前值 |
|---|---|---|
| `F = m(g + acc_z)` | 悬停推力前馈 + 期望加速度 | — |
| `kp_z` | 位置误差 → 加速度 | 1.0 |
| `ki_z` | 位置误差积分（消除常值扰动下的静差） | 0.0 |
| `kd_z` | 垂直速度阻尼 | 0.1（见 §7） |

**假设机体近似水平**：推力方向取机体 z 轴，不做 `cos(tilt)` 补偿，也没有水平分量。倾角一大，实际垂直推力下降、同时产生横向漂移。

**没有参考加速度前馈**：正弦跟踪时相位滞后由 `H(s)` 决定；若要做轨迹跟踪，应加 `z̈_des`。

### 3.2 角速度环

```python
e_w = self.w_des - state.omega                  # [rad/s]，机体系
self.int_rate = clip(self.int_rate + e_w*dt, -1, 1)
tau = kp_rate*e_w + ki_rate*self.int_rate       # [N·m]
```

`w_des` 目前恒为 0，所以这一环只是**零角速度阻尼**，不是姿态保持：姿态角为常值时角速度也为零，这是一个平衡点，不会自动回水平。

### 3.3 混控

```python
w2 = B_inv @ np.array([F, *tau])     # u = [F, τx, τy, τz] = B @ w²
```

`B = params.allocation`，形状 `(4, n_rotors)`，由旋翼位置与转向生成：

```
row0: kf
row1: kf * y_i          (τx = Σ y_i f_i)
row2: -kf * x_i         (τy = -Σ x_i f_i)
row3: -kf * km_ratio * spin_i
```

X3 是 4 旋翼 X 型，`B` 可逆；且 `B⁻¹ @ [F,0,0,0] = (F/(4kf))·1`，即四个电机等速。

### 3.4 饱和缩放 `_scale()`

```python
if w2.max() > w2_max or w2.min() < 0.0:
    w2 = self._scale(w2)
```

语义：**保留（裁剪后的）均值，按比例 λ 收缩各电机的差异**，使结果落在 `[0, w2_max]`。

- 水平机动时会损失一部分力矩权限；
- 如果均值本身已超上限，`λ = 0`，四个电机全部固定在 `w2_max`，此时**总推力受限、力矩权限也为零**；
- 负的 `w2` 分量同样被压到 0，意味着该侧电机停转。

### 3.5 输出与契约

```python
return np.sqrt(np.clip(w2, 0.0, self.w2_max))
```

> ⚠️ [`base.py`](../src/quadfc/control/base.py) 的契约是「控制器不要裁到 `[0, omega_max]`，交给 mixer，这样 runner 才能统计饱和」。本控制器在内部完成了全部限幅，**导致 runner / 测试工具的饱和率恒为 0**。修法见 §7-1。

---

## 4. 参数、整定与边界

### 4.1 小信号模型

在悬停点线性化（`w²` 的开方非线性恰好与 `dF/dw` 抵消），电机一阶滞后 `τ_m` 下：

```
ΔF_act ≈ ΔF_cmd / (1 + τ_m s)
τ_m · z⃛ + z̈ + kd_z · ż + kp_z · z = kp_z · z_des
```

忽略滞后时的标准二阶形式：

```
ωn = √kp_z            [rad/s]
ζ  = kd_z / (2√kp_z)
```

### 4.2 稳定性条件（Routh）

特征多项式 `τ_m s³ + s² + kd_z s + kp_z`：

```
kd_z > τ_m · kp_z          （必要且充分，另需 kp_z > 0）
```

X3 的 `τ_down = 0.025 s` 是较坏情况，所以 `kp_z=1` 时 `kd_z` 至少 0.025 才稳定——但只是「不发散」，阻尼比仍接近 0。

### 4.3 推力 / 加速度权限（X3）

| 量 | 值 |
|---|---|
| 总质量 `m` | 1.52 kg |
| 悬停推力 `mg` | 14.911 N |
| 最大推力 `4·kf·ω_max²` | 21.884 N |
| 悬停转速 | 660.4 rad/s（ω_max 的 82.6%） |
| 向上加速度上限 | **+4.59 m/s²** |
| 向下加速度上限 | −9.81 m/s²（电机到 0） |

推论：单步阶跃 `Δz` 若满足 `kp_z·Δz ≤ 4.59`，就不会在起步阶段饱和。这也是「1 m 阶跃看不出饱和」的原因。

### 4.4 推荐增益

| 目标 | kp_z | kd_z | ωn | ζ | 说明 |
|---|---|---|---|---|---|
| 临界阻尼、已实测 | 2.5 | 3.2 | 1.58 rad/s (0.25 Hz) | 1.01 | 推荐默认值 |
| 更保守 | 1.0 | 2.0 | 1.0 rad/s | 1.0 | 稳态慢约 1.6 倍 |
| 不推荐 | 1.0 | 0.1 | 1.0 rad/s | 0.05 | 仅勉强稳定，Q≈10 |

参数化写法：

```python
wn, zeta = 1.6, 1.0
self.kp_z = wn ** 2            # 2.56
self.kd_z = 2 * zeta * wn      # 3.2
```

---

## 5. 与仿真器的接口

| 项 | 约定 |
|---|---|
| `controller.step(state, dt)` | `dt = 1/rate`（默认 100 Hz） |
| 用到的状态 | `position[2]`、`velocity[2]`、`omega` |
| **未用到** | `quat` / `euler()` → 无姿态反馈 |
| 输出 | `(4,)` rad/s，顺序 `rotor_0..3` |
| 时序 | plant 先给出状态 → 控制器算指令 → plant 积分 `dt`（内部 `substep=1e-3` RK4） |
| 电机滞后 | `τ_up = 0.0125 s`，`τ_down = 0.025 s`（plant 侧） |

`reset(state)` 会清零两个积分器，并把 `z_des` 设为**当前高度**——所以 CLI 跑出来永远是「原地悬停」，无法通过命令行指定目标高度。

---

## 6. 测试

### 6.1 工具

```bash
cd src
python3 tools/test/step_sine_rate_alt.py --mode both --kp 2.5 --kd 3.2 --plot
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode step\|sine\|both` | both | 阶跃 / 扫频 / 都做 |
| `--altitude` | 1.0 | 起始高度 |
| `--step` / `--step-at` / `--hold` | 1.0 / 0.5 / 10 | 阶跃幅值、时刻、之后仿真时长 |
| `--amplitude` / `--freqs` | 0.05 / 0.05…0.30 | 正弦幅值与频率点 |
| `--kp` / `--kd` | 取代码默认 | 临时覆盖，不改源码 |
| `--csv` | — | 导出 `t,z,vz,thrust,saturated` |

工具在每个控制周期写 `ctrl.z_des = ref(t)`，**不改控制器**。

### 6.2 指标定义

| 指标 | 定义 |
|---|---|
| `t_rise_10_90` | 首次到达 10% 与 90% 的时间差 |
| `overshoot` | `(峰值 − 目标) / 阶跃幅值` |
| `t_settle_2%` | 最后一次离开 ±2% 带的时刻；`nan` = 结束前未稳定 |
| `steady_err` | 最后 0.5 s 均值 − 目标 |
| `sat` | 指令越界的采样比例（当前实现失效，见 §7-1） |
| 正弦 `gain` / `lag` | 对稳态段做 `c + a·sin + b·cos` 最小二乘，`gain = √(a²+b²)/A`，`lag = −atan2(b, a)` |

### 6.3 理论基准

`kp=2.5, kd=3.2` → `ωn = 1.581 rad/s`，`ζ = 1.012`：

| 量 | 理论 |
|---|---|
| 10–90% 上升 | `3.358/ωn = 2.124 s` |
| 2% 调节 | `5.834/ωn = 3.689 s` |
| 超调 | 0%（ζ ≥ 1） |
| −3 dB 带宽 | `0.633·ωn = 1.001 rad/s = 0.159 Hz` |
| 增益 `\|H(jω)\|` | `1/√((1−r²)² + (2ζr)²)`，`r = ω/ωn` |

### 6.4 实测记录（2026-11，analytic 后端）

**阶跃 1 → 2 m**

| 指标 | 实测 | 理论 |
|---|---|---|
| rise 10–90 | 2.130 s | 2.124 s |
| overshoot | 0.000% | 0% |
| settle 2% | 3.770 s | 3.689 s |
| steady err | −0.000 m | 0 |
| sat | 0% | — |

**正弦扫频（A = 0.05 m）**

| f [Hz] | 实测增益 | 理论 | 实测滞后 | 理论 |
|---|---|---|---|---|
| 0.05 | 0.96 | 0.960 | 22.7° | 22.7° |
| 0.10 | 0.86 | 0.859 | 43.6° | 43.7° |
| 0.16 | 0.71 | 0.705 | 65.0° | 65.2° |
| 0.20 | 0.61 | 0.606 | 77.0° | 77.1° |
| 0.30 | 0.42 | 0.408 | 100.2° | 99.9° |

结论：与二阶模型偏差 ≤ 3%（0.3 Hz 处由电机滞后 + 100 Hz 零阶保持解释），实现正确，增益可用。

> 注意：0.05 Hz 只有 0.96 不是标度错误——带宽 0.16 Hz 离它并不远。要验 DC 增益，加 `0.01` 到 `--freqs`。

**带宽打印的已知 bug**：`np.interp(-3.0, [db[i-1], db[i]], ...)` 的 `xp` 是递减的，结果直接取端点，打印 0.200 Hz（真值 ≈0.160 Hz）。修正：

```python
bw = float(np.interp(-3.0, [db[i], db[i - 1]], [fs[i], fs[i - 1]]))
```

### 6.5 饱和测试

```bash
python3 tools/test/step_sine_rate_alt.py --mode step --step 3 --hold 12 --kp 2.5 --kd 3.2
```

| 阶跃 | 正向饱和时长（估算） | 现象 |
|---|---|---|
| 1 m | 0 s | 全程线性 |
| 3 m | ≈0.19 s | 指标几乎不变，只有 `t_settle` 3.77→3.82 |
| 6 m | ≈0.58 s | 上升明显变慢 |
| 8 m | ≈0.80 s | 明显的推力受限段 |
| ≥12 m | 正向 + 反向 | 刹车段减速度超过 −9.81 m/s²，**开始超调** |

`saturated = 0%` 时如何确认真的饱和了：看 CSV 的 `thrust` 列是否在 21.884 N 上停留约 19 个采样点（0.19 s @ 100 Hz）。

```python
import numpy as np
a = np.genfromtxt("/tmp/sat.csv", delimiter=",", names=True)
m = np.isclose(a["thrust"], 21.884, atol=1e-3)
print(m.sum(), a["t"][m].min(), a["t"][m].max())
```

### 6.6 角速度环测试（待做）

`w_des` 恒零，内环从未被激励。隔离方法：把高度增益置 0（此时 `F ≡ mg`），每个周期写

```python
ctrl.w_des = np.array([A_w * np.sin(2*np.pi*f*s.t), 0.0, 0.0])
ctrl.kp_z = ctrl.ki_z = ctrl.kd_z = 0.0
```

扫 `f = 1…10 Hz`、`A_w = 0.1 rad/s`，拟合 `state.omega[0]`。预期带宽 ≈ `kp_rate/I`：`0.71 / 0.0356 ≈ 20 rad/s ≈ 3.2 Hz`。注意 100 Hz 控制频率在 10 Hz 已接近离散化误差明显区。

### 6.7 Gazebo 交叉验证

```bash
python3 -m quadfc --backend gz --controller rate_alt --altitude 2 --duration 8 --gui
```

gz 初始状态不如 analytic 干净（teleport + 落地瞬态），前 1–2 s 不参与指标评价。analytic 与 gz 趋势明显不一致时，优先查符号/单位（参考 [`compare_backends.py`](../src/tools/compare_backends.py)）。

### 6.8 回归测试建议

把 §6.3 的解析式作为断言，容差 0.03（实测最大偏差 0.012，标度错误会差数倍）：

```python
def test_bode_matches_second_order(params):
    kp, kd = 2.5, 3.2
    wn, zeta = np.sqrt(kp), kd / (2 * np.sqrt(kp))
    for f in (0.05, 0.10, 0.16, 0.20):
        r = 2 * np.pi * f / wn
        gain_theory = 1 / np.hypot(1 - r * r, 2 * zeta * r)
        assert abs(measured_gain(f) - gain_theory) < 0.03
```

---

## 7. 已知问题与 TODO

| # | 问题 | 影响 | 建议 |
|---|---|---|---|
| 1 | 控制器内部完成 `_scale` + `clip`，破坏了 `base.py` 的约定 | runner / 工具的饱和率**恒为 0**，无法用于调增益 | 在 `step()` 里记录 `self.saturated`（见下） |
| 2 | `reset()` 覆盖 `z_des` 为当前高度 | CLI 只能「原地悬停」，无法指定目标 | `__init__(..., z_des=None)`，`reset` 仅在未指定时取当前高度 |
| 3 | `w_des` 恒为 0 | 零角速度 ≠ 水平；有初始倾角会一直斜着飘 | 增加姿态外环（角度 → `w_des`） |
| 4 | `kp_rate` 写死，`I = np.diag(params.inertia)` 未使用 | 换机型即失效 | `self.kp_rate = np.diag(params.inertia) * wc` |
| 5 | `ki_z` 可开但无抗饱和 | 饱和期间积分累积，恢复后过冲 | 条件积分或 back-calculation，可复用 `self.saturated` |
| 6 | 无 `z̈_des` 前馈 | 正弦/轨迹跟踪存在相位滞后（0.16 Hz 已 65°） | `acc_z += z_des_ddot` |
| 7 | 默认 `kd_z = 0.1` | ζ = 0.05，Q ≈ 10，仅勉强稳定 | 改为 §4.4 推荐值 |
| 8 | 推力沿机体 z 轴，无倾角补偿 | 大倾角时高度环失效、产生横向漂移 | 用 `quat` 把期望推力分解到机体系 |
| 9 | `_scale` 全超限时退化为恒最大推力 | 同时丢失力矩权限 | 记录并在上层限制期望加速度 |

问题 1 的最小修法：

```python
w2 = self.B_inv @ np.array([F, *tau])
self.saturated = bool(w2.max() > self.w2_max or w2.min() < 0.0)
if self.saturated:
    w2 = self._scale(w2)
```

工具里改用 `sat[i] = ctrl.saturated`（必须在 `step()` **之后**读）。

---

## 8. 复现命令

```bash
cd /home/nuaawzh/detect_design/src

# 基线（原地悬停，验证配平）
python3 -m quadfc --backend analytic --controller rate_alt \
    --altitude 2 --duration 5 --no-realtime

# 阶跃 + 扫频，当前推荐增益
python3 tools/test/step_sine_rate_alt.py --mode both --kp 2.5 --kd 3.2 --plot

# 饱和分支
python3 tools/test/step_sine_rate_alt.py --mode step --step 6 --hold 12 \
    --kp 2.5 --kd 3.2 --csv /tmp/sat.csv

# 低频增益校验（验 DC 增益 ≈ 1）
python3 tools/test/step_sine_rate_alt.py --mode sine --freqs 0.01,0.05,0.10

# Gazebo
python3 -m quadfc --backend gz --controller rate_alt --altitude 2 --duration 8 --gui
```