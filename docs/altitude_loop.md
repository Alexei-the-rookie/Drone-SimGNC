# 级联控制器（高度 / 姿态 / 角速度）结构与测试

> 代码：[`../src/quadfc/control/cascade.py`](../src/quadfc/control/cascade.py)（串联）、
> [`altitude.py`](../src/quadfc/control/altitude.py)、
> [`attitude.py`](../src/quadfc/control/attitude.py)、
> [`rate.py`](../src/quadfc/control/rate.py)、
> [`mixer.py`](../src/quadfc/control/mixer.py)
> 被控对象：[`../src/quadfc/plant/analytic.py`](../src/quadfc/plant/analytic.py)
> 参数：[`../src/config/x3.yaml`](../src/config/x3.yaml)
> 测试工具：[`../src/tools/test/step_sine_rate_alt.py`](../src/tools/test/step_sine_rate_alt.py)（高度环）、
> [`../src/tools/test/step_sine_att.py`](../src/tools/test/step_sine_att.py)（姿态环）、
> [`../src/tools/test/step_sine_rate.py`](../src/tools/test/step_sine_rate.py)（角速度环 + 混控）

---

## 1. 分层与范围

控制栈按「一个 `step()` 只做一件事」切成四层（外加串联层），只有最外层实现 `Controller`：

| 层 | 类 | `step()` 签名 | 状态 | 文件 |
|---|---|---|---|---|
| 高度 | `AltitudeController` | `(z, vz, dt) -> acc_z: float` | `int` | `altitude.py` |
| 姿态 | `AttitudeController` | `(R, f_des, yaw_des) -> w_des (3,)` | **无** | `attitude.py` |
| 角速度 | `RateController` | `(omega, w_des, dt) -> τ (3,)` | `int` | `rate.py` |
| 混控 | `MixerController` | `(f, τ) -> ω_cmd (4,)` | 无（`saturated` 是输出） | `mixer.py` |
| 串联 | `CascadeController(Controller)` | `(state, dt) -> ω_cmd (4,)` | 转发 | `cascade.py` |

依赖是单向的 `cascade → {altitude, attitude, rate, mixer}`。四个内层**不 import `base.Controller`**，
也不认识 `State`——它们只收标量/矩阵/向量，因此每个都能脱离 plant 单测。
`attitude` 和 `mixer` 是**纯函数式**的（无状态，连 `reset()` 都没有），`altitude` 和 `rate` 各自只有一个积分器。

> 旧实现是一个 `RateAltController` 单类。它之所以长成屎山，根因是
> `Controller.step(state, dt)` 是**整条栈**的接口：让每层都继承它，就等于要求每层都自成一条完整栈，
> 于是 `w_des` 只能变成「自己从不设置、只能被别人写」的字段。分层后 `w_des` 回到入参位置。

| 包含 | 不包含 |
|---|---|
| 高度 PD（可选 I） | 水平位置 / 速度环 |
| 姿态 P（矢量式，§3.2） | 轨迹前馈 `z̈_des` |
| 角速度 P（可选 I） | 电机动力学（属于 plant） |
| 混控 `B⁻¹` + 推力饱和缩放 | 陀螺耦合补偿 `ω×Iω`（属于 plant） |

倾角补偿**不再是缺口**：`f = ‖m(a_des + G·e_z)‖` 天然含 `1/cos θ`，倾斜时总推力自动加大（§3.2、§7-8）。

因此它能验证「升到指定高度并停住」、「倾斜到指定姿态」（§6.7）和「角速度跟踪 `w_des`」（§6.6），
**不能验证航点跟踪**——水平方向还没有闭环（见 §7-3）。

---

## 2. 数据流

```
 z_des ──►┌────────────────┐
          │ Altitude       │
  z, vz ─►│  Controller    ├──► acc_z ──┐
          └────────────────┘            │
                                        ▼
                              a_des = [0, 0, acc_z]        ← 世界系 [m/s²]，速度层以后写 [0:2]
                                        │
                              f_des = m·(a_des + G·e_z)    ← 世界系矢量 [N]
                                        │
                       ┌────────────────┴────────────────┐
                       │ 方向                             │ 模长
                       ▼                                  ▼
          ┌────────────────────────┐              ┌──────────────┐
 yaw_des ►│ Attitude               │              │ f = ‖f_des‖  │
 R ──────►│  Controller            │              └──────┬───────┘
          └───────────┬────────────┘                     │
                      │ w_des (3,) 机体系                │
  omega ──►┌──────────▼─────┐                            │
           │ Rate           │                            │
           │  Controller    ├──► τ (3,) ──┐              │
           └────────────────┘             ▼              ▼
                                    ┌──────────────────────────┐
                                    │ Mixer                    │
                                    │ w² = B⁻¹·[f, τ]          │
                                    │ _scale() → 饱和          │
                                    │ ω_cmd = √w²              │
                                    └────────────┬─────────────┘
                                                 ▼  (4,) rad/s
```

`CascadeController.step()` 就六行，一行一层：

```python
acc_z = self.alt.step(state.position[2], state.velocity[2], dt)
a_des = np.array([0.0, 0.0, acc_z])
f_des = self.params.mass * (a_des + np.array([0.0, 0.0, G]))
w_des = self.att.step(state.rotation(), f_des, self.yaw_des)
tau   = self.rate.step(state.omega, w_des, dt)
return self.mix.step(float(np.linalg.norm(f_des)), tau)
```

两个关键点：

- **`f_des` 算一次、用两次**：姿态层吃它的**方向**，混控吃它的**模长**。不要在两层里各算一遍 `m(a_des + G·e_z)`，否则换重力模型或加风扰时要改两个地方。
- 控制器输出的是**电机转速**，不是推力/力矩。推力到转速之间是 `w = √(F/(4kf))` 的非线性（X 型等权分配时）。

悬停点自检（`a_des = 0`）：`‖f_des‖ = mg = 14.911200 N`，`w_des = 0`，`τ = 0`，`ω_cmd = 660.3578 rad/s`（= `hover_omega`）。**逐位相同**，这就是串级接对了的判据。

---

## 3. 逐块说明

### 3.1 高度环（`altitude.py`）

```python
error = self.z_des - z                        # [m]，世界系 ENU 的 z
self.int = clip(self.int + error*dt, -1, 1)   # 单位 m·s，限幅 ±1
acc_z = self.kp*error - self.kd*vz + self.ki*self.int   # [m/s²]
```

| 项 | 含义 | 当前值 |
|---|---|---|
| `kp` | 位置误差 → 加速度 | 2.5 |
| `ki` | 位置误差积分（消除常值扰动下的静差） | 0.0 |
| `kd` | 垂直速度阻尼 | 3.2 |

**`z_des` 是这一层唯一的 setpoint，且用 `None` 作哨兵**：

```python
def reset(self, z0):
    self.int = 0.0
    if self.z_des is None:      # 只在未设定时才取当前高度
        self.z_des = z0
```

这样 CLI 不设 `z_des`（保持 `None`）时行为是「原地悬停」，而测试工具每周期写 `z_des` 之后不会被 `reset` 抹掉——§7-2 的静默 bug 由此修好。

**这一层不需要 `params`**：纯 PD，不含任何机体参数。如果哪天它开始需要 `params`，说明它在偷做 plant 的活。

**假设机体近似水平**：`acc_z` 只是一个标量，推力方向仍取机体 z 轴，不做 `cos(tilt)` 补偿，也没有水平分量。倾角一大，实际垂直推力下降、同时产生横向漂移。（B 方案接上姿态层后由 `|F_des|` 自动吸收，见 §9。）

**没有参考加速度前馈**：正弦跟踪时相位滞后由 `H(s)` 决定；若要做轨迹跟踪，应加 `z̈_des`。

### 3.2 姿态环（`attitude.py`）

**这一层不命令姿态，它命令的是「为了产生这个推力需要什么姿态」。** 四旋翼的推力只能沿机体 z 轴，
所以「往 +x 加速」等价于「把机体 z 轴指向 +x 上方」——姿态是推力的奴隶。

```python
R_des = yaw_des_rotation(f_des, yaw_des)      # 推力方向 + 期望航向 -> 期望姿态
return -self.kp * attitude_error(R, R_des)    # 姿态误差 -> 期望角速度（机体系）
```

**构造 `R_des`**：

```
z_b = f_des / ‖f_des‖
x_c = [cos yaw_des, sin yaw_des, 0]           # 期望机头方向在世界系水平面上的投影
y_b = normalize(cross(z_b, x_c))              # 顺序写反，推力会指向反方向
x_b = cross(y_b, z_b)
R_des = column_stack([x_b, y_b, z_b])
```

**姿态误差**（Lee 等人的 SO(3) 误差，机体系）：

```
M   = R_desᵀ R − Rᵀ R_des
e_R = ½ [M[2,1], M[0,2], M[1,0]]              # vee 的索引是 (2,1)/(0,2)/(1,0)
```

`e_R` 是「把 `R_des` 转到 `R` 的旋转矢量」，所以修正要取负：`w_des = −kp·e_R`。

| 细节 | 说明 |
|---|---|
| `M` 用 `.T` 而非 `inv` | 保证 `M` 严格反对称（`inv` 会把 `R` 的数值漂移放大到 vee 的假设之外），且零成本 |
| `e_R` 的符号 | `R=I`、`R_des=Rx(+10°)` → `e_R = [−0.17365, 0, 0]`；注意是 `−sin10°`，不是 `−0.1745`（`vee` 给一阶形式） |
| 退化保护 | `‖f_des‖ < 1e-6` 或 `‖z_b × x_c‖ < 1e-6` 时回退到 `R_z(yaw_des)`。后者对应 90° 倾斜，X3 权限上限约 47°，碰不到；但不加保护时叉乘的舍入噪声会被归一化放大成**随机姿态**，不报错 |
| 无状态 | 纯 P，无积分、无记忆，所以没有 `reset()` |
| 不需要 `params` | 增益是标量，不含机体参数 |

`kp = [8.0, 8.0, 1.0]`（roll / pitch / yaw），推导见 §9.2。

**欧拉角在这里不可用**：`yaw = 90°` 时同样的倾斜由 roll 吸收，pitch 变成 0。所以测试要用 `R_des[:,2] == f_des/‖f_des‖` 断言，不能用「pitch 应该是多少度」。这也正是选矢量式（B 方案）而不是欧拉角级联的原因。

### 3.3 角速度环（`rate.py`）

```python
error = np.asarray(w_des, float) - np.asarray(omega, float)   # [rad/s]，机体系
self.int = clip(self.int + error*dt, -1, 1)
tau = self.kp*error + self.ki*self.int                        # [N·m]
```

| 项 | 值 |
|---|---|
| `kp` | `[0.71, 1.41, 0.5]` rad/s → N·m/rad/s |
| `ki` | 全 0 |
| `int` | 限幅 ±1 |

**`w_des` 是入参，不是字段。** 这是分层的核心：上游算得出来的量不该由本层持有。

**`w_des` 是机体系角速度**，不是欧拉角速率 `rpy_dot`。水平附近两者数值接近，一倾斜就完全不同，而症状表现为「增益不对」——很容易误诊。

`ki = 0` 时积分器**仍然在累加**（会涨到限幅），只是不参与输出。这是有意保留的旧行为。

`kp` 目前是手调的，**不是** `I·ωc`；构造函数收 `params` 只是为了让四层签名统一（见 §7-4）。

### 3.4 混控（`mixer.py`）

```python
w2 = self.B_inv @ np.array([f, *tau])   # u = [F, τx, τy, τz] = B @ w²
```

`B = params.allocation`，形状 `(4, n_rotors)`，由旋翼位置与转向生成：

```
row0: kf
row1: kf * y_i          (τx = Σ y_i f_i)
row2: -kf * x_i         (τy = -Σ x_i f_i)
row3: -kf * km_ratio * spin_i
```

X3 是 4 旋翼 X 型，`B` 可逆；且 `B⁻¹ @ [F,0,0,0] = (F/(4kf))·1`，即四个电机等速。

**这一层没有物理知识**：不知道重力、不知道姿态、不认识 `State`。它的不变量就是一条可证伪的恒等式：

```
不饱和时   B @ (ω_cmd²) == [f, τ]      误差 < 1e-15
τ = 0 时   四路 ω_cmd 完全相等 == √(f / (4·kf))
```

两条都直接写成断言，不用跑仿真。

### 3.5 饱和缩放 `_scale()`

```python
self.saturated = bool(w2.max() > w2_max or w2.min() < 0.0)   # ← 必须在 _scale 之前
if self.saturated:
    w2 = self._scale(w2)
```

语义：**保留（裁剪后的）均值，按比例 λ 收缩各电机的差异**，使结果落在 `[0, w2_max]`。

- 水平机动时会损失一部分力矩权限；
- 如果均值本身已超上限，`λ = 0`，四个电机全部固定在 `w2_max`，此时**总推力受限、力矩权限也为零**；
- 负的 `w2` 分量同样被压到 0，意味着该侧电机停转。

判定顺序不可交换：`_scale` 之后「是否越界」这个信息就不存在了。

### 3.6 输出与契约

```python
return np.sqrt(np.clip(w2, 0.0, self.w2_max))
```

> ⚠️ [`base.py`](../src/quadfc/control/base.py) 的契约是「控制器不要裁到 `[0, omega_max]`，交给 mixer，这样 runner 才能统计饱和」。本控制器在内部完成了全部限幅，
> 所以额外记录了一个 `MixerController.saturated`（在 `_scale` **之前**取）。
> 注意 `runner.py` 的饱和统计仍用「`cmd` 与 `clip(cmd)` 之差」，对这条路径**恒为 0**；要做饱和统计请读 `cascade.mix.saturated`。

> `w2_max` 与 `omega_max` 别混用：`f` 是 N，`w2` 是 rad²/s²，输出才是 rad/s。把 `clip` 上限写成 `omega_max` 不报错，但电机永远到不了满转。

---

## 4. 参数、整定与边界

### 4.1 小信号模型

在悬停点线性化（`w²` 的开方非线性恰好与 `dF/dw` 抵消），电机一阶滞后 `τ_m` 下：

```
ΔF_act ≈ ΔF_cmd / (1 + τ_m s)
τ_m · z⃛ + z̈ + kd · ż + kp · z = kp · z_des
```

忽略滞后时的标准二阶形式：

```
ωn = √kp            [rad/s]
ζ  = kd / (2√kp)
```

### 4.2 稳定性条件（Routh）

特征多项式 `τ_m s³ + s² + kd·s + kp`：

```
kd > τ_m · kp          （必要且充分，另需 kp > 0）
```

X3 的 `τ_down = 0.025 s` 是较坏情况，所以 `kp=1` 时 `kd` 至少 0.025 才稳定——但只是「不发散」，阻尼比仍接近 0。

### 4.3 推力 / 加速度权限（X3）

| 量 | 值 |
|---|---|
| 总质量 `m` | 1.52 kg |
| 悬停推力 `mg` | 14.911 N |
| 最大推力 `4·kf·ω_max²` | 21.884 N |
| 悬停转速 | 660.4 rad/s（ω_max 的 82.6%） |
| 向上加速度上限 | **+4.59 m/s²** |
| 向下加速度上限 | −9.81 m/s²（电机到 0） |

推论：单步阶跃 `Δz` 若满足 `kp·Δz ≤ 4.59`，就不会在起步阶段饱和。这也是「1 m 阶跃看不出饱和」的原因。

### 4.4 推荐增益

| 目标 | `kp` | `kd` | ωn | ζ | 说明 |
|---|---|---|---|---|---|
| 临界阻尼、已实测 | 2.5 | 3.2 | 1.58 rad/s (0.25 Hz) | 1.01 | 推荐默认值 |
| 更保守 | 1.0 | 2.0 | 1.0 rad/s | 1.0 | 稳态慢约 1.6 倍 |
| 不推荐 | 1.0 | 0.1 | 1.0 rad/s | 0.05 | 仅勉强稳定，Q≈10 |

参数化写法：

```python
wn, zeta = 1.6, 1.0
self.kp = wn ** 2            # 2.56
self.kd = 2 * zeta * wn      # 3.2
```

这三个值是 `AltitudeController.__init__` 的默认值（2.5 / 0.0 / 3.2）。改它们之前先跑 §6.3 的理论对照和 §6.4 的实测记录。

---

## 5. 与仿真器的接口

| 项 | 约定 |
|---|---|
| `controller.step(state, dt)` | `dt = 1/rate`（默认 100 Hz） |
| 用到的状态 | `position[2]`、`velocity[2]`、`quat`（→ `rotation()`）、`omega` |
| 输出 | `(4,)` rad/s，顺序 `rotor_0..3` |
| 时序 | plant 先给出状态 → 控制器算指令 → plant 积分 `dt`（内部 `substep=1e-3` RK4） |
| 电机滞后 | `τ_up = 0.0125 s`，`τ_down = 0.025 s`（plant 侧） |

**坐标与参考系**（写每一层时都要对齐）：

| 量 | 参考系 |
|---|---|
| `position` / `velocity` / `a_des` / `f_des` | 世界系 ENU |
| `quat` / `rotation()` | 机体 → 世界（FLU → ENU） |
| `omega` / `w_des` / `e_R` / `τ` | **机体系** |

`w_des` 和 `omega` 都是机体系，这不是巧合——姿态误差 `e_R` 本来就在机体系算出来，中间不需要任何换系。若哪天把 `w_des` 换成欧拉角速率 `rpy_dot`，水平附近看起来还能飞，一倾斜就全错。

`CascadeController.reset(state)` 转发给有状态的两层：

```python
self.alt.reset(state.position[2])   # int 清零；z_des 仅在 None 时取当前高度
self.rate.reset()                   # int 清零
self.yaw_des = 0.0                  # 无条件归零（见下）
```

`attitude` 和 `mixer` 无状态，不需要 reset。

**CLI 仍只能原地悬停**：不设 `z_des`（保持 `None`）时 `alt.reset` 会取当前高度。若要指定目标高度，得在 `reset` 之后写 `cascade.alt.z_des`——目前 CLI 没有暴露这个参数（§7-2 已修好哨兵，缺的只是命令行入口）。

**`yaw_des` 是无条件归零，和 `z_des` 的哨兵语义不一致。** 从 `yaw=0` 起步或纯 analytic 后端无差别；但 `gz` 后端 teleport 后若机体带非零 yaw，`reset` 会把目标拍回 0°，姿态环立刻把机头拧回世界正东。要「保持当前朝向」得改成哨兵：

```python
# __init__: self.yaw_des = None
# reset:    if self.yaw_des is None: self.yaw_des = state.euler()[2]
```

---

## 6. 测试

### 6.1 工具

三个工具，**每个只串它声明要测的那几层**：

| 工具 | 被测对象 | 隔离手法 |
|---|---|---|
| `step_sine_rate_alt.py` | 高度通道（走 `CascadeController`） | 只覆盖 `alt.kp/kd`，其余层保持默认 |
| `step_sine_att.py` | `AttitudeController` + rate + mixer | 手动串三层，高度环静音（`a_des[2]=0`） |
| `step_sine_rate.py` | `RateController` + `MixerController` | 手动串两层，`f ≡ m·G` |

后两个刻意**不**用 cascade：它们声明的是「测姿态」和「测角速度」，把高度层拖进来只会让测试对象和被声明的不一致。

```bash
cd src
python3 tools/test/step_sine_rate_alt.py --mode both --kp 2.5 --kd 3.2 --plot
python3 tools/test/step_sine_att.py --mode all --plot
python3 tools/test/step_sine_rate.py --mode both --plot
```

高度环工具的参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode step\|sine\|both` | both | 阶跃 / 扫频 / 都做 |
| `--altitude` | 1.0 | 起始高度 |
| `--step` / `--step-at` / `--hold` | 1.0 / 0.5 / 10 | 阶跃幅值、时刻、之后仿真时长 |
| `--amplitude` / `--freqs` | 0.05 / 0.05…0.30 | 正弦幅值与频率点 |
| `--kp` / `--kd` | 取 `AltitudeController` 默认 | 临时覆盖，不改源码 |
| `--csv` | — | 导出 `t,z,vz,thrust,saturated` |

工具在每个控制周期写 `ctrl.alt.z_des = ref(t)`，**不改控制器**。`--kp/--kd` 的默认值直接实例化 `AltitudeController()` 读取，避免工具和控制器各写一份而悄悄失配。

姿态环工具（`step_sine_att.py`）的参数见 §6.7。

### 6.2 指标定义

| 指标 | 定义 |
|---|---|
| `t_rise_10_90` | 首次到达 10% 与 90% 的时间差 |
| `overshoot` | `(峰值 − 目标) / 阶跃幅值` |
| `t_settle_2%` | 最后一次离开 ±2% 带的时刻；`nan` = 结束前未稳定 |
| `steady_err` | 最后 0.5 s 均值 − 目标 |
| `sat` | 读 `mix.saturated` 的采样比例（§7-1 已修） |
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

`saturated` 一列现在直接读控制器的 `self.saturated`（2026-11 修好，见 §7-1），
实测 3 m 1.67%、6 m 5.00%，与上表估算一致。若要看饱和的**物理后果**（推力被钉在 21.884 N），
仍可查 CSV 的 `thrust` 列：

```python
import numpy as np
a = np.genfromtxt("/tmp/sat.csv", delimiter=",", names=True)
m = np.isclose(a["thrust"], 21.884, atol=1e-3)
print(m.sum(), a["t"][m].min(), a["t"][m].max())
```

### 6.6 角速度环测试

工具：[`../src/tools/test/step_sine_rate.py`](../src/tools/test/step_sine_rate.py)

```bash
cd src
python3 tools/test/step_sine_rate.py --mode both --plot
```

分层之后这一层可以直接单独接线，不再需要「把别的增益置 0」这种绕法：

```python
plant = AnalyticPlant(params, substep=1e-3, altitude=z0)
rate_ctl = RateController(params)
mix = MixerController(params)

f = params.mass * G                       # 高度层缺席：推力钉在悬停值
w_des = np.zeros(3)
w_des[axis] = ref(t)
u = mix.step(f, rate_ctl.step(s.omega, w_des, dt))
```

这就是 [`step_sine_rate.py`](../src/tools/test/step_sine_rate.py) 的全部主循环。**不改控制器源码**，与 §6.1 同一手法。

> 副作用：没有高度环就没有 `cos(tilt)` 补偿，倾斜时垂直推力下降，飞机会缓慢下沉。
> 对 `omega` 的测量没有影响（analytic 后端没有地面），但别拿这一组的 `z` 当参考。

#### ⚠️ 一阶近似不够用：电机滞后决定了 ωn 和 ζ

`τ_cmd →(1/g)→ ω_cmd →1/(1+τ_m s)→ ω →g→ τ →1/(I s)→ ω`，混控增益 `g` 在环路两头抵消，剩下

```
H(s) = kp / (I·τ_m·s² + I·s + kp)
ωn   = √(kp / (I·τ_m))            ζ = ½·√(I / (kp·τ_m))
```

所以「带宽 = kp/I」只在忽略电机滞后时成立。`kp/I = 20 rad/s` 的升滚道被电机滞后抬到
`ωn ≈ 40 rad/s`，同时 ζ 落到 1.0——**看起来像带宽翻倍，其实是二阶系统换了形状**。

| 轴 | I [kg·m²] | `kp` | `kp/I` [rad/s] | ωn [rad/s] | ζ |
|---|---|---|---|---|---|
| roll | 0.0357 | 0.71 | 19.9 | 39.9 | 1.00 |
| pitch | 0.0703 | 1.41 | 20.0 | 40.0 | 1.00 |
| yaw | 0.0989 | 0.50 | 5.1 | 20.1 | 1.99 |

（ωn/ζ 用 `τ_up = 0.0125 s`；`τ_down = 0.025 s` 时 ωn 更低、ζ 略高，见下表括号。）

#### 实测（analytic、100 Hz、A = 0.02 rad/s、2026-11）

**阶跃 0 → 0.1 rad/s**

| 轴 | rise 10–90 | 超调 | settle 2% | 稳态误差 | τ_peak | sat | 理论 (ζ) |
|---|---|---|---|---|---|---|---|
| roll | 0.070 s | 3.68 % | 0.190 s | 0.0000 rad/s | 0.071 N·m | 0 % | ζ=1.00 |
| pitch | 0.070 s | 3.77 % | 0.190 s | 0.0000 rad/s | 0.141 N·m | 0 % | ζ=1.00 |
| yaw | 0.380 s | 0 % | 0.700 s | −0.0007 rad/s | 0.050 N·m | 0 % | ζ=1.99 |

roll/pitch 出现 3.7% 超调，而 `ζ = 1.00` 理论上不该有超调：这一环是**不对称**的
（加速用 `τ_up = 12.5 ms`，刹车用 `τ_down = 25 ms`），减速段比加速段慢，回落后会小幅冲过。

**扫频（100 Hz 控制）**

| f [Hz] | roll 增益 / 滞后 | pitch 增益 / 滞后 | yaw 增益 / 滞后 |
|---|---|---|---|
| 0.2 | 1.000 / 3.6° | 1.000 / 3.6° | 0.977 / 14.1° |
| 0.5 | 0.999 / 9.1° | 0.999 / 9.0° | 0.879 / 33.1° |
| 1.0 | 0.996 / 18.3° | 0.997 / 18.2° | 0.674 / 56.7° |
| 2.0 | 0.977 / 37.5° | 0.978 / 37.3° | 0.405 / 83.9° |
| 4.0 | 0.819 / 78.0° | 0.823 / 77.7° | 0.197 / 112.3° |
| 6.0 | 0.562 / 111.9° | 0.567 / 111.7° | 0.117 / 130.5° |
| 8.0 | 0.366 / 135.1° | 0.369 / 135.0° | 0.077 / 144.2° |

| 轴 | 实测 −3 dB | 理论 τ_up / τ_mean / τ_down |
|---|---|---|
| roll | 4.773 Hz | 4.064 / 4.342 / 4.422 Hz |
| pitch | 4.807 Hz | 4.093 / 4.370 / 4.447 Hz |
| yaw | 0.908 Hz | 0.852 / 0.881 / 0.911 Hz |

**100 Hz 的离散化误差是可以量出来的**。同一组频率改到 `--rate 1000`：

| f [Hz] | 100 Hz 增益 / 滞后 | 1000 Hz 增益 / 滞后 | 差值 |
|---|---|---|---|
| 0.5 | 0.999 / 9.1° | 0.997 / 9.1° | 0.002 / 0.0° |
| 2.0 | 0.977 / 37.5° | 0.945 / 36.7° | 0.032 / 0.8° |
| 4.0 | 0.819 / 78.0° | 0.753 / 72.7° | 0.066 / 5.3° |
| 8.0 | 0.366 / 135.1° | 0.348 / 119.2° | 0.018 / 15.9° |

`--rate 1000` 下 −3 dB 为 **4.331 Hz**，与 `τ_mean` 二阶模型 4.342 Hz 差 0.3%，
`f ≤ 40 Hz` 内增益最大偏差 0.002、滞后 0.6°——**实现正确**。

100 Hz 下的偏差就是零阶保持造成的半个采样周期延迟（`T/2 = 5 ms`）：
8 Hz 处 5 ms 对应 `8×360×0.005 = 14.4°`，与实测 15.9° 吻合。滞后吃掉相位裕度，
闭环在 4 Hz 附近反而**更峰**（0.819 > 0.753），所以 100 Hz 读出的 4.77 Hz 带宽是偏乐观的。
要真正把这个内环做到 5 Hz 以上，先把控制频率提到 500 Hz 以上。

#### 力矩权限（从混控矩阵直接算，X3）

| 轴 | 悬停时可用力矩上限 | 说明 |
|---|---|---|
| roll / pitch | ≈ 1.46 N·m | 受总推力约束：升速侧到 `ω_max²` 时降速侧还剩 `232144 rad²/s²` |
| yaw | **≈ 0.112 N·m** | 反扭矩差，比 roll 小一个数量级 |

实测核对：`--step 5` 的 roll 阶跃 `τ_peak = 1.4644 N·m`（= 上表上限），饱和 9%；
`--axis yaw --step 0.3` 的 `τ_peak = 0.1067 N·m`，饱和 10%。

推论：**yaw 的转速指令有硬上限** `ω_des,max ≈ 0.112 / kp[yaw]`。
当前 `kp = 0.5` → 0.22 rad/s；若按 §7-4 把 `kp` 改成 `I·ωc = 1.98`，上限只剩 **0.057 rad/s**。
yaw 想要带宽就得牺牲转速权限，两者不能同时要。

### 6.7 姿态环测试

工具：[`../src/tools/test/step_sine_att.py`](../src/tools/test/step_sine_att.py)

姿态环**没有自己的 setpoint**——倾角是推力的奴隶，所以注射的是**横向加速度**，测的是倾角：

```bash
cd src
python3 tools/test/step_sine_att.py --mode all --plot                  # pitch 轴全量
python3 tools/test/step_sine_att.py --mode step --axis roll            # roll 轴
python3 tools/test/step_sine_att.py --mode step --step 1.0 --with-altitude   # 联调
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode step\|sine\|recover\|yaw\|all` | all | 阶跃 / 扫频 / 倾角恢复 / 偏航阶跃 |
| `--axis pitch\|roll\|yaw` | pitch | 注入 `a_des` 的哪个分量（yaw 模式测 `yaw_des`） |
| `--step` / `--step-at` / `--hold` | 1.0 / 0.3 / 1.0 | 加速度阶跃 [m/s²] 及时刻 |
| `--amplitude` / `--freqs` | 0.3 / 0.5…8 | 正弦幅值 [m/s²] 与频率点 |
| `--tilt` | 10 | recover 模式的初始倾角 [deg] |
| `--yaw-step` | 10 | yaw 阶跃 [deg]，超 10° 会先撞权限 |
| `--kp` | 取 `AttitudeController` 默认 | 三轴统一覆盖 |
| `--with-altitude` | 关 | 打开后高度环同时工作（联调用） |

**符号约定**（写测试时最容易错的一处）：`a_des` 的 x 分量对应 **+pitch**，y 分量对应 **−roll**——同样的正加速度在 ENU+FLU+ZYX 下会表现为负的 roll 角。工具里用 `AXES` 表的第 4 个字段处理。

**目标倾角** `arctan(a/G)`（线性化后可视作 `a/G`）。`--axis roll` 的目标是**负值**，`step_metrics` 已按符号处理。

#### 实测（analytic、100 Hz、高度环静音、2026-11）

**阶跃 `a = 0 → 1.0 m/s²`**（目标倾角 5.820°）

| 指标 | pitch | roll | 理论（K=8） |
|---|---|---|---|
| rise 10–90 | 0.150 s | 0.140 s | 0.171 s |
| overshoot | 0.37 % | 0.56 % | 1.22 % |
| settle 2% | 0.260 s | 0.250 s | 0.275 s |
| steady_err | −0.0000° | 0.0000° | 0 |
| `τ_peak` | 0.897 N·m (τy) | 0.576 N·m (τx) | — |
| sat | 3.0 % | 0 % | — |

实测比三阶理论**快约 12 %**、超调更小：理论模型把内环当成理想二阶，而真实 plant 的电机滞后是**不对称**的（`τ_up < τ_down`），加速段比理论快，减速段比理论慢，两者在阶跃里部分抵消。

**扫频（A = 0.3 m/s²）**

| f [Hz] | gain | lag |
|---|---|---|
| 0.5 | 0.985 | 22.5° |
| 1.0 | 0.941 | 44.8° |
| 2.0 | 0.782 | 88.8° |
| 3.0 | 0.557 | 129.9° |
| 4.0 | 0.351 | 162.9° |
| 6.0 | 0.136 | 195.9°（`−156°` 展开后） |
| 8.0 | 0.062 | 227.0°（`−133°` 展开后） |

实测 −3 dB = **2.292 Hz**，三阶理论 = 1.993 Hz。差距来自同一处：理论用 `τ_up` 估内环 `ωn=40/ζ=1.00`，而实际力矩上升沿用的是 `τ_up`、回落沿用 `τ_down`，等效内环略快。

> 相位超过 180° 是正常的（三阶系统），绘图时要用 `np.unwrap`，否则曲线会在 `±180°` 处跳变。

**倾角恢复**（初始 +10°，力指令朝上）

```
|angle| < 0.5° 于 t = 0.260 s    末态 −0.0000°    峰值 |angle| = 10.000°（无超调）    sat 4.7 %
```

这条是整个姿态环**最有效的正确性测试**：若 `step()` 里漏掉负号，飞机会从 10° 冲到 40° 然后翻掉，而 `a` 阶跃测试**不一定**能发现（它只验证「朝对的方向加速」，不验证收敛）。

**yaw 阶跃 10°**（权限内，`w_des` 峰值 0.1745 < 0.223 rad/s）

| 指标 | 实测 | 理论（K=1, wn=20.1, ζ=1.99） |
|---|---|---|
| rise 10–90 | **1.760 s** | 1.765 s |
| settle 2% | **3.210 s** | 3.198 s |
| overshoot | 0 % | 0 % |
| `τ_peak` | 0.0868 N·m | — |
| sat | 0 % | — |

偏差 0.3 % / 0.4 % —— yaw 是单轴过阻尼、没有横滚耦合，三阶模型在这里几乎精确。

**yaw 超过 ~10° 会先撞权限**：初始 `w_des = K·Δψ`，`Δψ = 0.35 rad` 时 `0.35 > 0.223`，一上来就饱和，实际响应反而更慢。测 yaw 用 `≤ 10°`。

#### ⚠️ 联调时饱和会出现，孤立测姿态层时是 0 %

同一个 `a = 1.0 m/s²` 阶跃，`--with-altitude` 前后：

| | sat | settle |
|---|---|---|
| 高度环静音 | 3.0 % | 0.260 s |
| 高度环工作 | 3.0 % | 0.240 s |
| `a = 2.0`，高度环静音 | — | — |
| `a = 2.0`，高度环工作 | **9.0 %** | **0.520 s** |

原因：一倾斜，垂直推力掉 `cos θ`，飞机下沉，高度环立刻加大 `f`；而 `f` 和力矩共用同一组电机，混控被推到边界。`_scale` 压缩力矩权限 → 姿态响应变慢。

**结论：姿态环的 `sat` 读数必须看联调那组。** 孤立测出来的 0 % 只说明「这一层自己没超限」，不说明系统不饱和。

---

### 6.8 Gazebo 交叉验证

```bash
python3 -m quadfc --backend gz --controller cascade --altitude 2 --duration 8 --gui
```

gz 初始状态不如 analytic 干净（teleport + 落地瞬态），前 1–2 s 不参与指标评价。analytic 与 gz 趋势明显不一致时，优先查符号/单位（参考 [`compare_backends.py`](../src/tools/compare_backends.py)）。

### 6.9 回归测试建议

**分层重构的回归指纹**：改任何一层之后，两组数字必须逐位一致（2026-11 实测，重构前后完全相同）：

| 测试 | 指标 | 期望 |
|---|---|---|
| `step_sine_rate_alt.py --mode step --kp 2.5 --kd 3.2` | rise / OS / settle / steady | `2.130 / 0.000 / 3.770 / -0.000` |
| `step_sine_rate_alt.py --mode step --step 6 --hold 12` | settle / sat | `4.030 / 5.000` |
| `step_sine_rate.py --mode step`（roll） | rise / OS / settle | `0.0700 / 3.6801 / 0.1900` |
| `step_sine_rate.py --mode step`（yaw） | rise / OS / settle | `0.3800 / 0.0000 / 0.7000` |
| `step_sine_att.py --mode step --axis pitch` | rise / OS / settle / steady | `0.1500 / 0.3668 / 0.2600 / -0.0000` |
| `step_sine_att.py --mode yaw` | rise / settle | `1.7600 / 3.2100` |
| `--backend analytic --controller cascade` 悬停 | z / sat | `2.000 / 0.0%` |

任何一位不一样，就说明是「重写」而不是「重构」。

混控层可以脱离仿真直接断言（不饱和时误差 < 1e-15）：

```python
m = MixerController(params)
cmd = m.step(params.weight, np.array([0.01, -0.02, 0.005]))
assert not m.saturated
assert np.abs(params.allocation @ cmd**2 - [params.weight, 0.01, -0.02, 0.005]).max() < 1e-12

cmd = m.step(params.weight, np.zeros(3))          # tau = 0 -> 四路等速
assert np.ptp(cmd) < 1e-9
assert np.isclose(cmd[0], params.hover_omega)
```

姿态层的两个纯函数也可以直接断言：

```python
from quadfc.control.attitude import yaw_des_rotation, attitude_error

f_des = params.mass * np.array([0.0, 0.0, G])
assert np.allclose(yaw_des_rotation(f_des, 0.0), np.eye(3))       # 悬停 -> 水平

f_des = params.mass * np.array([1.0, 0.0, G])
R = yaw_des_rotation(f_des, 0.0)
assert np.allclose(R[:, 2], f_des / np.linalg.norm(f_des))        # 机体 z 指向推力
assert np.isclose(np.degrees(np.arcsin(-R[2, 0])), 5.8204, atol=1e-4)

th = np.radians(10)                                               # e_R 符号
Rx = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
assert np.allclose(attitude_error(np.eye(3), Rx), [-0.173648, 0, 0], atol=1e-6)
```

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
| 1 | ~~控制器内部完成 `_scale` + `clip`，破坏 `base.py` 的约定~~ | 已修：`MixerController` 在 `_scale` **之前**记录 `saturated`，三个测试工具读 `mix.saturated` | — |
| 2 | ~~`reset()` 无条件覆盖 `z_des`~~ | 已修：`AltitudeController.z_des` 用 `None` 作哨兵。**但 CLI 仍没暴露「目标高度」参数**，所以命令行行为还是原地悬停 | 加 `--target-z`，在 `cascade.alt.z_des` 上设值 |
| 3 | ~~`w_des` 恒为 0~~ | 已修：姿态层（矢量式）已接入，`w_des = −kp·e_R`。倾斜恢复见 §6.7 | **水平方向仍未闭环**：`a_des[0:2]` 恒为 0，加速度环后才有航点跟踪 |
| 4 | `RateController.kp` 写死，`I = np.diag(params.inertia)` 未使用 | 换机型即失效；且三轴 `kp/I` 不相等（yaw 只有 5.1 rad/s，见 §6.6） | `self.kp = np.diag(params.inertia) * wc`；注意 yaw 权限只有 0.112 N·m |
| 5 | `ki` 可开但无抗饱和 | 饱和期间积分累积，恢复后过冲 | 条件积分或 back-calculation，可复用 `mix.saturated` |
| 6 | 无 `z̈_des` 前馈 | 正弦/轨迹跟踪存在相位滞后（0.16 Hz 已 65°） | `acc_z += z_des_ddot` |
| 7 | ~~默认 `kd_z = 0.1`~~ | 已改为 `AltitudeController.kp/ki/kd = 2.5 / 0.0 / 3.2` | — |
| 8 | ~~推力沿机体 z 轴，无倾角补偿~~ | 已修：`f = ‖m(a_des + G·e_z)‖`，`1/cos θ` 自动出现。实测 `a=2.0` 时 `‖f_des‖/mg = 1.018` | 横向漂移仍在（没有速度反馈），属于 §7-3 |
| 9 | `_scale` 全超限时退化为恒最大推力 | 同时丢失力矩权限 | 记录并在上层限制期望加速度 |
| 10 | `_scale` 里 `mean` 被 `clip`、`diff` 却基于未 clip 的均值 | 均值 > `w2_max` 时二者不自洽 | 重构时原样保留（改了会动指纹），单独一轮修 |
| 11 | ~~`CascadeController.w_des` 是过渡字段~~ | 已删除，改为 `att.step()` 的返回值 | — |
| 12 | `yaw_des` 在 `reset` 里无条件归零，和 `z_des` 的哨兵不一致 | `gz` 后端 teleport 后带非零 yaw 时，会自主拧回世界正东 | 改成 `None` 哨兵，`reset` 时取当前 `state.euler()[2]` |
| 13 | 姿态环 `kp = [8, 8, 1]` 写死 | 换机型即失效；且三轴不一致是**有意**的（yaw 权限小、内环过阻尼） | 若要参数化，用 `I·ωc` 的比值而不是绝对增益，否则 yaw 会更慢 |
| 14 | 姿态环只有 P | 常值扰动（如不对称推力）下会有稳态倾角误差 | 加 `ki·∫e_R`，配合 `mix.saturated` 做抗饱和 |

问题 1 的修法（已落到代码）：

```python
w2 = self.B_inv @ np.array([f, *tau])
self.saturated = bool(w2.max() > self.w2_max or w2.min() < 0.0)   # 必须在 _scale 之前
if self.saturated:
    w2 = self._scale(w2)
```

工具里读 `sat[i] = mix.saturated`（必须在 `step()` **之后**读）。
修好之后 §6.5 的估算可以被直接验证：3 m 阶跃 → 1.67%（12.5 s × 1.67% ≈ 0.21 s，估 0.19 s），
6 m 阶跃 → 5.00%（≈0.63 s，估 0.58 s）。

---

## 8. 复现命令

```bash
cd /home/nuaawzh/detect_design/src

# 基线（原地悬停，验证配平）
python3 -m quadfc --backend analytic --controller cascade \
    --altitude 2 --duration 5 --no-realtime

# 阶跃 + 扫频，当前推荐增益
python3 tools/test/step_sine_rate_alt.py --mode both --kp 2.5 --kd 3.2 --plot

# 饱和分支
python3 tools/test/step_sine_rate_alt.py --mode step --step 6 --hold 12 \
    --kp 2.5 --kd 3.2 --csv /tmp/sat.csv

# 低频增益校验（验 DC 增益 ≈ 1）
python3 tools/test/step_sine_rate_alt.py --mode sine --freqs 0.01,0.05,0.10

# 角速度环：三轴阶跃 + 扫频
python3 tools/test/step_sine_rate.py --mode both --plot

# 角速度环：排除 100 Hz 零阶保持的影响
python3 tools/test/step_sine_rate.py --mode sine --axis roll --rate 1000 \
    --freqs 0.5,2,4,6,8

# 角速度环：力矩权限边界（yaw ≈0.112 N·m、roll ≈1.46 N·m）
python3 tools/test/step_sine_rate.py --mode step --axis yaw --step 0.3
python3 tools/test/step_sine_rate.py --mode step --axis roll --step 5

# 姿态环：阶跃 + 扫频 + 倾角恢复 + 偏航阶跃
python3 tools/test/step_sine_att.py --mode all --plot
python3 tools/test/step_sine_att.py --mode step --axis roll

# 姿态环联调（高度环同时工作 -> 看真实饱和）
python3 tools/test/step_sine_att.py --mode step --step 2.0 --with-altitude

# Gazebo
python3 -m quadfc --backend gz --controller cascade --altitude 2 --duration 8 --gui
```

---

## 9. 姿态层（B 方案）的设计与整定

姿态层已实现（代码见 §3.2，实测见 §6.7）。本节保留推导，因为它是将来做加速度/速度环的入口。

### 9.1 为什么是矢量式而不是欧拉角级联

| | A. 欧拉角级联 | B. 推力矢量（本轮选的） |
|---|---|---|
| 姿态环输入 | `roll_des, pitch_des, yaw_des` | `f_des` 矢量 + `yaw_des` |
| 上游（速度环）输出 | `roll_des = −kp·(v_y − v_y_des)` 这类经验式 | `a_des = kp(v_des−v) + ki∫ + kd(...)` |
| 大倾角 | 失效（≈30° 以上） | 精确 |
| `cos θ` 补偿 | 要手工除 | **自动**：`f = ‖f_des‖` |
| 姿态误差 | `euler_des − euler`，yaw 要处理 ±π | `e_R`，无奇异 |
| 满足 `yaw=90°` 时的一致性 | 三个角互相串扰 | 天然解耦 |

决定性的一条是 `cos θ`：A 方案要在高度环里手工除，而 B 方案里 `‖m(a_des + G·e_z)‖` 已经包含了它。

### 9.2 增益推导

把 §6.6 实测的内环等效成二阶，再串一个 P：

```
内环:  ω/ω_des = ωn² / (s² + 2ζωn s + ωn²)        roll/pitch: ωn=39.9, ζ=1.00
姿态:  ω_des = −K·e_R                              yaw:        ωn=20.1, ζ=1.99

闭环:  s³ + 2ζωn·s² + ωn²·s + K·ωn²
Routh: K < 2ζωn        ->  roll/pitch 79.8 rad/s，yaw 80.2 rad/s
```

稳定性上限非常宽松（因为内环几乎临界阻尼）——**瓶颈不是稳定，是饱和和 100 Hz 采样**。

roll / pitch 的阶跃预测（`theory_step_metrics` 可复现）：

| K | 闭环极点 | 超调 | rise 10–90 | settle 2% |
|---|---|---|---|---|
| 4 | −51.1, −23.4, −5.3 | 0 % | 0.434 s | 0.803 s |
| 6 | −53.3, −13.3±1.9j | 0 % | 0.252 s | 0.448 s |
| **8** | −55.1, −12.3±8.9j | **1.22 %** | **0.171 s** | **0.275 s** |
| 10 | −56.7, −11.6±12.1j | 4.7 % | 0.131 s | 0.376 s |
| 16 | −60.4, −9.7±18.1j | 17.4 % | 0.083 s | 0.418 s |

**K=8 是 settle 的极值点**：K=10 的 rise 更快，但超调吃掉收益，settle 反而变慢（0.275 → 0.376）。这是选 8 而不是「感觉更快」的 20 的理由。

yaw（内环 `ζ=1.99`，级联后主极点阻尼比只有 ~0.55，**怎么调 K 都压不下去**）：

| K_yaw | 闭环极点 | 超调 | rise 10–90 | settle 2% | 0.2 rad 阶跃 |
|---|---|---|---|---|---|
| 0.5 | −74.6, −4.8, −0.56 | 0 % | 3.95 s | 7.19 s | 不饱和 |
| **1.0** | −74.7, −4.0, −1.36 | 0 % | **1.77 s** | **3.20 s** | 不饱和（峰值 0.198 < 0.223） |
| 1.2 | −74.7, −3.4, −1.89 | 0 % | 1.41 s | 2.50 s | **饱和**（0.24 > 0.223） |

取 **K_yaw = 1.0**：正好卡在权限边界内侧。再往上，饱和会让实际响应比 0.8 还慢——那个 1.41 s 是纸面数字。实测 1.760 s / 3.210 s，与理论差 0.3 % / 0.4 %（§6.7）。

### 9.3 符号约定备忘

| 量 | 方向 |
|---|---|
| `a_des[0] > 0`（向 +x 加速） | `+pitch` |
| `a_des[1] > 0`（向 +y 加速） | **`−roll`**（ENU+FLU+ZYX 下正加速度给负 roll 角） |
| `e_R` | 「`R_des` → `R`」的旋转矢量，所以修正取负 |
| `f_des` 与 `x_c` 平行 | 90° 倾斜，X3 权限上限约 47°，碰不到，但代码里已回退保护 |

### 9.4 稳态倾角参考

`tilt = arctan(a/G)`，`‖f_des‖ = mg·√(1+(a/G)²)`：

| `a` [m/s²] | 稳态倾角 | `‖f_des‖/mg` |
|---|---|---|
| 0.5 | 2.918° | 1.0013 |
| 1.0 | 5.820° | 1.0052 |
| 2.0 | 11.523° | 1.0206 |
| 4.59（向上加速度上限） | 25.074° | 1.1040 |

### 9.5 下一步：加速度 / 速度 / 位置环

姿态层打通之后，接缝已经现成——**只要往 `a_des[0:2]` 写东西**：

```
位置环:  p_des -> v_des            (最外层)
速度环:  v_des -> a_des[0:2]       (写入 cascade 的 a_des)
```

三件要先想清楚的事：

1. **`a_des` 必须限幅。** 现在的 `‖f_des‖` 没有上界检查，`a_des` 一旦超过 +4.59 m/s²（推力上限）或让 `‖f_des‖` 退化为小值，混控只会老实报 `saturated`，不会纠错。限幅应放在**算 `f_des` 之前**。
2. **姿态环 ~2.3 Hz 带宽决定了外环上限。** 经验上外环带宽不超过内环的 1/3～1/5，也就是**速度环 ≤ 0.5 Hz、位置环 ≤ 0.15 Hz**。这比高度环的 0.16 Hz 带宽宽松，但比直觉紧得多。
3. **`z` 通道已经有了。** 位置环要加的只是 `x`/`y` 两个通道——不要在位置环里再写一遍高度，否则两个积分器会互相打架（§7-3）。

---

