# 扩展卡尔曼滤波(EKF)实现文档

## 1. 状态空间模型
定义状态向量：
$$
\mathbf{X}_k = \begin{bmatrix} 
x \\ y \\ z \\ v_x \\ v_y \\ v_z 
\end{bmatrix}，初始值 = \begin{bmatrix} 0 \\ 0 \\ 1.8 \\ 0 \\ 0 \\ 0
\end{bmatrix}
$$

状态方程：
$$
\mathbf{X}_k = f(\mathbf{X}_{k-1}) + \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(0,\mathbf{Q})
$$

观测方程：
$$
\mathbf{Z}_k = h(\mathbf{X}_k) + \mathbf{v}_k, \quad \mathbf{v}_k \sim \mathcal{N}(0,\mathbf{R})
$$





### 1.2 离散时间动力学方程
$$
\mathbf{X}_k = \underbrace{
\begin{bmatrix}
1 & 0 & 0 & \Delta t & 0 & 0 \\
0 & 1 & 0 & 0 & \Delta t & 0 \\
0 & 0 & 1 & 0 & 0 & \Delta t \\
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
}_{\mathbf{F}} \mathbf{X}_{k-1} + 
\underbrace{
\begin{bmatrix}
\frac{\Delta t^2}{2} & 0 & 0 \\
0 & \frac{\Delta t^2}{2} & 0 \\
0 & 0 & \frac{\Delta t^2}{2} \\
\Delta t & 0 & 0 \\
0 & \Delta t & 0 \\
0 & 0 & \Delta t
\end{bmatrix}
}_{\mathbf{G}} \mathbf{a}_k
$$

#### 3. 过程噪声协方差推导
$$
\begin{aligned}
\mathbf{Q} &= \mathbf{G}\mathbf{G}^\top \sigma_a^2 \\
&= \begin{bmatrix}
\frac{\Delta t^4}{4} & 0 & 0 & \frac{\Delta t^3}{2} & 0 & 0 \\
0 & \frac{\Delta t^4}{4} & 0 & 0 & \frac{\Delta t^3}{2} & 0 \\
0 & 0 & \frac{\Delta t^4}{4} & 0 & 0 & \frac{\Delta t^3}{2} \\
\frac{\Delta t^3}{2} & 0 & 0 & \Delta t^2 & 0 & 0 \\
0 & \frac{\Delta t^3}{2} & 0 & 0 & \Delta t^2 & 0 \\
0 & 0 & \frac{\Delta t^3}{2} & 0 & 0 & \Delta t^2
\end{bmatrix} \sigma_a^2
\end{aligned}
$$



## 2. 非线性模型
### 2.1 CV运动模型
#### 2.1.1 状态转移函数
$$
f(\mathbf{X}) = \begin{bmatrix}
x + v_x\Delta t \\
y + v_y\Delta t \\
z + v_z\Delta t \\
v_x \\
v_y \\
v_z
\end{bmatrix},   \mathbf{Q}为过程噪声常量
$$

### 2.2 CA运动模型
#### 2.2.1 状态转移函数
$$
f(\mathbf{X}) = \begin{bmatrix}
x + v_x\Delta t \\
y + v_y\Delta t \\
z + v_z\Delta t \\
v_x \\
v_y \\
v_z
\end{bmatrix},   \mathbf{Q}为加速度噪声（非常量）
$$

#### 2.2.2 离散时间动力学方程
$$
\mathbf{X}_k = \underbrace{
\begin{bmatrix}
1 & 0 & 0 & \Delta t & 0 & 0 \\
0 & 1 & 0 & 0 & \Delta t & 0 \\
0 & 0 & 1 & 0 & 0 & \Delta t \\
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
}_{\mathbf{F}} \mathbf{X}_{k-1} + 
\underbrace{
\begin{bmatrix}
\frac{\Delta t^2}{2} & 0 & 0 \\
0 & \frac{\Delta t^2}{2} & 0 \\
0 & 0 & \frac{\Delta t^2}{2} \\
\Delta t & 0 & 0 \\
0 & \Delta t & 0 \\
0 & 0 & \Delta t
\end{bmatrix}
}_{\mathbf{G}} \mathbf{a}_k
$$

#### 2.2.3 过程噪声协方差推导
$$
\begin{aligned}
\mathbf{Q} &= \mathbf{G}\mathbf{G}^\top \sigma_a^2 \\
&= \begin{bmatrix}
\frac{\Delta t^4}{4} & 0 & 0 & \frac{\Delta t^3}{2} & 0 & 0 \\
0 & \frac{\Delta t^4}{4} & 0 & 0 & \frac{\Delta t^3}{2} & 0 \\
0 & 0 & \frac{\Delta t^4}{4} & 0 & 0 & \frac{\Delta t^3}{2} \\
\frac{\Delta t^3}{2} & 0 & 0 & \Delta t^2 & 0 & 0 \\
0 & \frac{\Delta t^3}{2} & 0 & 0 & \Delta t^2 & 0 \\
0 & 0 & \frac{\Delta t^3}{2} & 0 & 0 & \Delta t^2
\end{bmatrix} \sigma_a^2
\end{aligned}
$$

### 2.3 观测函数

$$
\begin{aligned}
x_{12}x+y_{12}y&=x_{12}x_a + y_{12}y_a -r\frac{\lambda\varphi_{12}}{2\pi}\\
x_{34}x+y_{34}y&=x_{34}x_a + y_{34}y_a -r\frac{\lambda\varphi_{34}}{2\pi}\\
x_{56}x+y_{56}y&=x_{56}x_a + y_{56}y_a -r\frac{\lambda\varphi_{56}}{2\pi}\\
x_{78}x+y_{78}y&=x_{78}x_a + y_{78}y_a -r\frac{\lambda\varphi_{78}}{2\pi}\\
\end{aligned}
$$

从而推导出


$$
h(\mathbf{X}) = \begin{bmatrix}
\varphi_{12} \\
\varphi_{34} \\
\varphi_{56} \\
\varphi_{78} \\
r
\end{bmatrix}
$$

其中：
$$
r = \sqrt{(x-x_a)^2 + (y-y_a)^2 + (z-z_a)^2}
$$



## 3. 雅可比矩阵
### 3.1 状态转移
$$
\mathbf{F} = \begin{bmatrix}
1 & 0 & 0 & \Delta t & 0 & 0 \\
0 & 1 & 0 &0 & \Delta t & 0 \\
0 & 0 & 1 & 0 & 0 & \Delta t \\
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 &0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
$$



### 3.2 观测雅可比
$$
\mathbf{H} \triangleq \frac{\partial h}{\partial \mathbf{X}} = 
\begin{bmatrix}
\frac{\partial \varphi_{12}}{\partial x} & \frac{\partial \varphi_{12}}{\partial y} & \frac{\partial \varphi_{12}}{\partial z} & 0 & 0 & 0 \\
\frac{\partial \varphi_{34}}{\partial x} & \frac{\partial \varphi_{34}}{\partial y} & \frac{\partial \varphi_{34}}{\partial z} & 0 & 0 & 0 \\
\frac{\partial \varphi_{56}}{\partial x} & \frac{\partial \varphi_{56}}{\partial y} & \frac{\partial \varphi_{56}}{\partial z} & 0 & 0 & 0 \\
\frac{\partial \varphi_{78}}{\partial x} & \frac{\partial \varphi_{78}}{\partial y} & \frac{\partial \varphi_{78}}{\partial z} & 0 & 0 & 0 \\
\frac{\partial r}{\partial x} & \frac{\partial r}{\partial y} & \frac{\partial r}{\partial z} & 0 & 0 & 0
\end{bmatrix}
$$

## 4. EKF+IMM算法流程
### 4.1 预测步骤
$$
\begin{aligned}
\mathbf{X}_{k} &= \mathbf{F}\mathbf{X}_{k-1}\\
\mathbf{P}_{k} &= \mathbf{F}\mathbf{P}_{k-1}\mathbf{F}^\top + \mathbf{Q}
\end{aligned}
$$

### 4.2 更新步骤
$$
\begin{aligned}
\mathbf{K}_k &= \mathbf{P}_{k}\mathbf{H}^\top(\mathbf{H}\mathbf{P}_{k}\mathbf{H}^\top + \mathbf{R})^{-1} \\
\hat{\mathbf{X}}_k &= \mathbf{X}_{k} + \mathbf{K}_k(\mathbf{Z}_k - h(\mathbf{X}_{k})) \\
\hat{\mathbf{P}}_k &= (\mathbf{I} - \mathbf{K}_k\mathbf{H})\mathbf{P}_{k}
\end{aligned}
$$

## 4. EKF+IMM算法流程

### 4.1 单模型预测（并行）
$$
\begin{aligned}
\mathbf{X}_{k|k-1}^{(i)} &= \mathbf{F}^{(i)}\mathbf{X}_{k-1}^{(i)} \\
\mathbf{P}_{k|k-1}^{(i)} &= \mathbf{F}^{(i)}\mathbf{P}_{k-1}^{(i)}(\mathbf{F}^{(i)})^\top + \mathbf{Q}^{(i)}
\end{aligned}
$$

### 4.2 单模型更新（并行）
$$
\begin{aligned}
\mathbf{K}_k^{(i)} &= \mathbf{P}_{k|k-1}^{(i)} \mathbf{H}^{(i)\top} \left( \mathbf{H}^{(i)} \mathbf{P}_{k|k-1}^{(i)} \mathbf{H}^{(i)\top} + \mathbf{R} \right)^{-1} \\
\mathbf{X}_k^{(i)} &= \mathbf{X}_{k|k-1}^{(i)} + \mathbf{K}_k^{(i)} (\mathbf{Z}_k - h^{(i)}(\mathbf{X}_{k|k-1}^{(i)})) \\
\mathbf{P}_k^{(i)} &= (\mathbf{I} - \mathbf{K}_k^{(i)}\mathbf{H}^{(i)}) \mathbf{P}_{k|k-1}^{(i)}
\end{aligned}
$$

### 4.3 模型似然计算
$$
\mathcal{L}_k^{(i)} = \frac{\exp\left(-\frac{1}{2} \boldsymbol{\nu}^{(i)\top} \mathbf{S}^{(i)-1} \boldsymbol{\nu}^{(i)}\right)}{\sqrt{(2\pi)^n \det \mathbf{S}^{(i)}}}
$$
其中：
$$
\begin{aligned}
\boldsymbol{\nu}^{(i)} &= \mathbf{Z}_k - h^{(i)}(\mathbf{X}_{k|k-1}^{(i)}) \\
\mathbf{S}^{(i)} &= \mathbf{H}^{(i)} \mathbf{P}_{k|k-1}^{(i)} \mathbf{H}^{(i)\top} + \mathbf{R}
\end{aligned}
$$

### 4.4 模型概率更新
$$
\mu_k^{(i)} = \frac{\mathcal{L}_k^{(i)} \sum_{j=1}^r p_{ji} \mu_{k-1}^{(j)}}{\sum_{m=1}^r \mathcal{L}_k^{(m)} \sum_{j=1}^r p_{jm} \mu_{k-1}^{(j)}}
$$

### 4.5 输出融合
$$
\begin{aligned}
\hat{\mathbf{X}}_k &= \sum_{i=1}^r \mu_k^{(i)} \mathbf{X}_k^{(i)} \\
\hat{\mathbf{P}}_k &= \sum_{i=1}^r \mu_k^{(i)} \left[ \mathbf{P}_k^{(i)} + (\mathbf{X}_k^{(i)} - \hat{\mathbf{X}}_k)(\cdot)^\top \right]
\end{aligned}
$$

## 5. 噪声参数
| 参数 | 值 | 说明 |
|------|----|------|
| $\mathbf{Q}$ | $\text{diag}([0.5, 0.5, 0.01, 0.3, 0.3, 0.001])$ | 过程噪声协方差 |
| $\sigma_\phi$ | $5^\circ$ | 相位测量标准差 |
| $\sigma_d$ | 0.05m | 距离测量标准差 |
| $\lambda$ | 0.0461m | UWB信号波长 |
| $\mathbf{R}$ | $\text{diag}([(\sigma_\phi)^2, (\sigma_\phi)^2, (\sigma_\phi)^2, (\sigma_d)^2])$ | 观测噪声协方差 |
| $\mathbf{P}$ | $\text{diag}([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])$ | 初始协方差 |

