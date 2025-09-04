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



## 2. 非线性模型
### 2.1 CV运动模型（也尝试过再加一个CA模型（由于状态向量无加速度，实际将加速度设为一种噪声），用IMM，但效果不佳，偏离相对明显）
状态转移函数：
$$
f(\mathbf{X}) = \begin{bmatrix}
x + v_x\Delta t \\
y + v_y\Delta t \\
z + v_z\Delta t \\
v_x \\
v_y \\
v_z
\end{bmatrix}
$$



### 2.2 观测函数

$$
\begin{aligned}
x_{12}x+y_{12}y&=x_{12}x_a + y_{12}y_a -r\frac{\lambda\varphi_{12}}{2\pi}\\
x_{34}x+y_{34}y&=x_{34}x_a + y_{34}y_a -r\frac{\lambda\varphi_{34}}{2\pi}\\
x_{56}x+y_{56}y&=x_{56}x_a + y_{56}y_a -r\frac{\lambda\varphi_{56}}{2\pi}\\
\end{aligned}
$$

从而推导出


$$
h(\mathbf{X}) = \begin{bmatrix}
\varphi_{12} \\
\varphi_{34} \\
\varphi_{56} \\
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
\frac{\partial \varphi_{34}}{\partial x} & \frac{\partial \varphi_{34}}{\partial y} & \frac{\partial \Delta r_{34}}{\partial z} & 0 & 0 & 0 \\
\frac{\partial \varphi_{56}}{\partial x} & \frac{\partial \varphi_{56}}{\partial y} & \frac{\partial \Delta r_{56}}{\partial z} & 0 & 0 & 0 \\
\frac{\partial r}{\partial x} & \frac{\partial r}{\partial y} & \frac{\partial r}{\partial z} & 0 & 0 & 0
\end{bmatrix}
$$

## 4. EKF算法流程
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

## 5. 噪声参数
| 参数 | 值 | 说明 |
|------|----|------|
| $\mathbf{Q}$ | $\text{diag}([0.01, 0.08, 0.08, 0.5, 0.1, 0.1])$ | 过程噪声协方差 |
| $\sigma_\phi$ | $5^\circ$ | 相位测量标准差 |
| $\sigma_d$ | 0.05m | 距离测量标准差 |
| $\lambda$ | 0.125m | UWB信号波长 |
| $\mathbf{R}$ | $\text{diag}([(\sigma_\phi)^2, (\sigma_\phi)^2, (\sigma_\phi)^2, (\sigma_d)^2])$ | 观测噪声协方差 |
| $\mathbf{P}$ | $\text{diag}([0.05, 0.05, 0.05, 0.2, 0.2, 0.2])$ | 初始协方差 |


## 6. 仿真结果与分析


| 轨迹 | 2D估计 | 3D估计 | Z误差 |
|------|--------|--------|-------|
| 0 | ![fig](result/0/1.png) | ![fig](result/0/2.png) | ![fig](result/0/3.png) |
| 1 | ![fig](result/1/1.png) | ![fig](result/1/2.png) | ![fig](result/1/3.png) |
| 2 | ![fig](result/2/1.png) | ![fig](result/2/2.png) | ![fig](result/2/3.png) |
| 3 | ![fig](result/3/1.png) | ![fig](result/3/2.png) | ![fig](result/3/3.png) |
| 4 | ![fig](result/5/1.png) | ![fig](result/5/2.png) | ![fig](result/5/3.png) |
| 5 | ![fig](result/6/1.png) | ![fig](result/6/2.png) | ![fig](result/6/3.png) |
| 6 | ![fig](result/7/1.png) | ![fig](result/7/2.png) | ![fig](result/7/3.png) |
| 7 | ![fig](result/8/1.png) | ![fig](result/8/2.png) | ![fig](result/8/3.png) |

**分析：**  
- 整体来看，轨迹1/6/7估计效果较差 ，其他相对较准确
- 总体XY估计相对准确（相位差与XY强相关）  
- 但Z估计模糊（现有观测关联较弱）    
- 我觉得可以新增俯仰角等与Z强相关的观测量

| 152 | ![fig](result/152.png) | 158 | ![fig](result/158.png) |
|-----|-------------------------|-----|-------------------------|
| 160 | ![fig](result/160.png) | 165 | ![fig](result/165.png) |
|-----|-------------------------|-----|-------------------------|
| 175 | ![fig](result/175.png) | 178 | ![fig](result/178.png) |