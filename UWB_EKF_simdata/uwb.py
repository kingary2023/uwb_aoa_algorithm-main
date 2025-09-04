import numpy as np
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
from simulate_data import simulate_data

# 初始化参数
lambda_val = 299792458 / 6.5e9  # 波长
dAntenna = 0.0204  # 天线间距
eps = np.finfo(float).eps
DeltaThetaDeg = [0, 120, 240, 0 - 90, 120 - 90, 240 - 90]
# DeltaThetaDeg = [0, 120]

kAntennaPairCount = len(DeltaThetaDeg)

Nab = np.zeros((2, kAntennaPairCount))  # 天线向量
Nab[0, 0] = 0
Nab[1, 0] = dAntenna

kAntennaCount = Nab.shape[1]


for col in range(1, kAntennaPairCount):
    theta = DeltaThetaDeg[col] / 180 * np.pi
    Nab[:, col] = np.dot(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], Nab[:, 0]
    )

print(Nab)
print(Nab.shape[0])
print(kAntennaCount)



# 计算向量和等于零的组合
"zero antenna group"
Index = range(kAntennaPairCount)

for count in range(3, kAntennaPairCount):
    Choose = np.array(list(combinations(Index, count)))

    for row in range(len(Choose)):
        NabSum = np.array([0, 0])

        for col in range(len(Choose[row])):
            NabSum = NabSum + Nab[:, Choose[row][col] - 1]

        if np.all(np.abs(NabSum) < eps):
            chosen_row = Choose[row, :]


disStd = 0.05 * 1.0
phiStd = 5 * 1.0 * np.pi / 180

anchorZ = 3
AnchorPos = np.array([0, 0, anchorZ])  #基站坐标

dataSource = 0

match dataSource:
    case 0:  # 采用仿真数据
        tagZ = 1.8
        tagHeightRange = 0  # 标签在指定平面上下活动的范围
# ######   标量       标签真实轨迹   真实方位角俯仰角     相位差测量值（含噪声）   距离测量值（含噪声）
        kSampleCount, TrackTruth,    PhiThetaTruth  ,       PhiDiff,              Dis = simulate_data(
            Nab, lambda_val, AnchorPos, tagZ, disStd, phiStd
        )
    # case 1:
    #     DataTable = pd.read_excel(
    #         "data/NLink_LinkTrack_AOA_Node_Frame0_20210326_195200.xlsx"
    #     )
    #     data = DataTable.iloc[:, :]
    case _:
        print("dataSource is invalid! Check and retry.")


# if dataSource != 0:
#     # 若非仿真数据，需要从 table 中提取
#     kSampleCount = data.shape[0]
#     Dis = (data[:, 6] + data[:, 12] + data[:, 18]) / 3
#     Theta = (90 + np.array([data[:, 7], data[:, 13], data[:, 19]])) * np.pi / 180
#     PhiDiff = 2 * np.pi / lambda_val * dAntenna * np.cos(Theta)



# disStd = 0.1 * 0.6 * 0.01
# angleStd = 5 * 0.6 * np.pi / 180 * 0.01

TrackWithDis = np.full((3, kSampleCount), np.nan)  # PDOA+TWR
TrackWithoutDis = np.copy(TrackWithDis)  # PDOA + 固定标签高度
TrackWithDisAndFixedHeight = np.copy(TrackWithDis)  # PDOA + TWR + 固定标签高度
PhiTheta = np.full((2, kSampleCount), np.nan)  # 空间角

# 转换 aoa 角度到统一坐标 0~360
A = Nab.T
print(  A)
AAA = np.linalg.lstsq(A.T @ A, A.T, rcond=None)[0]

for index in range(kSampleCount):
    r = Dis[index]
    DeltaR = PhiDiff[:, index] * lambda_val / (2 * np.pi)
    B = A[:, 0] * AnchorPos[0] + A[:, 1] * AnchorPos[1] - r * DeltaR
    X = AAA @ B
    # FIXME 这里只是为了临时屏蔽大角度边缘区域高度计算虚数问题
    z = AnchorPos[2] - np.sqrt(
        np.abs(r**2 - (X[0] - AnchorPos[0]) ** 2 - (X[1] - AnchorPos[1]) ** 2)
    )
    TrackWithDis[:, index] = np.concatenate((X, [z]))

    # 仅根据相位差计算空间角
    B1 = -DeltaR
    X1 = AAA @ B1
    a = X1[0]
    b = X1[1]
    PhiTheta[0, index] = np.arctan2(b, a)
    sin_theta = np.sqrt(a**2 + b**2)
    sin_theta_clipped = np.clip(sin_theta, 0, 1)  # 限制在[0,1]范围内
    

    if z - AnchorPos[2] >= 0:
        PhiTheta[1, index] = np.arcsin(sin_theta_clipped)
        # PhiTheta[1, index] = np.arcsin(np.sqrt(a**2 + b**2))
    else:
        PhiTheta[1, index] = np.pi - np.arcsin(sin_theta_clipped)
        # PhiTheta[1, index] = np.pi - np.arcsin(np.sqrt(a**2 + b**2))
# 
        


    # 根据空间角及标签设定高度，基站实际高度计算位置
    dz = tagZ - anchorZ
    phi = PhiTheta[0, index]   #方向角
    theta = PhiTheta[1, index]  #俯仰角
    r1 = dz / np.cos(theta)
    TrackWithoutDis[:, index] = np.array(
        [
            AnchorPos[0] + r1 * np.sin(theta) * np.cos(phi),
            AnchorPos[1] + r1 * np.sin(theta) * np.sin(phi),
            tagZ,
        ]
    )

    # print(r)
    # print(dz)
    # 仅使用水平角+距离+固定高度计算
    # rh = np.sqrt(r**2 - dz**2)
    rh_squared = r**2 - dz**2
    rh = np.sqrt(np.maximum(rh_squared, 0))  # 确保非负
    if rh_squared < 0:
        rh = 0  # 或替换为合理小值（如 rh = np.abs(r) - np.abs(dz)）


    TrackWithDisAndFixedHeight[:, index] = np.array(
        [AnchorPos[0] + rh * np.cos(phi), AnchorPos[1] + rh * np.sin(phi), tagZ]
    )


plt.figure()
plt.plot(np.arange(1, kSampleCount + 1), Dis, label="Dis")
plt.legend()


# #########################
# 加入EKF
############################

# 初始状态（假设标签从原点静止启动）
x_est = np.array([0, 0, 0, 0, 0, 0])  # [x, y, z, vx, vy, vz]
P_est = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])  # 初始协方差矩阵

# 初始化EKF轨迹存储（3行：x,y,z；kSampleCount列）
TrackEKF = np.full((3, kSampleCount), np.nan)  # 初始值为NaN

print
# 过程噪声（CT模型需更大噪声）
Q = np.diag([0.5, 0.5, 0.1, 0.3, 0.3, 0.01])

# 观测噪声（从仿真参数生成）
R_phi = (phiStd ** 2) * np.eye(kAntennaCount)  # 相位差噪声
R_dis = disStd ** 2                           # 距离噪声
R = np.block([[R_phi, np.zeros((kAntennaCount, 1))],
              [np.zeros((1, kAntennaCount)), R_dis]])



def observation_model(x, AnchorPos, Nab, lambda_val):
    # 提取位置
    pos = x[:3]
    
    # 计算距离
    r = np.linalg.norm(pos - AnchorPos)
    
    # 计算相位差
    vTO = AnchorPos - pos
    vTO_normalized = vTO / r  # 单位方向向量
    PhiDiff_ekf = np.zeros(Nab.shape[1])
    for i in range(Nab.shape[1]):
        vNab = np.array([Nab[0, i], Nab[1, i], 0])  # 天线对向量
        deltaR = np.dot(vTO_normalized, vNab)
        PhiDiff_ekf[i] = (2 * np.pi / lambda_val) * deltaR
    
    # 合并观测值
    z = np.concatenate([PhiDiff_ekf, [r]])
    return z

def get_F_ct(x, dt):
    vx, vy, vz = x[3], x[4], x[5]
    v_xy = np.sqrt(vx**2 + vy**2)
    if v_xy > 0.1:  # 避免除零
        omega = np.arctan2(vy, vx)  # 当前速度方向
        F = np.array([
            [1, 0, 0, np.sin(omega*dt)/omega, (np.cos(omega*dt)-1)/omega, 0],
            [0, 1, 0, (1-np.cos(omega*dt))/omega, np.sin(omega*dt)/omega, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, np.cos(omega*dt), -np.sin(omega*dt), 0],
            [0, 0, 0, np.sin(omega*dt), np.cos(omega*dt), 0],
            [0, 0, 0, 0, 0, 1]
        ])
    else:
        F = np.eye(6)  # 低速时退化为CV模型
    return F


for index in range(kSampleCount):
    # 获取当前测量值
    z_meas = np.concatenate([PhiDiff[:, index], [Dis[index]]])
    
    # --- 预测步（CT模型）---
    dt = 1.0 / kSampleCount  # 根据实际时间调整
    F = get_F_ct(x_est, dt)
    x_pred = F @ x_est
    P_pred = F @ P_est @ F.T + Q
    
    # --- 更新步 ---
    # 计算雅可比矩阵H（数值法简化实现）
    H = np.zeros((kAntennaCount + 1, 6))
    h0 = observation_model(x_pred, AnchorPos, Nab, lambda_val)  #相位差
    eps = 1e-6
    for i in range(6):
        dx = np.zeros(6)
        dx[i] = eps
        h_eps = observation_model(x_pred + dx, AnchorPos, Nab, lambda_val)
        H[:, i] = (h_eps - h0) / eps
    
    # 卡尔曼增益
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    
    # 状态更新
    z_pred = observation_model(x_pred, AnchorPos, Nab, lambda_val)  #估计值
    x_est = x_pred + K @ (z_meas - z_pred)
    P_est = (np.eye(6) - K @ H) @ P_pred   #协方差更新
    
    # 存储结果
    TrackEKF[:, index] = x_est[:3]



plt.figure(figsize=(10, 6))
plt.plot(TrackTruth[0], TrackTruth[1], 'r-', linewidth=2, label='groundtruth')
plt.plot(TrackEKF[0], TrackEKF[1], 'b--', linewidth=1.5, label='EKFestimate')
plt.scatter(AnchorPos[0], AnchorPos[1], c='green', s=100, marker='*', label='base')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('EKF(2D)')
plt.legend()
plt.grid(True)
plt.axis('equal')  # 保证x/y轴比例一致


plt.show()
