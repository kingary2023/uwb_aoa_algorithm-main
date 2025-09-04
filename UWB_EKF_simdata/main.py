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
AnchorPos = np.array([0, 0, anchorZ])

dataSource = 0

match dataSource:
    case 0:  # 采用仿真数据
        tagZ = 1.8
        tagHeightRange = 0  # 标签在指定平面上下活动的范围

        kSampleCount, TrackTruth, PhiThetaTruth, PhiDiff, Dis = simulate_data(
            Nab, lambda_val, AnchorPos, tagZ, disStd, phiStd
        )
    # case 1:
    #     DataTable = pd.read_excel(
    #         "data/NLink_LinkTrack_AOA_Node_Frame0_20210326_195200.xlsx"
    #     )
    #     data = DataTable.iloc[:, :]
    case _:
        print("dataSource is invalid! Check and retry.")


if dataSource != 0:
    # 若非仿真数据，需要从 table 中提取
    kSampleCount = data.shape[0]
    Dis = (data[:, 6] + data[:, 12] + data[:, 18]) / 3
    Theta = (90 + np.array([data[:, 7], data[:, 13], data[:, 19]])) * np.pi / 180
    PhiDiff = 2 * np.pi / lambda_val * dAntenna * np.cos(Theta)

# disStd = 0.1 * 0.6 * 0.01
# angleStd = 5 * 0.6 * np.pi / 180 * 0.01

TrackWithDis = np.full((3, kSampleCount), np.nan)  # PDOA+TWR
TrackWithoutDis = np.copy(TrackWithDis)  # PDOA + 固定标签高度
TrackWithDisAndFixedHeight = np.copy(TrackWithDis)  # PDOA + TWR + 固定标签高度
PhiTheta = np.full((2, kSampleCount), np.nan)  # 空间角

# 转换 aoa 角度到统一坐标 0~360
A = Nab.T
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

    # 根据空间角及标签设定高度，基站实际高度计算位置
    dz = tagZ - anchorZ
    phi = PhiTheta[0, index]
    theta = PhiTheta[1, index]
    r1 = dz / np.cos(theta)
    TrackWithoutDis[:, index] = np.array(
        [
            AnchorPos[0] + r1 * np.sin(theta) * np.cos(phi),
            AnchorPos[1] + r1 * np.sin(theta) * np.sin(phi),
            tagZ,
        ]
    )

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

plt.figure()
for index in range(PhiDiff.shape[0]):
    plt.plot(
        np.arange(1, kSampleCount + 1),
        PhiDiff[index, :] * 180 / np.pi,
        label=f"phi diff {index}",
    )
plt.plot(
    np.arange(1, kSampleCount + 1),
    np.sum(PhiDiff, axis=0) * 180 / np.pi,
    label="phi diff sum",
)
plt.legend()

plt.figure()
if "TrackTruth" in locals():
    plt.plot(TrackTruth[0, :], TrackTruth[1, :], label="TrackWithDis groundtruth")

plt.plot(TrackWithDis[0, :], TrackWithDis[1, :], label="TrackWithDis")
# plt.plot(TrackWithoutDis[0, :], TrackWithoutDis[1, :], label="TrackWithoutDis")
plt.plot(
    TrackWithDisAndFixedHeight[0, :],
    TrackWithDisAndFixedHeight[1, :],
    label="TrackWithDisAndFixedHeight",
)
plt.legend()

plt.figure()
plt.plot(np.arange(1, kSampleCount + 1), TrackWithDis[0, :], label="TrackWithDis_x")
plt.plot(np.arange(1, kSampleCount + 1), TrackWithDis[1, :], label="TrackWithDis_y")
plt.plot(np.arange(1, kSampleCount + 1), TrackWithDis[2, :], label="TrackWithDis_z")

plt.plot(
    np.arange(1, kSampleCount + 1), TrackWithoutDis[0, :], label="TrackWithoutDis_x"
)
plt.plot(
    np.arange(1, kSampleCount + 1), TrackWithoutDis[1, :], label="TrackWithoutDis_y"
)
plt.plot(
    np.arange(1, kSampleCount + 1), TrackWithoutDis[2, :], label="TrackWithoutDis_z"
)

plt.plot(
    np.arange(1, kSampleCount + 1),
    TrackWithDisAndFixedHeight[0, :],
    label="TrackWithDisAndFixedHeight_x",
)
plt.plot(
    np.arange(1, kSampleCount + 1),
    TrackWithDisAndFixedHeight[1, :],
    label="TrackWithDisAndFixedHeight_y",
)
plt.plot(
    np.arange(1, kSampleCount + 1),
    TrackWithDisAndFixedHeight[2, :],
    label="TrackWithDisAndFixedHeight_z",
)

if "TrackTruth" in locals():
    plt.plot(np.arange(1, kSampleCount + 1), TrackTruth[0, :], label="GroundTruth_x")
    plt.plot(np.arange(1, kSampleCount + 1), TrackTruth[1, :], label="GroundTruth_y")
    plt.plot(np.arange(1, kSampleCount + 1), TrackTruth[2, :], label="GroundTruth_z")

plt.legend()

if "TrackTruth" in locals():
    plt.figure()
    plt.plot(
        range(1, kSampleCount + 1),
        TrackWithDis[0, :] - TrackTruth[0, :],
        label="error_x(TrackWithDis-groundtruth)",
    )
    plt.plot(
        range(1, kSampleCount + 1),
        TrackWithDis[1, :] - TrackTruth[1, :],
        label="error_y(TrackWithDis-groundtruth)",
    )
    plt.plot(
        range(1, kSampleCount + 1),
        TrackWithDis[2, :] - TrackTruth[2, :],
        label="error_z(TrackWithDis-groundtruth)",
    )
    plt.plot(
        range(1, kSampleCount + 1),
        TrackWithoutDis[0, :] - TrackTruth[0, :],
        label="error_x(TrackWithoutDis-groundtruth)",
    )
    plt.plot(
        range(1, kSampleCount + 1),
        TrackWithoutDis[1, :] - TrackTruth[1, :],
        label="error_y(TrackWithoutDis-groundtruth)",
    )
    plt.plot(
        range(1, kSampleCount + 1),
        TrackWithoutDis[2, :] - TrackTruth[2, :],
        label="error_z(TrackWithoutDis-groundtruth)",
    )
    plt.plot(
        range(1, kSampleCount + 1),
        TrackWithDisAndFixedHeight[0, :] - TrackTruth[0, :],
        label="error_x(TrackWithDisAndFixedHeight-groundtruth)",
    )
    plt.plot(
        range(1, kSampleCount + 1),
        TrackWithDisAndFixedHeight[1, :] - TrackTruth[1, :],
        label="error_y(TrackWithDisAndFixedHeight-groundtruth)",
    )
    plt.plot(
        range(1, kSampleCount + 1),
        TrackWithDisAndFixedHeight[2, :] - TrackTruth[2, :],
        label="error_z(TrackWithDisAndFixedHeight-groundtruth)",
    )

    plt.legend()


plt.figure()
plt.plot(range(1, kSampleCount + 1), PhiTheta[0, :] * 180 / np.pi, label="Phi")
plt.plot(range(1, kSampleCount + 1), PhiTheta[1, :] * 180 / np.pi, label="Theta")
if "PhiThetaTruth" in locals():
    plt.plot(
        range(1, kSampleCount + 1), PhiThetaTruth[0, :] * 180 / np.pi, label="PhiTruth"
    )
    plt.plot(
        range(1, kSampleCount + 1),
        PhiThetaTruth[1, :] * 180 / np.pi,
        label="ThetaTruth",
    )
plt.legend()

plt.show()
