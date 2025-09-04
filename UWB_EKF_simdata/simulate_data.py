import numpy as np


def simulate_data(Nab, _lambda, AnchorPos, tagZ, disStd, phiStd):
    # 根据输入参数返回指定轨迹对应的三对天线相位差测量值以及测距值

    dataSource = 7





    # 最大半径
    r_max = 6

    kAntennaCount = Nab.shape[1]  #3

    # 使用 match-case 语句
    match dataSource:
        case 0:  # 轨迹1 圆
            t = np.arange(0, 40, 40 / 370)
            # 计算需要多少点数
            kSampleCount = t.shape[0]   
            TrackTruth = np.zeros((3, kSampleCount))
            TrackTruth[0, :] = r_max * np.cos(1/20 * np.pi * t) + AnchorPos[0]
            TrackTruth[1, :] = r_max * np.sin(1/20 * np.pi * t) + AnchorPos[1]
            TrackTruth[2, :] = tagZ
        case 1:  # 轨迹2 螺旋充满整个平面
            n = 200  # 一圈分成n个点
            cycles = 5  # 轨迹螺旋圈数
            dr = r_max / cycles  # 转一圈半径增加dr

            t = np.arange(0, cycles, 1 /n )
            r = dr * t
            # 计算需要多少点数
            kSampleCount = t.shape[0]
            TrackTruth = np.zeros((3, kSampleCount))
            TrackTruth[0, :] = r * np.cos(2 * np.pi * t) + AnchorPos[0]
            TrackTruth[1, :] = r * np.sin(2 * np.pi * t) + AnchorPos[1]
            TrackTruth[2, :] = tagZ
        case 2:  # 轨迹1 圆
            t = np.arange(0, 1, 1 / 300)
            # 计算需要多少点数
            kSampleCount = t.shape[0]   
            TrackTruth = np.zeros((3, kSampleCount))
            TrackTruth[0, :] = r_max * np.cos(2 * np.pi * t) + AnchorPos[0]
            TrackTruth[1, :] = r_max * np.sin(2 * np.pi * t) + AnchorPos[1]
            TrackTruth[2, :] = -10*t + tagZ
        case 3:  # 匀速直线（斜向）
            kSampleCount = 300
            t = np.linspace(0, 10, kSampleCount)  # 10秒内匀速运动
            TrackTruth = np.zeros((3, kSampleCount))
            TrackTruth[0, :] = 2 * t + AnchorPos[0]  # x方向速度0.5m/s
            TrackTruth[1, :] = 2 * t + AnchorPos[1]  # y方向速度0.3m/s
            TrackTruth[2, :] = tagZ  # 高度恒定
        case 4:  # 匀加速（抛物线）
            kSampleCount = 300
            t = np.linspace(0, 5, kSampleCount)  # 5秒内加速
            TrackTruth = np.zeros((3, kSampleCount))
            TrackTruth[0, :] = 0.1 * t**2 + AnchorPos[0]  # x方向加速度0.2m/s²
            TrackTruth[1, :] = 0.05 * t**2 + AnchorPos[1] # y方向加速度0.1m/s²
            TrackTruth[2, :] = tagZ
        case 5:  # 正弦蛇形
            kSampleCount = 600
            t = np.linspace(0, 2*np.pi, kSampleCount)
            TrackTruth = np.zeros((3, kSampleCount))
            TrackTruth[0, :] = r_max * np.cos(t) + AnchorPos[0]
            TrackTruth[1, :] = r_max * np.sin(2*t) + AnchorPos[1]  # y方向双频正弦
            TrackTruth[2, :] = tagZ - 10 * t  # 
        case 6:  # 锥形螺旋（半径缩小+高度上升）
            kSampleCount = 600
            t = np.linspace(0, 4*np.pi, kSampleCount)  # 两圈
            TrackTruth = np.zeros((3, kSampleCount))
            TrackTruth[0, :] = (r_max - 0.1*t) * np.cos(t) + AnchorPos[0]  # 半径递减
            TrackTruth[1, :] = (r_max - 0.1*t) * np.sin(t) + AnchorPos[1]
            TrackTruth[2, :] = tagZ - 0.08 * t  # 高度线性上升
        case 7:  # 三维Lissajous图形
            kSampleCount = 1000
            t = np.linspace(0, np.pi, kSampleCount)
            TrackTruth = np.zeros((3, kSampleCount))
            TrackTruth[0, :] = r_max * np.sin(3*t + np.pi/4) + AnchorPos[0]  # x: 3倍频
            TrackTruth[1, :] = r_max * np.sin(2*t) + AnchorPos[1]             # y: 2倍频
            TrackTruth[2, :] = tagZ- 10 *t                     # z: 基础正弦
        case 8:  # 组合机动（直线→急转→加速）
            kSampleCount = 400
            t = np.linspace(0, 10, kSampleCount)
            TrackTruth = np.zeros((3, kSampleCount))
            # 阶段1: 匀速直线
            TrackTruth[0, :200] = 0.5 * t[:200] + AnchorPos[0]
            TrackTruth[1, :200] = 0.3 * t[:200] + AnchorPos[1]
            # 阶段2: 90度急转
            TrackTruth[0, 200:300] = TrackTruth[0, 199] + 0.3 * (t[200:300]-t[200]) * np.cos(np.pi/2 * (t[200:300]-t[200])/1.0)
            TrackTruth[1, 200:300] = TrackTruth[1, 199] + 0.3 * (t[200:300]-t[200]) * np.sin(np.pi/2 * (t[200:300]-t[200])/1.0)
            # 阶段3: 匀加速
            TrackTruth[0, 300:] = TrackTruth[0, 299] + 0.5*(t[300:]-t[300]) + 0.1*(t[300:]-t[300])**2
            TrackTruth[1, 300:] = TrackTruth[1, 299] + 0.2*(t[300:]-t[300]) 
            TrackTruth[2, :] = tagZ
        case 9:  # 高度阶跃（测试滤波器瞬态响应）
            kSampleCount = 300
            TrackTruth = np.zeros((3, kSampleCount))
            TrackTruth[0, :] = AnchorPos[0] + 0.1 * np.arange(kSampleCount)  # x匀速
            TrackTruth[1, :] = AnchorPos[1]  # y静止
            TrackTruth[2, :100] = tagZ        # 阶段1: 高度=tagZ
            TrackTruth[2, 100:200] = tagZ + 2.0  # 阶段2: 突然升高2m
            TrackTruth[2, 200:] = tagZ - 1.0     # 阶段3: 下降1m
        case 11:  # 多段直线轨迹
            kSampleCount = 500  # 总点数
            t = np.linspace(0, 10, kSampleCount)  # 时间轴0~10秒
            TrackTruth = np.zeros((3, kSampleCount))
            
            # 分段参数
            t1 = int(kSampleCount * 0.3)  # 0~3秒：匀速
            t2 = int(kSampleCount * 0.6)  # 3~6秒：加速
            t3 = int(kSampleCount * 0.8)  # 6~8秒：静止
            t4 = kSampleCount             # 8~10秒：折返
            
            # 阶段1：匀速直线（东北方向）
            TrackTruth[0, :t1] = 1.0 * t[:t1] + AnchorPos[0]  # x速度1.0m/s
            TrackTruth[1, :t1] = 0.5 * t[:t1] + AnchorPos[1]  # y速度0.5m/s
            
            # 阶段2：加速运动（东偏北方向）
            TrackTruth[0, t1:t2] = TrackTruth[0, t1-1] + 1.0*(t[t1:t2]-t[t1]) + 0.3*(t[t1:t2]-t[t1])**2  # x加速度0.6m/s²
            TrackTruth[1, t1:t2] = TrackTruth[1, t1-1] + 0.5*(t[t1:t2]-t[t1]) + 0.1*(t[t1:t2]-t[t1])**2  # y加速度0.2m/s²
            
            # 阶段3：静止停顿
            TrackTruth[0, t2:t3] = TrackTruth[0, t2-1]  # x位置不变
            TrackTruth[1, t2:t3] = TrackTruth[1, t2-1]  # y位置不变
            
            # 阶段4：折返运动（西南方向）
            TrackTruth[0, t3:t4] = TrackTruth[0, t3-1] - 1.2 * (t[t3:t4]-t[t3])  # x速度-1.2m/s
            TrackTruth[1, t3:t4] = TrackTruth[1, t3-1] - 0.8 * (t[t3:t4]-t[t3])  # y速度-0.8m/s
            
            # 高度恒定
            TrackTruth[2, :] = tagZ
        case _:  # 默认情况
            print("in simulate_data, dataSource is invalid! Check and retry.")
            return None, None, None, None, None

    # 根据轨迹生成到基站的3路角度及距离数据
    Dis = np.sqrt(
        (TrackTruth[0, :] - AnchorPos[0]) ** 2
        + (TrackTruth[1, :] - AnchorPos[1]) ** 2
        + (TrackTruth[2, :] - AnchorPos[2]) ** 2
    ) + np.random.normal(0, disStd, kSampleCount)

    PhiDiff = np.full((kAntennaCount, kSampleCount), np.nan)
    PhiThetaTruth = np.full((2, kSampleCount), np.nan)

    for index in range(kSampleCount):##300个
        vTO = np.array(
            [
                AnchorPos[0] - TrackTruth[0, index],
                AnchorPos[1] - TrackTruth[1, index],
                AnchorPos[2] - TrackTruth[2, index],
            ]
        )

        for indexj in range(kAntennaCount):##3个
            vNab = np.array([Nab[0, indexj], Nab[1, indexj], 0]) ##天线
            deltaR = np.dot(vTO, vNab) / np.linalg.norm(vTO)
            PhiDiff[indexj, index] = 2 * np.pi * deltaR / _lambda

        # 需要以基站为原点，计算目标位置相对基站的位置
        x = TrackTruth[0, index] - AnchorPos[0]
        y = TrackTruth[1, index] - AnchorPos[1]
        z = TrackTruth[2, index] - AnchorPos[2]
        PhiThetaTruth[0, index] = np.arctan2(y, x)
        PhiThetaTruth[1, index] = np.arctan2(np.sqrt(x**2 + y**2), z)

    PhiDiff = PhiDiff + np.random.normal(0, phiStd, (kAntennaCount, kSampleCount))

    return kSampleCount, TrackTruth, PhiThetaTruth, PhiDiff, Dis
