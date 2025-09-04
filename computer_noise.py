import numpy as np
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import json

anchorZ = 3
AnchorPos = np.array([0, 0, anchorZ])  #基站坐标

def load_real_data(file_path):
    # 方法1：手动处理JSON列（100%兼容您的格式）
    with open(file_path, 'r') as f:
        # 读取所有行
        lines = f.readlines()
        
        # 处理表头
        header = lines[0].strip().split(',')
        
        # 处理数据行
        data = []
        for line in lines[1:]:
            # 分割前4个常规字段
            parts = line.strip().split(',', 4)  # 最多分割4次
            if len(parts) == 5:
                # 解析JSON（兼容单/双引号）
                phase_diff = json.loads(parts[4].replace("'", '"'))
                data.append([
                    parts[0],       # timestamp
                    float(parts[1]), # distance(m)
                    float(parts[2]), # azimuth(deg)
                    float(parts[3]), # elevation(deg)
                    phase_diff      # phase_diff (已解析为dict)
                ])
    
    # 转换为DataFrame
    df = pd.DataFrame(data, columns=header)
    
    # 获取所有可能的相位差键（如["1-0", "2-1", "3-2", "0-3"]）
    all_keys = set()
    for x in df['phase_diff']:
        all_keys.update(x.keys())
    all_keys = sorted(all_keys)  # 固定顺序
    
    # 将字典转换为固定长度的数组
    phase_diff_array = np.zeros((len(all_keys), len(df)))
    for i, x in enumerate(df['phase_diff']):
        for j, key in enumerate(all_keys):
            phase_diff_array[j, i] = x.get(key, 0.0)  # 不存在则填0
    
    return (pd.to_datetime(df['timestamp']).values,
            df['distance(m)'].values,
            df['azimuth(deg)'].values,
            df['elevation(deg)'].values,
            phase_diff_array)

def prepare_real_data(file_path, AnchorPos):
    # 加载数据
    timestamps, distances, azimuths, elevations, phase_diffs = load_real_data(file_path)
    
    # 计算真实轨迹（极坐标→笛卡尔坐标）
    TrackTruth = np.zeros((3, len(distances)))
    TrackTruth[0, :] = distances * np.cos(np.deg2rad(elevations)) * np.cos(np.deg2rad(azimuths)) + AnchorPos[0]
    TrackTruth[1, :] = distances * np.cos(np.deg2rad(elevations)) * np.sin(np.deg2rad(azimuths)) + AnchorPos[1]
    TrackTruth[2, :] = distances * np.sin(np.deg2rad(elevations)) + AnchorPos[2]
    
    # 相位差矩阵 (N_samples × N_antenna_pairs)
    PhiDiff = phase_diffs.T  # 转置为(kAntennaCount, kSampleCount)
    
    return  timestamps, len(distances), TrackTruth, PhiDiff, distances


timestamps, kSampleCount, TrackTruth, PhiDiff, Dis = prepare_real_data("uwb_data_819_noise_410.csv", AnchorPos)

def estimate_noise(timestamps, distances, phase_diffs, static_duration=10):
    """从真实数据中估计噪声参数
    Args:
        timestamps: 时间戳数组
        distances: 距离测量数组
        phase_diffs: 相位差矩阵 (n_antenna_pairs × n_samples)
        static_duration: 视为静态数据的持续时间(秒)
    Returns:
        phi_std: 各天线对相位差噪声标准差 (弧度)
        dis_std: 距离噪声标准差 (米)
    """
    # 转换时间戳为相对秒
    t_sec = (timestamps - timestamps[0]).astype('timedelta64[s]').astype(float)
    
    # 选取静态阶段（前N秒）
    mask = t_sec < static_duration
    static_phase = phase_diffs[:, mask]  # (n_antenna_pairs, n_static_samples)
    static_dis = distances[mask]
    
    # 计算噪声标准差
    phi_std = np.std(static_phase, axis=1)  # 各天线对单独计算
    dis_std = np.std(static_dis)
    
    return phi_std, dis_std


phi_std, dis_std = estimate_noise(
    timestamps,
    distances=Dis,
    phase_diffs=PhiDiff.T,  # 转置为(n_antenna_pairs × n_samples)
    static_duration=10
)

print(f"相位差噪声标准差 (各天线对): {phi_std} rad")
print(f"距离噪声标准差: {dis_std:.3f} m")