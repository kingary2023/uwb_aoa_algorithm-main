import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_real_data(file_path):
    # Method 1: Manual JSON column processing (100% compatible with your format)
    with open(file_path, 'r') as f:
        # Read all lines
        lines = f.readlines()
        
        # Process header
        header = lines[0].strip().split(',')
        
        # Process data rows
        data = []
        for line in lines[1:]:
            # Split first 4 regular fields
            parts = line.strip().split(',', 4)  # Split at most 4 times
            if len(parts) == 5:
                # Parse JSON (compatible with single/double quotes)
                phase_diff = json.loads(parts[4].replace("'", '"'))
                data.append([
                    parts[0],       # timestamp
                    float(parts[1]), # distance(m)
                    float(parts[2]), # azimuth(deg)
                    float(parts[3]), # elevation(deg)
                    phase_diff      # phase_diff (parsed as dict)
                ])
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=header)
    
    # Get all possible phase difference keys (e.g., ["1-0", "2-1", "3-2", "0-3"])
    all_keys = set()
    for x in df['phase_diff']:
        all_keys.update(x.keys())
    all_keys = sorted(all_keys)  # Fixed order
    
    # Convert dictionary to fixed-length array
    phase_diff_array = np.zeros((len(all_keys), len(df)))
    for i, x in enumerate(df['phase_diff']):
        for j, key in enumerate(all_keys):
            phase_diff_array[j, i] = x.get(key, 0.0)  # Fill with 0 if not exists
    
    return (pd.to_datetime(df['timestamp']).values,
            df['distance(m)'].values,
            df['azimuth(deg)'].values,
            df['elevation(deg)'].values,
            phase_diff_array)

def prepare_real_data(file_path, AnchorPos):
    # Load data
    timestamps, distances, azimuths, elevations, phase_diffs = load_real_data(file_path)
    
    # Calculate true trajectory (polar → Cartesian coordinates)
    TrackTruth = np.zeros((3, len(distances)))
    TrackTruth[0, :] = distances * np.cos(np.deg2rad(elevations)) * np.cos(np.deg2rad(azimuths)) + AnchorPos[0]
    TrackTruth[1, :] = -distances * np.cos(np.deg2rad(elevations)) * np.sin(np.deg2rad(azimuths)) + AnchorPos[1]
    TrackTruth[2, :] = -distances * np.sin(np.deg2rad(elevations)) + AnchorPos[2]
    
    # Phase difference matrix (N_samples × N_antenna_pairs)
    PhiDiff = phase_diffs # Transpose to (kAntennaCount, kSampleCount)
    
    return len(distances), TrackTruth,PhiDiff, distances,azimuths

AnchorPos = np.array([0, 0, 1])  # Anchor coordinates

kSampleCount, TrackTruth, PhiDiff, Dis, azimuths = prepare_real_data("uwb_data_826.csv", AnchorPos)
PhiDiff_deg = PhiDiff.T
PhiDiff = PhiDiff_deg * np.pi / 180
print(PhiDiff.shape)
print(azimuths.shape)


phase_data = PhiDiff_deg[:, 0]  # First column, phase difference data for first antenna pair (degrees)
# print(phase_data.shape)
# Generate angle sequence (assuming uniform circular motion)
angles = np.linspace(0, 360, len(phase_data))  # 0° to 360°


# 生成示例数据（4个通道，3圈数据）
n_circles = 2
n_points_per_circle = 834
n_points_total = 1669
n_channels = 4

# 生成时间索引
t = np.linspace(0, n_circles * 2 * np.pi, n_points_total)

# 为每个通道定义不同的正弦参数
channel_params = [
    {'A': 10, 'omega': 1, 'phi': 0.5, 'C': 5},    # 通道0
    {'A': 8,  'omega': 1, 'phi': 1.2, 'C': 3},    # 通道1
    {'A': 8, 'omega': 1, 'phi': 1.2, 'C': 3},    # 通道2
    {'A': 9,  'omega': 1, 'phi': 0.3, 'C': 4}     # 通道3
]


# 定义正弦拟合函数
def sin_func(t, A, omega, phi, C):
    return A * np.sin(omega * t + phi) + C

# 为每个通道进行拟合
index = np.arange(n_points_total)
fit_results = []
fit_curves = np.zeros_like(PhiDiff_deg)
rmse_values = []

print("开始四个通道的正弦拟合...")
print("=" * 60)

for channel in range(n_channels):
    phase_data = PhiDiff_deg[:, channel]

    
    # 初始参数猜测
    A_guess = (np.max(phase_data) - np.min(phase_data)) / 2
    C_guess = np.mean(phase_data)
    omega_guess = (n_circles * 2 * np.pi) / n_points_total
    phi_guess = 0
    p0 = [A_guess, omega_guess, phi_guess, C_guess]

    if channel==2:
        print(p0)
    
    try:
        # 执行拟合
        popt, pcov = curve_fit(sin_func, index, phase_data, p0=p0, maxfev=10000)
        A_fit, omega_fit, phi_fit, C_fit = popt
        
        # 计算拟合曲线和误差
        fit_curve = sin_func(index, A_fit, omega_fit, phi_fit, C_fit)
        fit_curves[:, channel] = fit_curve
        residuals = phase_data - fit_curve
        rmse = np.sqrt(np.mean(residuals**2))
        
        fit_results.append({
            'channel': channel,
            'A': A_fit,
            'omega': omega_fit,
            'phi': phi_fit,
            'C': C_fit,
            'rmse': rmse
        })
        
        rmse_values.append(rmse)
        
        print(f"通道 {channel} 拟合成功:")
        print(f"  A={A_fit:.4f}, ω={omega_fit:.6f}, φ={phi_fit:.4f}, C={C_fit:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  估计周期: {2 * np.pi / omega_fit:.2f} 点/圈")
        print("-" * 40)
        
    except Exception as e:
        print(f"通道 {channel} 拟合失败: {e}")
        # 使用初始猜测
        fit_curves[:, channel] = sin_func(index, *p0)
        fit_results.append({
            'channel': channel,
            'A': p0[0],
            'omega': p0[1],
            'phi': p0[2],
            'C': p0[3],
            'rmse': np.nan
        })

# 生成校准表（使用最后一圈数据）
last_circle_start = n_points_total-834
calibration_data = []
print(last_circle_start,n_points_total)
# 为每个采样点创建校准表条目
for i in range(last_circle_start, n_points_total):
    entry = []
    
    # 原始四个通道的值
    for channel in range(n_channels):
        entry.append(PhiDiff_deg[i, channel])
    # 对应的拟合值（理想值）
    for channel in range(n_channels):
        entry.append(fit_curves[i, channel])

    entry.append(azimuths[i])
    calibration_data.append(entry)



calibration_table = np.array(calibration_data)

# 保存校准表（CSV格式）
header = (
    'Raw_Channel0,Raw_Channel1,Raw_Channel2,Raw_Channel3,'
    'Calibrated_Channel0,Calibrated_Channel1,Calibrated_Channel2,Calibrated_Channel3,'
     'Azimuth(deg)'

)

np.savetxt('calibration_table_4channels_with_azimuth.csv', calibration_table, 
           delimiter=',', 
           header=header, 
           comments='',
           fmt='%.6f')

print(f"\n校准表已保存为 'calibration_table_4channels_.csv'")
print(f"包含 {len(calibration_table)} 个数据点")
print(f"文件包含：4个原始通道值 + 4个校准后的理想值")

# 可视化结果
plt.figure(figsize=(15, 8))

# 为每个通道创建子图
for channel in range(n_channels):
    plt.subplot(2, 2, channel + 1)
    
    # 原始数据
    plt.plot(index, PhiDiff_deg[:, channel], 'b-', alpha=0.6, 
             label=f'raw Ch{channel}', linewidth=1)
    
    # 拟合曲线
    plt.plot(index, fit_curves[:, channel], 'r-', 
             label=f'new Ch{channel}', linewidth=2)
    
    plt.xlabel('num')
    plt.ylabel('phi (°)')
    plt.title(f' {channel} - RMSE: {rmse_values[channel]:.4f}' if channel < len(rmse_values) else f' {channel}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.suptitle('result', fontsize=16)
plt.subplots_adjust(top=0.92)
plt.show()

# 显示校准表统计信息
print("\n校准表统计信息:")
print("=" * 50)
for channel in range(n_channels):
    raw_data = calibration_table[:, channel]
    calib_data = calibration_table[:, 4 + channel]
    error = raw_data - calib_data
    
    print(f"通道 {channel}:")
    print(f"  原始值范围: [{raw_data.min():.3f}, {raw_data.max():.3f}]")
    print(f"  校准值范围: [{calib_data.min():.3f}, {calib_data.max():.3f}]")
    print(f"  平均误差: {np.mean(error):.4f}°, 标准差: {np.std(error):.4f}°")
    print("-" * 30)

# 显示校准表前5行
print("\n校准表前5行:")
print("Raw0     Raw1     Raw2     Raw3     Calib0    Calib1    Calib2    Calib3")
print("=" * 80)
for i in range(min(5, len(calibration_table))):
    row = calibration_table[i]
    print(f"{row[0]:6.2f}  {row[1]:6.2f}  {row[2]:6.2f}  {row[3]:6.2f}  "
          f"{row[4]:7.2f}  {row[5]:7.2f}  {row[6]:7.2f}  {row[7]:7.2f}")