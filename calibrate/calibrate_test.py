import numpy as np
from scipy import interpolate
from scipy.spatial import KDTree

class PhaseCalibrator:
    def __init__(self, calibration_table_path='calibration_table_4channels_with_azimuth.csv'):
        # 加载校准表
        self.calibration_table = np.loadtxt(calibration_table_path, delimiter=',', skiprows=1)
        
        # 提取数据（根据您的格式）
        self.raw_phases = self.calibration_table[:, 0:4]      # 第0-3列: Raw_Ch0-3
        self.calibrated_phases = self.calibration_table[:, 4:8] # 第4-7列: Calibrated_Ch0-3
        self.azimuths = self.calibration_table[:, 8]          # 第8列: 方位角
        
        # 构建KDTree用于快速最近邻搜索（基于方位角和原始相位）
        features = np.column_stack([self.azimuths, self.raw_phases])
        self.kdtree = KDTree(features)
        
        # 为每个通道创建插值函数（可选，用于快速单通道校准）
        self.interpolators = []
        for channel in range(4):
            # 获取该通道的原始值和校准值
            raw_values = self.calibration_table[:, channel]      # Raw_Ch0-3
            calib_values = self.calibration_table[:, 4 + channel] # Calibrated_Ch0-3
            
            # 创建线性插值函数
            interp_func = interpolate.interp1d(raw_values, calib_values, 
                                              kind='linear', bounds_error=False, fill_value='extrapolate')
            self.interpolators.append(interp_func)
            
            # 打印通道信息
            rmse = np.sqrt(np.mean((raw_values - calib_values)**2))
            # print(f"通道 {channel}: RMSE = {rmse:.4f}°")
    
    def calibrate_with_azimuth(self, azimuth, raw_phases, method='nearest'):
        if method == 'nearest':
            # 使用KDTree找到最接近的校准点
            query_point = np.array([azimuth, *raw_phases])
            distance, idx = self.kdtree.query(query_point, k=1)
            
            # 返回对应的校准值
            return self.calibrated_phases[idx]
        
        elif method == 'interpolate':
            # 使用插值方法（可能不如最近邻准确，但可以作为备选）
            calibrated_phases = []
            for i in range(4):
                calibrated = self.interpolators[i](raw_phases[i])
                calibrated_phases.append(calibrated)
            return np.array(calibrated_phases)
        
        else:
            raise ValueError("method must be 'nearest' or 'interpolate'")
    
    def find_nearest_calibration_point(self, azimuth, raw_phases):
        query_point = np.array([azimuth, *raw_phases])
        distance, idx = self.kdtree.query(query_point, k=1)
        
        return {
            'index': idx,
            'distance': distance,
            'azimuth_match': self.azimuths[idx],
            'raw_phases_match': self.raw_phases[idx],
            'calibrated_phases_match': self.calibrated_phases[idx]
        }

# 直接处理你的数据格式（现在需要包含方位角）
def calibrate_measurement_data_with_azimuth(z, azimuth):
  
    # 初始化校准器（建议作为全局变量只初始化一次）
    if not hasattr(calibrate_measurement_data_with_azimuth, 'calibrator'):
        calibrate_measurement_data_with_azimuth.calibrator = PhaseCalibrator()
    
    calibrator = calibrate_measurement_data_with_azimuth.calibrator
    
    # 提取相位差部分进行校准
    raw_phases = z[:4]
    calibrated_phases = calibrator.calibrate_with_azimuth(azimuth, raw_phases, method='nearest')
    
    # 组合结果（保持距离不变）
    return np.concatenate([calibrated_phases, [z[4]]])

# 使用示例
if __name__ == "__main__":
    # 你的数据格式（现在需要额外提供方位角）
    z = np.array([12,171,-17,3.1,-167])  # [相位差0, 相位差1, 相位差2, 相位差3, 距离]
    azimuth = -86  # 当前测量的方位角
    
    # 进行校准
    calibrator = PhaseCalibrator()
    calibrated_z_nearest = calibrator.calibrate_with_azimuth(azimuth, z[:4], method='nearest')
    
    # 方法2: 使用插值校准（备选）
    calibrated_z_interp = calibrator.calibrate_with_azimuth(azimuth, z[:4], method='interpolate')
    
    # 查找匹配的校准点信息（用于调试）
    match_info = calibrator.find_nearest_calibration_point(azimuth, z[:4])
    
    print("原始数据:", z[:4])
    print("方位角:", azimuth)
    print("最近邻校准结果:", calibrated_z_nearest)
    print("插值校准结果:", calibrated_z_interp)
    print("\n匹配的校准点信息:")
    print(f"  索引: {match_info['index']}")
    print(f"  距离: {match_info['distance']:.4f}")
    print(f"  匹配方位角: {match_info['azimuth_match']:.2f}°")
    print(f"  匹配原始相位: {match_info['raw_phases_match']}")
    print(f"  匹配校准相位: {match_info['calibrated_phases_match']}")
    
    # 使用便捷函数
    calibrated_z_full = calibrate_measurement_data_with_azimuth(z, azimuth)
    print("\n完整校准结果:", calibrated_z_full)