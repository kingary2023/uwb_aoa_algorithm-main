import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =====================
# 数据提取模块
# =====================
def extract_trajectory_from_log(log_file_path):
    """
    从visual_scatter.log文件中提取轨迹坐标
    返回: (xy_coordinates, xz_coordinates, timestamps)
    """
    xy_points = []
    xz_points = []
    timestamps = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析时间戳
            time_match = re.search(r'\[(.*?)\]', line)
            if time_match:
                timestamps.append(time_match.group(1))
            
            # 解析JSON payload
            payload_match = re.search(r'payload=(\{.*\})', line)
            if payload_match:
                try:
                    data = json.loads(payload_match.group(1))
                    if 'xy' in data and len(data['xy']) == 2:
                        xy_points.append(data['xy'])
                    if 'xz' in data and len(data['xz']) == 2:
                        xz_points.append(data['xz'])
                except json.JSONDecodeError:
                    print(f"JSON解析错误: {payload_match.group(1)}")
                    continue
    
    return np.array(xy_points), np.array(xz_points), timestamps
# =====================
# 可视化模块
# =====================

def plot_xy_trajectory(xy_coords, save_path=None):
    """绘制XY平面轨迹图"""
    plt.figure(figsize=(10, 8))
    
    # 绘制轨迹线
    plt.plot(xy_coords[:, 0], xy_coords[:, 1], 'b-', linewidth=2, alpha=0.7, label='trace')
    
    # 绘制轨迹点（颜色表示时间序列）
    scatter = plt.scatter(xy_coords[:, 0], xy_coords[:, 1], 
                         c=range(len(xy_coords)), 
                         cmap='viridis', s=30, alpha=0.6)
    
    # 标记起点和终点
    plt.scatter(xy_coords[0, 0], xy_coords[0, 1], c='green', s=200, marker='o', label='begin')
    plt.scatter(xy_coords[-1, 0], xy_coords[-1, 1], c='red', s=200, marker='s', label='final')
    
    plt.xlabel('X坐标 (m)')
    plt.ylabel('Y坐标 (m)')
    plt.title('XY平面运动轨迹')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('时间序列')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_xz_trajectory(xz_coords, save_path=None):
    """绘制XZ平面轨迹图"""
    plt.figure(figsize=(10, 6))
    
    plt.plot(xz_coords[:, 0], xz_coords[:, 1], 'r-', linewidth=2, alpha=0.7, label='轨迹')
    scatter = plt.scatter(xz_coords[:, 0], xz_coords[:, 1], 
                         c=range(len(xz_coords)),
                         cmap='plasma', s=30, alpha=0.6)
    
    plt.scatter(xz_coords[0, 0], xz_coords[0, 1], c='green', s=200, marker='o', label='起点')
    plt.scatter(xz_coords[-1, 0], xz_coords[-1, 1], c='red', s=200, marker='s', label='终点')
    
    plt.xlabel('X坐标 (m)')
    plt.ylabel('Z坐标 (m)')
    plt.title('XZ平面运动轨迹')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('时间序列')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_3d_trajectory(xy_coords, xz_coords, save_path=None):
    """绘制3D轨迹图"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 重构3D坐标
    x = xy_coords[:, 0]
    y = xy_coords[:, 1]
    z = xz_coords[:, 1]
    
    # 绘制3D轨迹
    ax.plot(x, y, z, 'b-', linewidth=2, alpha=0.7)
    scatter = ax.scatter(x, y, z, c=range(len(x)), 
                         cmap='viridis', s=30, alpha=0.8)
    
    # 标记起点和终点
    ax.scatter(x[0], y[0], z[0], c='green', s=200, marker='o', label='起点')
    ax.scatter(x[-1], y[-1], z[-1], c='red', s=200, marker='s', label='终点')
    
    ax.set_xlabel('X坐标 (m)')
    ax.set_ylabel('Y坐标 (m)')
    ax.set_zlabel('Z坐标 (m)')
    ax.set_title('3D运动轨迹')
    ax.legend()
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('时间序列')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# =====================
# 数据分析模块
# =====================

def analyze_trajectory(xy_coords, xz_coords):
    """分析轨迹特征"""
    # 计算移动距离
    xy_deltas = np.diff(xy_coords, axis=0)
    step_distances = np.sqrt(np.sum(xy_deltas**2, axis=1))
    total_distance = np.sum(step_distances)
    
    # 高度分析
    z_values = xz_coords[:, 1]
    z_range = np.max(z_values) - np.min(z_values)
    
    print("\n===== 轨迹分析报告 =====")
    print(f"轨迹点数量: {len(xy_coords)}")
    print(f"总移动距离: {total_distance:.2f} 米")
    print(f"高度变化范围: {z_range:.3f} 米")
    print(f"X坐标范围: [{np.min(xy_coords[:, 0]):.2f}, {np.max(xy_coords[:, 0]):.2f}]")
    print(f"Y坐标范围: [{np.min(xy_coords[:, 1]):.2f}, {np.max(xy_coords[:, 1]):.2f}]")
    print(f"Z坐标范围: [{np.min(z_values):.2f}, {np.max(z_values):.2f}]")
    print("======================")

def save_trajectory_data(xy_coords, xz_coords, timestamps):
    """保存提取的轨迹数据"""
    # 保存为CSV
    np.savetxt('xy_trajectory.csv', xy_coords, 
               delimiter=',', header='x,y', comments='',
               fmt='%.6f')
    # np.savetxt('xz_trajectory.csv', xz_coords, 
    #            delimiter=',', header='x,z', comments='',
    #            fmt='%.6f')
    
    # 保存为NPZ（二进制格式）
    # np.savez('trajectory_data.npz', 
    #          xy=xy_coords, xz=xz_coords, timestamps=timestamps)
    
    print("\n数据已保存为:")
    print("- xy_trajectory.csv")
    # print("- xz_trajectory.csv")
    # print("- trajectory_data.npz")

# =====================
# 主程序
# =====================

if __name__ == "__main__":
    # 1. 从日志文件提取数据
    log_file =  r"C:\Users\24847\Downloads\uwb_aoa_algorithm-main\uwb_aoa_algorithm-main\data\0905\scatter.log"
    xy, xz, timestamps = extract_trajectory_from_log(log_file )

    
    # 2. 数据验证
    print(f"成功提取 {len(xy)} 个轨迹点")
    print("前5个XY坐标示例:")
    print(xy[:5])
    print("\n前5个XZ坐标示例:")
    print(xz[:5])
    
    # 3. 轨迹分析
    analyze_trajectory(xy, xz)
    
    # 4. 可视化
    plot_xy_trajectory(xy, "xy_trajectory.png")
    # plot_xz_trajectory(xz, "xz_trajectory.png")
    # plot_3d_trajectory(xy, xz, "3d_trajectory.png")
    
    # 5. 保存数据
    save_trajectory_data(xy, xz, timestamps)