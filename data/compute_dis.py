import re
import json
import csv
from datetime import datetime

def extract_distance(log_line: str) -> dict:
    """从单行日志提取距离数据"""
    try:
        # 提取时间戳和payload
        time_str = re.search(r'\[(.*?)\]', log_line).group(1)
        payload = re.search(r'payload=(\{.*?\})', log_line).group(1)
        data = json.loads(payload)
        
        return {
            'time': time_str,
            'dis': data['dis'],
            'dis_est': data['dis_est']
        }
    except Exception as e:
        print(f"解析失败: {log_line[:50]}... | 错误: {str(e)}")
        return None

def save_distances_to_csv(log_path: str, output_csv: str):
    """处理日志文件并保存距离数据到CSV"""
    distances = []
    
    with open(log_path, 'r') as f:
        for line in f:
            if 'payload=' not in line:
                continue
                
            distance_data = extract_distance(line)
            if distance_data:
                distances.append(distance_data)
    
    # 保存CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dis'])  # 表头
        
        for data in distances:
            writer.writerow([
               
                data['dis'],
               
            ])
    
    print(f"成功保存 {len(distances)} 条数据到 {output_csv}")

# 使用示例
if __name__ == "__main__":
    save_distances_to_csv(
        log_path=r"C:\Users\24847\Downloads\uwb_aoa_algorithm-main\uwb_aoa_algorithm-main\data\0905\dis_0905_200.log",  # 替换为您的日志文件路径
        output_csv="0905_200_distance_data.csv"
    )