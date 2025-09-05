import re
import json
import csv
from datetime import datetime

def extract_phases(log_line: str) -> dict:
    """从单行日志提取相位差"""
    try:
        payload = re.search(r'payload=(\{.*?\})', log_line).group(1)
        data = json.loads(payload)
        return {
            'time': re.search(r'\[(.*?)\]', log_line).group(1),
            '1-0': data.get('1-0', 0),
            '2-1': data.get('2-1', 0),
            '3-2': data.get('3-2', 0),
            '0-3': data.get('0-3', 0)
        }
    except Exception as e:
        print(f"解析失败: {log_line[:50]}... | 错误: {str(e)}")
        return None

def save_phases_to_csv(log_path: str, output_csv: str):
    """处理日志文件并保存相位差到CSV"""
    phases_data = []
    
    with open(log_path, 'r') as f:
        for line in f:
            if 'payload=' not in line:
                continue
                
            phase_data = extract_phases(line)
            if phase_data:
                phases_data.append(phase_data)
    
    # 计算时间差dt
    for i in range(len(phases_data)):
        if i == 0:
            phases_data[i]['dt'] = 0.0
        else:
            t1 = datetime.strptime(phases_data[i-1]['time'], "%Y-%m-%d %H:%M:%S.%f")
            t2 = datetime.strptime(phases_data[i]['time'], "%Y-%m-%d %H:%M:%S.%f")
            phases_data[i]['dt'] = (t2 - t1).total_seconds()
    
    # 保存CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dt', 'phase_1-0', 'phase_2-1', 'phase_3-2', 'phase_0-3'])
        
        for data in phases_data:
            writer.writerow([
                data['dt'],
                data['1-0'],
                data['2-1'],
                data['3-2'],
                data['0-3']
            ])
    
    print(f"成功保存 {len(phases_data)} 条数据到 {output_csv}")

# 使用示例
if __name__ == "__main__":
    save_phases_to_csv(
        log_path= r"C:\Users\24847\Downloads\uwb_aoa_algorithm-main\uwb_aoa_algorithm-main\data\0905\phi_cali_0905_200.log",  # 替换为您的日志文件路径
        output_csv="0905_200_phase_cali_data.csv"
    )