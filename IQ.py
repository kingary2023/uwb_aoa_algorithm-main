import re
import serial
import math
import json
import numpy as np
from datetime import datetime

# 串口配置
COM_PORT = "COM4"
BAUD_RATE = 921600

# 数据存储文件（CSV格式）
DATA_FILE = "uwb_data_826.csv"



# 初始化串口
COM = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)

# 初始化数据存储（CSV文件头）
with open(DATA_FILE, "a") as f:
    f.write("timestamp,distance(m),azimuth(deg),elevation(deg),phase_diff\n")

ANT_IQ = []
phase_diffs = {}

# 正则表达式匹配模式
pattern1 = r"\[APP\]\[INFO\]ANT IQ:\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)"
pattern2 = r"FiRa DS-TWR Deferred Controlee SEQ NUM"
pattern3 = r"\[APP\]\[INFO\]Peer AAA1, Distance (\d+)cm, PDoA Azimuth ([-+]?\d+) Elevation ([-+]?\d+) Azimuth FoM"

while COM.is_open:
    data = COM.readline().decode("utf-8", errors="ignore")
    match1 = re.findall(pattern1, data)
    match2 = re.findall(pattern2, data)
    match3 = re.findall(pattern3, data)

    # 处理 ANT IQ 数据（计算相位差）
    if match1:
        for i, (i_real, i_imag) in enumerate(match1):
            ANT_IQ.append(complex(float(i_real), float(i_imag)))

    # 触发相位差计算
    if match2:
        if len(ANT_IQ) < 4:
            print("Warning: ANT_IQ data incomplete, skipping...")
            ANT_IQ.clear()
            continue

        # 计算相邻天线的相位差
        for i in range(len(ANT_IQ)):
            if i == 3:
                s1 = ANT_IQ[i]
                s2 = ANT_IQ[0]
                R = np.conj(s1) * s2
                phase_diffs[f"{0}-{i}"] = np.angle(R) * 180 / np.pi
            else:
                s1 = ANT_IQ[i]
                s2 = ANT_IQ[i + 1]
                R = np.conj(s1) * s2
                phase_diffs[f"{i+1}-{i}"] = np.angle(R) * 180 / np.pi

        # 打印相位差（调试用）
        print("Phase Differences:", phase_diffs)

        # 清空缓存
        ANT_IQ.clear()

    # 处理距离和角度数据
    if match3:
        distance = float(match3[0][0]) / 100  # 转换为米
        azimuth = float(match3[0][1])         # 方位角（度）
        elevation = float(match3[0][2])       # 仰角（度）

        # 打印数据（调试用）
        print(f"Distance: {distance:.2f}m, Azimuth: {azimuth}°, Elevation: {elevation}°")

        # 存储到CSV文件
        with open(DATA_FILE, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp},{distance:.2f},{azimuth},{elevation},{json.dumps(phase_diffs)}\n")