import re
import time
import serial
import numpy as np
import json
import paho.mqtt.client as mqtt
from scipy.linalg import block_diag

# ------------------ 硬件配置 ------------------
COM_PORT = "COM4"
BAUD_RATE = 921600
COM = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)

# ------------------ MQTT配置 ------------------

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("connected successfully.")
    else:
        print(f"connection failed with code {rc}.")


def on_disconnect(client, userdata, rc):
    if rc == 0:
        print("disconnected successfully.")
    else:
        print(f"unexpected disconnection with code {rc}.")

client = mqtt.Client()
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.connect("127.0.0.1", 1883, 60)
client.reconnect_delay_set(min_delay=1, max_delay=4)
client.loop_start()

# ------------------ 系统参数 ------------------
lambda_val = 299792458 / 6.5e9  # 波长
dAntenna = 0.0204  # 天线间距
eps = np.finfo(float).eps
DeltaThetaDeg = [0, 90, 180,270]

disStd = 0.019 * 1.0
phiStd = np.array([12.87, 4.43, 13.79, 6.23]) * np.pi / 180  # 四个天线对的噪声 [rad]
anchorZ = 0.9
AnchorPos = np.array([0, 0, anchorZ])  #基站坐标

# DeltaThetaDeg = [0, 120]

kAntennaPairCount = len(DeltaThetaDeg)

# 天线基线向量（根据实际硬件修改）
Nab = np.zeros((2, kAntennaPairCount))  # 天线向量
Nab[0, 0] = 0
Nab[1, 0] = dAntenna

kAntennaCount = Nab.shape[1]

for col in range(1, kAntennaPairCount):
    theta = DeltaThetaDeg[col] / 180 * np.pi
    Nab[:, col] = np.dot(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], Nab[:, 0]
    )


x_init = np.array([1, -1.5, 2.0, 0, 0, 0])  # [x, y, z, vx, vy, vz]
P_init = np.diag([0.4,0.4 , 0.04 ,0.3,0.3,0.3 ])  # 初始协方差矩1         

# ------------------ IMM-EKF初始化 ------------------
class IMMFilter:
    def __init__(self,sigma_a):

        self.f =self._create_cv_filter()  # 加速度噪声标准差
        # self.f = self._create_cv_filter()
        self.R_phi = np.diag(phiStd ** 2)  # 相位差噪声
        self.R_dis = disStd ** 2                           # 距离噪声
        self.R = np.block([[self.R_phi, np.zeros((kAntennaCount, 1))],
              [np.zeros((1, kAntennaCount)), self.R_dis]])
        
    def _create_cv_filter(self):
        """创建CV模型滤波器（低过程噪声）"""
        Q = np.diag([0.01, 0.08, 0.08, 0.5, 0.5, 0.1])  # 小噪声
        return {'x': x_init.copy(), 'P': P_init.copy(), 'Q': Q}
    def _create_ca_filter(self, sigma_a=0.5):
        """创建CA模型滤波器（高过程噪声）"""
        # CA模型的Q矩阵构造（参见前文公式）
        dt = 1.0 / 10
        q_pos = (dt**4)/4 * sigma_a**2
        q_vel = dt**2 * sigma_a**2
        q_cross = (dt**3)/2 * sigma_a**2
        Q = np.array([
            [q_pos, 0, 0, q_cross, 0, 0],
            [0, q_pos, 0, 0, q_cross, 0],
            [0, 0, q_pos, 0, 0, q_cross],
            [q_cross, 0, 0, q_vel, 0, 0],
            [0, q_cross, 0, 0, q_vel, 0],
            [0, 0, q_cross, 0, 0, q_vel]
        ])
        return {'x': x_init.copy(), 'P': P_init.copy(), 'Q': Q}
    
    def get_F_matrix( self,dt):
        """CV模型基础"""
        return np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
    
    def predict(self, dt):
        """各模型独立预测"""
        F = self.get_F_matrix( dt)  # 状态转移矩阵（CV/CA共用）
        self.f['x'] = F @ self.f['x']
        self.f['P'] = F @ self.f['P'] @ F.T + self.f['Q']
    
    def observation_model(self, x, AnchorPos, Nab, lambda_val):
        pos = x[:3]
        r = np.linalg.norm(pos - AnchorPos)
        
        vTO = AnchorPos - pos
        vTO = (AnchorPos - pos) / r  # 单位方向向量
        PhiDiff_ekf = np.zeros(Nab.shape[1])
        for i in range(Nab.shape[1]):
            vNab = np.array([Nab[0, i], Nab[1, i], 0])  # 天线对向量
            deltaR = np.dot(vTO, vNab)
            PhiDiff_ekf[i] = (2 * np.pi / lambda_val) * deltaR
        
        # 合并观测值
        z = np.concatenate([PhiDiff_ekf, [r]])
        return z
    

    def compute_H(self , x_pred, AnchorPos, Nab, lambda_val, eps=1e-6):
        """数值法计算雅可比矩阵（带防错处理）"""
        H = np.zeros((kAntennaCount + 1, 6))
        h0 = self.observation_model(x_pred, AnchorPos, Nab, lambda_val)
        for i in range(6):
            dx = np.zeros(6)
            dx[i] = eps
            h_eps = self.observation_model(x_pred + dx, AnchorPos, Nab, lambda_val)
            H[:, i] = (h_eps - h0) / eps
        return H
    
    def update(self, z, AnchorPos):
        """更新步"""
     
        # 计算雅可比矩阵H
        H = self.compute_H(self.f['x'], AnchorPos, Nab, lambda_val)
            
        # 计算卡尔曼增益
        S = H @ self.f['P'] @ H.T + self.R
        K = self.f['P'] @ H.T @ np.linalg.inv(S)
            
        # 更新状态和协方差
        z_pred = self.observation_model(self.f['x'], AnchorPos, Nab, lambda_val)
        innovation = z - z_pred
        self.f['x'] = self.f['x'] + K @ innovation
        self.f['P'] = (np.eye(6) - K @ H) @ self.f['P']
        
        return  self.f['x']

# ------------------ 主循环 ------------------
imm = IMMFilter(12.0)
ANT_IQ = []
last_time = None
phase_diffs = {}
pattern1 = r"\[APP\]\[INFO\]ANT IQ:\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)"
pattern2 = r"FiRa DS-TWR Deferred Controlee SEQ NUM"
pattern3 = r"\[APP\]\[INFO\]Peer AAA1, Distance (\d+)cm, PDoA Azimuth ([-+]?\d+) Elevation ([-+]?\d+) Azimuth FoM"

while COM.is_open:
    data = COM.readline().decode("utf-8", errors="ignore")
    
    # 解析数据
    match_iq = re.findall(pattern1, data)
    match_trigger = re.findall(pattern2, data)
    match_pos = re.findall(pattern3, data)
    # 处理I/Q数据
    if match_iq:
        for i_real, i_imag in match_iq:
            ANT_IQ.append(complex(float(i_real), float(i_imag)))
    
    # 触发相位差计算
    if match_trigger:
        if len(ANT_IQ) < 4:
            ANT_IQ.clear()
            continue
        for i in range(len(ANT_IQ)):
            if i == 3:
                s1 = ANT_IQ[i]
                s2 = ANT_IQ[0]
                R = np.conj(s1) * s2
                phase_diffs[f"{0}-{i}"] = np.angle(R)
            else:
                s1 = ANT_IQ[i]
                s2 = ANT_IQ[i + 1]
                R = np.conj(s1) * s2
                phase_diffs[f"{i+1}-{i}"] = np.angle(R)
        
        client.publish("vd/line", json.dumps(phase_diffs))
        ANT_IQ.clear()
    
    # 处理距离和角度数据
    if match_pos:
        current_time = time.time()
        dt = current_time - last_time if last_time else 0.1
        last_time = current_time
        
        # 构造观测向量 [相位差1, 相位差2, 相位差3, 相位差4, 距离]
        z = np.array([
            phase_diffs.get('1-0', 0),
            phase_diffs.get('2-1', 0),
            phase_diffs.get('3-2', 0),
            phase_diffs.get('0-3', 0),
            float(match_pos[0][0]) / 100  # 距离（米）
        ])
        
        # IMM-EKF处理
        imm.predict(dt)
        x_est = imm.update(z, AnchorPos)
        
        # 实时输出结果
        print(f"实时坐标: X={x_est[0]:.2f}m, Y={x_est[1]:.2f}m, Z={x_est[2]:.2f}m,dt= {dt}")
        data = { "p":[x_est[0],x_est[1],x_est[2]]}
        client.publish("vd/scatter3d", json.dumps(data))

        data = {"xy": [x_est[0],x_est[1]],"xz":[x_est[0],x_est[2]] }
        client.publish("vd/scatter", json.dumps(data))