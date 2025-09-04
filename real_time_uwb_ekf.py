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
    def __init__(self):
        # 模型概率初始化
        self.probs = np.array([0.2, 0.8])  # CV: 0.2, CA: 0.8
        self.last_probs = None
        
        # 模型转移概率
        self.transition = np.array([[0.2, 0.8],   # CV -> CV, CV -> CA
                                   [0.05, 0.95]]) # CA -> CV, CA -> CA
        
        # 初始化CV和CA模型滤波器
        self.filters = [
            self._create_cv_filter(),
            self._create_ca_filter(sigma_a=12.0)
        ]
        
        # 观测噪声矩阵
        self.R_phi = np.diag(phiStd ** 2)  # 相位差噪声
        self.R_dis = disStd ** 2                           # 距离噪声
        self.R = np.block([[self.R_phi, np.zeros((kAntennaCount, 1))],
              [np.zeros((1, kAntennaCount)), self.R_dis]])
    
    def _create_cv_filter(self):
        """创建匀速模型滤波器"""
        Q = np.diag([0.01, 0.01, 0.08, 0.5, 0.5, 0.1])
        return {'x': x_init.copy(), 'P': P_init.copy(), 'Q': Q}
    
    def _create_ca_filter(self, sigma_a=1.0):
        """创建匀加速模型滤波器"""
        dt = 1.0 / 10  # 假设默认采样率
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
    
    def get_F_matrix(self, dt):
        """状态转移矩阵（CV模型）"""
        return np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
    
    def mixing(self):
        """模型交互混合"""
        c = np.dot(self.transition.T, self.last_probs)
        mixing_weights = (self.transition * self.last_probs) / c[:, np.newaxis]
        
        mixed_states = []
        mixed_covariances = []
        for j in range(2):
            x_mixed = np.zeros(6)
            P_mixed = np.zeros((6, 6))
            for i in range(2):
                x_mixed += mixing_weights[i, j] * self.filters[i]['x']
                dx = self.filters[i]['x'] - x_mixed
                P_mixed += mixing_weights[i, j] * (self.filters[i]['P'] + np.outer(dx, dx))
            mixed_states.append(x_mixed)
            mixed_covariances.append(P_mixed)
        
        for i in range(2):
            self.filters[i]['x'] = mixed_states[i]
            self.filters[i]['P'] = mixed_covariances[i]
    
    def predict(self, dt):
        """预测步"""
        if self.last_probs is not None:
            self.mixing()
        F = self.get_F_matrix(dt)
        for f in self.filters:
            f['x'] = F @ f['x']
            f['P'] = F @ f['P'] @ F.T + f['Q']
        self.last_probs = self.probs.copy()
    
    def observation_model(self, x, AnchorPos):
        """观测模型：相位差 + 距离"""
        pos = x[:3]
        r = np.linalg.norm(pos - AnchorPos)
        vTO = (AnchorPos - pos) / r  # 单位方向向量
        
        PhiDiff_ekf = np.zeros(kAntennaCount)
        for i in range(kAntennaCount):
            vNab =np.array([Nab[0, i], Nab[1, i], 0]) 
            deltaR = np.dot(vTO, vNab)
            PhiDiff_ekf[i] = (2 * np.pi / lambda_val) * deltaR
        
        return np.concatenate([PhiDiff_ekf, [r]])
    
    def compute_H(self, x, AnchorPos, eps=1e-6):
        """数值法计算雅可比矩阵"""
        H = np.zeros((kAntennaCount + 1, 6))
        h0 = self.observation_model(x, AnchorPos)
        
        for i in range(6):
            dx = np.zeros(6)
            dx[i] = eps
            h_eps = self.observation_model(x + dx, AnchorPos)
            H[:, i] = (h_eps - h0) / eps
        
        return H
    
    def update(self, z, AnchorPos):
        """更新步"""
        likelihoods = []
        for i, f in enumerate(self.filters):
            # 计算雅可比矩阵H
            H = self.compute_H(f['x'], AnchorPos)
            if np.any(np.isnan(H)):
                H = np.zeros_like(H)
                print(f"滤波器{i}雅可比异常！")
            # 计算卡尔曼增益
            S = H @ f['P'] @ H.T + self.R
            S = 0.5 * (S + S.T) + 1e-6 * np.eye(S.shape[0])  #正则化
            try:
                K = f['P'] @ H.T @ np.linalg.inv(S)
            except:
                K = np.zeros((6, S.shape[0]))
                print(f"滤波器{i}增益计算失败！")
            
            z_pred = self.observation_model(f['x'], AnchorPos)
            innovation = z - z_pred
            f['x'] += K @ innovation
            f['P'] = (np.eye(6) - K @ H) @ f['P']
            
            # 计算模型似然
            # 计算模型似然（多元高斯分布）
            det_S = np.linalg.det(S)
            if det_S <= 0: det_S = 1e-6  # 防错处理
            likelihood = np.exp(-0.5 * innovation.T @ np.linalg.inv(S) @ innovation) / np.sqrt((2*np.pi)**4 * det_S)
            likelihoods.append(likelihood)
        
        # 更新模型概率
        pred_probs = self.transition.T @ self.probs
        self.probs = likelihoods * pred_probs / np.dot(likelihoods, pred_probs)
        self.probs = np.clip(self.probs, 0.03, 0.97)
        self.probs /= np.sum(self.probs)
        
        # 融合输出
        x_est = sum(p * f['x'] for p, f in zip(self.probs, self.filters))
        return x_est

# ------------------ 主循环 ------------------
imm = IMMFilter()
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
        data = { "p":[x_est[0],-x_est[1],x_est[2]]}
        client.publish("vd/scatter3d", json.dumps(data))

        data = {"xy": [x_est[0],-x_est[1]], }
        client.publish("vd/scatter", json.dumps(data))