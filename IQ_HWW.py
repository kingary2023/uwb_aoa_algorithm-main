import re
import serial
import math
import json
import numpy as np
import paho.mqtt.client as mqtt


MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
RADIUS = 3.0
UPDATE_HZ = 10


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
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.reconnect_delay_set(min_delay=1, max_delay=4)
client.loop_start()


COM = serial.Serial("COM4", 921600, timeout=1)
ANT_IQ = []
phase = []
phase_diffs = {}
pattern1 = r"\[APP\]\[INFO\]ANT IQ:\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)"
pattern2 = r"FiRa DS-TWR Deferred Controlee SEQ NUM"
pattern3 = r"\[APP\]\[INFO\]Peer AAA1, Distance (\d+)cm, PDoA Azimuth ([-+]?\d+) Elevation ([-+]?\d+) Azimuth FoM"
while COM.is_open:
    data = COM.readline().decode("utf-8", errors="ignore")
    print("Raw:", data.strip())  # 查看实际数据格式
    match1 = re.findall(pattern1, data)
    match2 = re.findall(pattern2, data)
    match3 = re.findall(pattern3, data)
    if match2:
        if len(ANT_IQ) < 4:
            print(ANT_IQ)
            phase.clear()
            ANT_IQ.clear()
            continue
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

        client.publish("vd/line", json.dumps(phase_diffs))
        # for i_real, i_imag in ANT_IQ:
        # phase.append(math.atan2(i_imag, i_real))
        # client.publish(
        #     "vd/line",
        #     json.dumps(
        #         {
        #             f"1-0": phase[1] - phase[0],
        #             "2-1": phase[2] - phase[1],
        #             "3-2": phase[3] - phase[2],
        #             "0-3": phase[0] - phase[3],
        #         }
        #     ),
        # )
        # phase.clear()
        ANT_IQ.clear()
    if match1:

        for i, (i_real, i_imag) in enumerate(match1):
            # ANT_IQ.append(float(i_real), float(i_imag))
            ANT_IQ.append(complex(float(i_real), float(i_imag)))
    if match3:

        client.publish(
            "PDoA",
            json.dumps(
                {
                    "dis": eval(match3[0][0]) / 100,
                    "Azimuth": eval(match3[0][1]),
                    "Elevation": eval(match3[0][2]),
                }
            ),
        )
