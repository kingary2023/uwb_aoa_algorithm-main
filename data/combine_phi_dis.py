import pandas as pd

# 读取两个CSV文件
df_phases = pd.read_csv("0905_200_phase_cali_data.csv")  # 相位差数据
df_dis = pd.read_csv("0905_200_distance_data.csv")   # 距离数据

# 检查数据长度是否匹配

# 合并数据（将dis添加到最右列）
df_combined = pd.concat([df_phases, df_dis["dis"]], axis=1)

# 保存合并后的CSV
df_combined.to_csv("0905_200_combined_data.csv", index=False)
print(f"合并完成！保存为 combined_data.csv")