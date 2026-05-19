import pandas as pd

# 读取数据
roast = pd.read_csv("master_summary.csv")
weather = pd.read_csv("weather_data.csv")

# 统一日期格式（只保留年月日）
roast['date'] = pd.to_datetime(roast['date']).dt.date
weather['date'] = pd.to_datetime(weather['date']).dt.date

# 合并
merged = pd.merge(roast, weather, on='date', how='left')
missing = merged['avg_humidity'].isna().sum()
print(f"共有 {missing} 条烘焙记录没有匹配到天气数据")

# 保存新文件
merged.to_csv("master_with_weather.csv", index=False)
print("已生成 master_with_weather.csv")
