import requests
import pandas as pd

# 请确认你的烘焙厂坐标（此处使用阿姆斯特丹中心坐标，如有更精确坐标可替换）
LAT = 52.3676
LON = 4.9041

# 根据你数据中的日期范围调整（建议覆盖所有烘焙日期）
START = "2024-01-01"
END = "2025-12-31"

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": START,
    "end_date": END,
    "daily": ["temperature_2m_mean", "relative_humidity_2m_mean"],
    "timezone": "Europe/Amsterdam"
}

response = requests.get(url, params=params)
data = response.json()

weather = pd.DataFrame({
    "date": pd.to_datetime(data["daily"]["time"]),
    "avg_temp": data["daily"]["temperature_2m_mean"],
    "avg_humidity": data["daily"]["relative_humidity_2m_mean"]
})

weather.to_csv("weather_data.csv", index=False)
print(f"已获取 {len(weather)} 天的天气数据，保存为 weather_data.csv")