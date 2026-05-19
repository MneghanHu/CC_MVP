# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor


def train():
    df = pd.read_csv('master_summary.csv')
    if df.empty:
        print("❌ No data to train")
        return

    # 特征列
    feature_cols = ['avg_humidity', 'avg_temp']

    # 咖啡种类编码
    if 'coffee_type' in df.columns:
        coffee_map = {'Paz': 0, 'FLORA': 1, 'Other': 2, 'Unknown': 3}
        df['coffee_code'] = df['coffee_type'].map(coffee_map).fillna(3)
        feature_cols.append('coffee_code')

    # 机器编码
    if 'machine_id' in df.columns:
        machine_map = {'Giesen 15': 0, 'Giesen 30A': 1, 'Unknown': 2}
        df['machine_code'] = df['machine_id'].map(machine_map).fillna(2)
        feature_cols.append('machine_code')

    # 目标变量
    target_start = 'start_gas'
    target_gas = 'total_gas'
    target_dev = 'deviation'

    # 删除缺失值
    df_model = df.dropna(subset=feature_cols + [target_start, target_gas, target_dev])
    if len(df_model) < 5:
        print(f"❌ Only {len(df_model)} valid samples, need at least 5")
        return

    X = df_model[feature_cols]

    # 训练三个随机森林模型
    model_start = RandomForestRegressor(n_estimators=50, random_state=42)
    model_start.fit(X, df_model[target_start])

    model_gas = RandomForestRegressor(n_estimators=50, random_state=42)
    model_gas.fit(X, df_model[target_gas])

    model_dev = RandomForestRegressor(n_estimators=50, random_state=42)
    model_dev.fit(X, df_model[target_dev])

    # 保存模型和特征列表
    joblib.dump(model_start, 'model_start.pkl')
    joblib.dump(model_gas, 'model_gas.pkl')
    joblib.dump(model_dev, 'model_dev.pkl')
    joblib.dump(feature_cols, 'model_features.pkl')

    print("✅ Models trained and saved.")


if __name__ == '__main__':
    train()