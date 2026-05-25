# train_model.py - 带训练/测试分割和保存指标到 CSV
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


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
    needed_cols = feature_cols + [target_start, target_gas, target_dev]
    df_model = df.dropna(subset=needed_cols)
    if len(df_model) < 5:
        print(f"❌ Only {len(df_model)} valid samples, need at least 5")
        return

    X = df_model[feature_cols]
    y_start = df_model[target_start]
    y_gas = df_model[target_gas]
    y_dev = df_model[target_dev]

    # 分割训练集和测试集 (80% / 20%)
    X_train, X_test, y_start_train, y_start_test, y_gas_train, y_gas_test, y_dev_train, y_dev_test = train_test_split(
        X, y_start, y_gas, y_dev, test_size=0.2, random_state=42
    )

    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # 训练模型（仅在训练集上）
    model_start = RandomForestRegressor(n_estimators=50, random_state=42)
    model_start.fit(X_train, y_start_train)

    model_gas = RandomForestRegressor(n_estimators=50, random_state=42)
    model_gas.fit(X_train, y_gas_train)

    model_dev = RandomForestRegressor(n_estimators=50, random_state=42)
    model_dev.fit(X_train, y_dev_train)

    # 在测试集上评估
    start_pred = model_start.predict(X_test)
    gas_pred = model_gas.predict(X_test)
    dev_pred = model_dev.predict(X_test)

    start_mae = mean_absolute_error(y_start_test, start_pred)
    start_r2 = r2_score(y_start_test, start_pred)
    gas_mae = mean_absolute_error(y_gas_test, gas_pred)
    gas_r2 = r2_score(y_gas_test, gas_pred)
    dev_mae = mean_absolute_error(y_dev_test, dev_pred)
    dev_r2 = r2_score(y_dev_test, dev_pred)

    print("\n=== Model Evaluation on Test Set ===")
    print(f"Start Gas  - MAE: {start_mae:.2f}%, R²: {start_r2:.3f}")
    print(f"Total Gas  - MAE: {gas_mae:.0f} %·s, R²: {gas_r2:.3f}")
    print(f"Deviation  - MAE: {dev_mae:.0f} °C·s, R²: {dev_r2:.3f}")

    # 保存评估指标到 CSV
    metrics_df = pd.DataFrame([{
        'start_gas_mae': start_mae,
        'start_gas_r2': start_r2,
        'total_gas_mae': gas_mae,
        'total_gas_r2': gas_r2,
        'deviation_mae': dev_mae,
        'deviation_r2': dev_r2
    }])
    metrics_df.to_csv('model_metrics.csv', index=False)
    print("✅ Model metrics saved to model_metrics.csv")

    # 使用全部数据重新训练模型（以获得最佳泛化能力）
    print("\nRetraining on full dataset...")
    model_start_full = RandomForestRegressor(n_estimators=50, random_state=42)
    model_start_full.fit(X, y_start)

    model_gas_full = RandomForestRegressor(n_estimators=50, random_state=42)
    model_gas_full.fit(X, y_gas)

    model_dev_full = RandomForestRegressor(n_estimators=50, random_state=42)
    model_dev_full.fit(X, y_dev)

    # 保存模型和特征列表
    joblib.dump(model_start_full, 'model_start.pkl')
    joblib.dump(model_gas_full, 'model_gas.pkl')
    joblib.dump(model_dev_full, 'model_dev.pkl')
    joblib.dump(feature_cols, 'model_features.pkl')

    print("✅ Models (trained on full data) saved.")


if __name__ == '__main__':
    train()