# process_data.py
import sys
import io
import pandas as pd
import numpy as np
import glob
import os
from scipy.interpolate import interp1d
import requests
from datetime import datetime
import subprocess

# 强制标准输出使用 UTF-8 编码（解决 Windows 控制台打印 emoji 的问题）
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DRY_END = 150
MAILLARD_END = 530
DEV_END = 800

# 烘焙厂坐标（IJmuiden）
LAT = 52.4591659
LON = 4.595659

def find_sheet_by_keywords(xls, keywords):
    for sheet in xls.sheet_names:
        if any(kw in sheet.lower() for kw in keywords):
            return sheet
    return None

def extract_start_gas(df_gas):
    """提取起始燃气值：优先入豆前最后5秒众数，否则取入豆后第一个非零值"""
    df_gas = df_gas.copy()
    df_gas['gascontrol'] = pd.to_numeric(df_gas['gascontrol'], errors='coerce')
    df_gas = df_gas.dropna(subset=['gascontrol'])

    pre = df_gas[df_gas['time'] < 0]
    if len(pre) > 0:
        last_time = pre['time'].max()
        start_time = max(last_time - 5, pre['time'].min())
        recent = pre[pre['time'] >= start_time]
        if len(recent) > 0:
            vals = recent['gascontrol'].values
            rounded = np.round(vals, 1)
            unique, counts = np.unique(rounded, return_counts=True)
            mode_val = unique[np.argmax(counts)]
            return int(round(mode_val))

    post = df_gas[df_gas['time'] >= 0]
    if len(post) > 0:
        non_zero = post[post['gascontrol'] != 0]
        if len(non_zero) > 0:
            first_val = non_zero.iloc[0]['gascontrol']
            return int(round(first_val))

    return np.nan

def get_weather(date_str):
    """根据日期（YYYY-MM-DD）获取当天平均温度和湿度"""
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": LAT,
            "longitude": LON,
            "start_date": date_str,
            "end_date": date_str,
            "daily": ["temperature_2m_mean", "relative_humidity_2m_mean"],
            "timezone": "Europe/Amsterdam"
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            temp = data["daily"]["temperature_2m_mean"][0]
            hum = data["daily"]["relative_humidity_2m_mean"][0]
            return temp, hum
        else:
            return np.nan, np.nan
    except Exception as e:
        print(f"  ⚠️ 获取天气失败: {e}")
        return np.nan, np.nan

def load_roast_data(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        bean_keywords = ['bean temperature', 'bt', '温度', 'bean temp', '豆温']
        gas_keywords = ['gas control', 'gas', '燃气', 'gas control', '燃气控制']

        bean_sheet = find_sheet_by_keywords(xls, bean_keywords)
        if bean_sheet is None:
            return None
        df_bean = pd.read_excel(xls, sheet_name=bean_sheet, header=None, usecols=[0,1])
        df_bean.columns = ['time', 'beantemp']
        df_bean['time'] = pd.to_numeric(df_bean['time'], errors='coerce')
        df_bean = df_bean.dropna(subset=['time']).sort_values('time')

        gas_sheet = find_sheet_by_keywords(xls, gas_keywords)
        if gas_sheet:
            df_gas = pd.read_excel(xls, sheet_name=gas_sheet, header=None, usecols=[0,1])
            df_gas.columns = ['time', 'gascontrol']
            df_gas['time'] = pd.to_numeric(df_gas['time'], errors='coerce')
            df_gas['gascontrol'] = pd.to_numeric(df_gas['gascontrol'], errors='coerce')
            df_gas = df_gas.dropna(subset=['time', 'gascontrol']).sort_values('time')
            start_gas = extract_start_gas(df_gas)
        else:
            df_gas = None
            start_gas = np.nan

        if df_gas is not None:
            merged = pd.merge(df_bean, df_gas, on='time', how='outer').sort_values('time')
            merged['gascontrol'] = merged['gascontrol'].ffill().fillna(0)
        else:
            merged = df_bean.copy()
            merged['gascontrol'] = 0

        merged = merged[merged['time'] >= 0].reset_index(drop=True)
        if merged.empty:
            return None

        time_max = int(np.ceil(merged['time'].max()))
        uniform_time = np.arange(0, time_max + 1)
        interp_bean = interp1d(merged['time'], merged['beantemp'], kind='linear', fill_value='extrapolate')
        interp_gas = interp1d(merged['time'], merged['gascontrol'], kind='linear', fill_value='extrapolate')
        bean_uniform = interp_bean(uniform_time)
        gas_uniform = interp_gas(uniform_time)

        result = pd.DataFrame({'time_sec': uniform_time, 'beantemp': bean_uniform, 'gascontrol': gas_uniform})
        result.attrs['start_gas'] = start_gas
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None

def compute_phase_metrics(df):
    df = df[df['time_sec'] >= 0].copy()
    if df.empty:
        return {}
    dry = df[df['time_sec'] <= DRY_END]
    mail = df[(df['time_sec'] > DRY_END) & (df['time_sec'] <= MAILLARD_END)]
    dev = df[(df['time_sec'] > MAILLARD_END) & (df['time_sec'] <= DEV_END)]

    dry_gas = dry['gascontrol'].sum() if not dry.empty else 0
    mail_gas = mail['gascontrol'].sum() if not mail.empty else 0
    dev_gas = dev['gascontrol'].sum() if not dev.empty else 0
    total_gas = dry_gas + mail_gas + dev_gas

    dry_temp_rise = dry['beantemp'].iloc[-1] - dry['beantemp'].iloc[0] if len(dry) > 1 else 0
    mail_temp_rise = mail['beantemp'].iloc[-1] - mail['beantemp'].iloc[0] if len(mail) > 1 else 0
    dev_temp_rise = dev['beantemp'].iloc[-1] - dev['beantemp'].iloc[0] if len(dev) > 1 else 0
    total_temp_rise = dry_temp_rise + mail_temp_rise + dev_temp_rise

    dry_eff = (dry_temp_rise / dry_gas * 100) if dry_gas > 0 else 0
    mail_eff = (mail_temp_rise / mail_gas * 100) if mail_gas > 0 else 0
    dev_eff = (dev_temp_rise / dev_gas * 100) if dev_gas > 0 else 0
    total_eff = (total_temp_rise / total_gas * 100) if total_gas > 0 else 0

    return {
        'dry_gas': dry_gas, 'mail_gas': mail_gas, 'dev_gas': dev_gas, 'total_gas': total_gas,
        'dry_temp_rise': dry_temp_rise, 'mail_temp_rise': mail_temp_rise,
        'dev_temp_rise': dev_temp_rise, 'total_temp_rise': total_temp_rise,
        'dry_efficiency': dry_eff, 'mail_efficiency': mail_eff,
        'dev_efficiency': dev_eff, 'total_efficiency': total_eff
    }

def compute_deviation(current_df, ref_df):
    interp_cur = interp1d(current_df['time_sec'], current_df['beantemp'], kind='linear', fill_value='extrapolate')
    temp_on_ref = interp_cur(ref_df['time_sec'])
    return np.trapz(np.abs(temp_on_ref - ref_df['beantemp']), ref_df['time_sec'])

def main():
    print("="*50)
    print("开始计算分阶段汇总指标...")
    print("="*50)

    if not os.path.exists('reference.xls'):
        print("❌ 基准文件 reference.xls 不存在！")
        return
    print("处理基准文件 reference.xls...")
    ref_df = load_roast_data('reference.xls')
    if ref_df is None:
        print("❌ 基准文件加载失败，无法继续。")
        return
    ref_df.to_csv('processed_ref.csv', index=False)
    ref_summary = compute_phase_metrics(ref_df)
    ref_summary['batch_id'] = 'reference'
    ref_summary['deviation'] = 0.0
    ref_summary['start_gas'] = ref_df.attrs.get('start_gas', np.nan)
    pd.DataFrame([ref_summary]).to_csv('ref_summary.csv', index=False)
    print("✅ 基准汇总已保存到 ref_summary.csv")

    roast_files = glob.glob('roasts/*.xls*')
    if not roast_files:
        print("⚠️ roasts 文件夹中没有找到任何 .xls 或 .xlsx 文件。")
        return

    print(f"\n找到 {len(roast_files)} 个待处理文件:")
    for f in roast_files:
        print(f"  - {os.path.basename(f)}")

    all_timeseries, all_summaries = [], []
    success_count = 0

    for f in roast_files:
        fname = os.path.basename(f)
        print(f"\n处理文件: {fname}")
        batch_df = load_roast_data(f)
        if batch_df is None:
            print(f"  ❌ 跳过文件 {fname}")
            continue
        batch_df['batch_id'] = fname
        all_timeseries.append(batch_df)

        summary = compute_phase_metrics(batch_df)
        summary['batch_id'] = fname
        summary['deviation'] = compute_deviation(batch_df, ref_df)
        summary['start_gas'] = batch_df.attrs.get('start_gas', np.nan)

        # 添加日期和天气
        try:
            df_gen = pd.read_excel(f, sheet_name='General')
            date_col = None
            for col in df_gen.columns:
                if 'date' in col.lower():
                    date_col = col
                    break
            if date_col is None and len(df_gen.columns) > 0:
                date_col = df_gen.columns[0]
            if date_col:
                first_valid = df_gen[date_col].dropna().iloc[0] if not df_gen[date_col].dropna().empty else None
                if first_valid is not None:
                    if isinstance(first_valid, str):
                        date_str = first_valid.split(',')[0].strip()
                    else:
                        date_str = pd.to_datetime(first_valid).date().isoformat()
                    temp, hum = get_weather(date_str)
                    summary['avg_temp'] = temp
                    summary['avg_humidity'] = hum
                else:
                    summary['avg_temp'] = np.nan
                    summary['avg_humidity'] = np.nan
            else:
                summary['avg_temp'] = np.nan
                summary['avg_humidity'] = np.nan
        except Exception as e:
            print(f"  ⚠️ 无法读取日期或天气: {e}")
            summary['avg_temp'] = np.nan
            summary['avg_humidity'] = np.nan

        # 提取机器型号
        try:
            machine_raw = df_gen.iloc[0]['Machine']
            if '30' in str(machine_raw) or '30A' in str(machine_raw):
                machine_id = 'Giesen 30A'
            elif '15' in str(machine_raw) or 'W15' in str(machine_raw):
                machine_id = 'Giesen 15'
            else:
                machine_id = machine_raw.strip()
        except:
            machine_id = 'Unknown'
        summary['machine_id'] = machine_id

        # 提取咖啡种类
        try:
            green_lots = df_gen.iloc[0].get('Green lots', '')
            if 'Paz' in str(green_lots):
                coffee_type = 'Paz'
            elif 'FLORA' in str(green_lots):
                coffee_type = 'FLORA'
            else:
                coffee_type = 'Other'
        except:
            coffee_type = 'Unknown'
        summary['coffee_type'] = coffee_type

        all_summaries.append(summary)
        success_count += 1
        print(f"  ✅ 成功处理 {fname}，起始燃气 = {summary['start_gas']:.1f}%")

    if all_timeseries:
        pd.concat(all_timeseries, ignore_index=True).to_csv('master_timeseries.csv', index=False)
    if all_summaries:
        df_summaries = pd.DataFrame(all_summaries)
        if 'start_gas' not in df_summaries.columns:
            df_summaries['start_gas'] = np.nan
        df_summaries.to_csv('master_summary.csv', index=False)
        print("✅ 汇总数据已保存，正在训练 AI 模型...")
        subprocess.run([sys.executable, 'train_model.py'])

    print(f"\n🎉 处理完成！成功处理 {success_count}/{len(roast_files)} 个文件。")

if __name__ == '__main__':
    main()