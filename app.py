# app.py - 完整版（不显示 MAE 和 R² 指标）
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import scipy.stats as stats
from scipy.interpolate import interp1d
import os
import sys
import subprocess
import glob
from datetime import date
import joblib

st.set_page_config(page_title="Coffee Roast MVP", layout="wide")
st.title("☕️ Coffee Energy Efficiency Monitoring System")


# ----------------------------------------------------------------------
# 辅助函数：安全运行 process_data.py
def run_process_data(show_message=True):
    """执行 process_data.py，返回 (success, error_message)"""
    try:
        os.makedirs("roasts", exist_ok=True)
        result = subprocess.run(
            [sys.executable, "process_data.py"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            if show_message:
                st.sidebar.success("✅ Data processed successfully!")
            return True, ""
        else:
            error_msg = f"❌ Process failed (code {result.returncode})\n{result.stderr}"
            if show_message:
                st.sidebar.error(error_msg)
            return False, error_msg
    except subprocess.TimeoutExpired:
        error_msg = "⏱️ process_data.py timed out after 120 seconds."
        if show_message:
            st.sidebar.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"⚠️ Unexpected error: {str(e)}"
        if show_message:
            st.sidebar.error(error_msg)
        return False, error_msg


def refresh_data_and_rerun():
    """清除缓存并强制刷新页面"""
    st.cache_data.clear()
    st.rerun()


# ----------------------------------------------------------------------
# 数据加载（带缓存）
@st.cache_data
def load_data():
    ref_ts = pd.read_csv('processed_ref.csv')
    ref_sum = pd.read_csv('ref_summary.csv')
    master_ts = pd.read_csv('master_timeseries.csv')
    master_sum = pd.read_csv('master_summary.csv')
    return ref_ts, ref_sum, master_ts, master_sum


# 尝试加载数据，若失败则显示警告但程序不崩溃
try:
    ref_ts, ref_sum, master_ts, master_sum = load_data()
    ref_sum = ref_sum.iloc[0]
    master_sum = master_sum.copy()
    master_sum['start_gas_num'] = pd.to_numeric(master_sum['start_gas'], errors='coerce')
    master_sum['start_gas_int'] = master_sum['start_gas_num'].round(0).astype('Int64')
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.warning(
        f"⚠️ Data not loaded: {e}\n\nPlease ensure you have run `python process_data.py` and generated the CSV files.")

# ----------------------------------------------------------------------
# 侧边栏全局筛选（仅在数据加载成功时显示）
if data_loaded:
    if 'machine_id' in master_sum.columns:
        machines = sorted(master_sum['machine_id'].dropna().unique())
        # 将 'Unknown' 移到最后
        machines = [m for m in machines if m != 'Unknown'] + (['Unknown'] if 'Unknown' in machines else [])
        selected_machine = st.sidebar.selectbox("Select Machine:", options=['All'] + machines)
    else:
        selected_machine = 'All'

    if 'coffee_type' in master_sum.columns:
        coffee_types = sorted(master_sum['coffee_type'].dropna().unique())
        # 将 'Other' 和 'Unknown' 移到最后
        coffee_types = [c for c in coffee_types if c not in ['Other', 'Unknown']] + \
                       [c for c in coffee_types if c in ['Other', 'Unknown']]
        selected_coffee = st.sidebar.selectbox("Select Coffee Type:", options=['All'] + coffee_types)
    else:
        selected_coffee = 'All'

    if selected_machine != 'All':
        master_sum = master_sum[master_sum['machine_id'] == selected_machine]
        master_ts = master_ts[master_ts['batch_id'].isin(master_sum['batch_id'])]
    if selected_coffee != 'All':
        master_sum = master_sum[master_sum['coffee_type'] == selected_coffee]
        master_ts = master_ts[master_ts['batch_id'].isin(master_sum['batch_id'])]

    available_gas = master_sum['start_gas_int'].dropna().unique()
    available_gas = sorted(available_gas)
    if available_gas:
        st.sidebar.write("**Counts per start gas value:**")
        for g in available_gas:
            cnt = (master_sum['start_gas_int'] == g).sum()
            st.sidebar.write(f"  {g}%: {cnt} batches")
    else:
        st.sidebar.info("No batches with valid start gas.")
else:
    st.sidebar.warning("Data not loaded. Please upload at least one batch or run process_data.py")

# ----------------------------------------------------------------------
# 文件上传与删除功能
st.sidebar.markdown("---")
st.sidebar.markdown("## 📤 Import New Roast")
os.makedirs("roasts", exist_ok=True)

uploaded_files = st.sidebar.file_uploader(
    "Select Excel files (.xls or .xlsx)",
    type=["xls", "xlsx"],
    accept_multiple_files=True,
    help="Select one or more Cropster export files."
)

# 复选框：将此批次设为所属咖啡种类的标准参考曲线
set_as_ref = st.sidebar.checkbox(
    "Set as standard reference for this coffee type",
    value=False,
    help="If checked, this roast will be saved as the reference curve for its coffee type (e.g., reference_Guji-12kg.xls)."
)

if st.sidebar.button("📤 Upload Selected Files"):
    if not uploaded_files:
        st.sidebar.warning("No files selected.")
    else:
        with st.spinner(f"Saving {len(uploaded_files)} file(s) and processing..."):
            for uploaded_file in uploaded_files:
                # 保存到 roasts 文件夹
                save_path = os.path.join("roasts", uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 如果用户勾选了“作为参考曲线”，则额外保存为 reference_{咖啡种类}.xls
                if set_as_ref:
                    try:
                        # 临时读取 Excel 获取咖啡种类（从 General 页签第二行 B 列）
                        temp_xls = pd.ExcelFile(uploaded_file)
                        df_gen = pd.read_excel(temp_xls, sheet_name='General', header=None)
                        row_idx = 1 if df_gen.shape[0] > 1 else 0
                        lot_name = df_gen.iloc[row_idx, 1] if df_gen.shape[1] > 1 else ''
                        coffee_type = str(lot_name).strip() if pd.notna(lot_name) and str(
                            lot_name).strip() != '' else 'Other'
                        # 清理文件名（替换空格和斜杠）
                        safe_name = coffee_type.replace(' ', '_').replace('/', '_')
                        ref_path = f"reference_{safe_name}.xls"
                        # 复制文件
                        with open(ref_path, "wb") as rf:
                            rf.write(uploaded_file.getbuffer())
                        st.sidebar.success(f"✅ Also saved as reference for '{coffee_type}' → {ref_path}")
                    except Exception as e:
                        st.sidebar.error(f"Failed to set as reference: {e}")

            st.sidebar.success(f"✅ Saved {len(uploaded_files)} file(s).")
            success, _ = run_process_data(show_message=True)
            if success:
                refresh_data_and_rerun()
            else:
                st.sidebar.error("Data processing failed. Check file format or error details.")

if st.sidebar.button("🔄 Manually Refresh Data"):
    with st.spinner("Re-running process_data.py..."):
        success, _ = run_process_data(show_message=True)
        if success:
            refresh_data_and_rerun()
        else:
            st.sidebar.error("Refresh failed. See error above.")

# 删除批次
st.sidebar.markdown("---")
st.sidebar.markdown("## 🗑️ Delete a Roast Batch")
if data_loaded and not master_sum.empty:
    all_batch_ids = master_sum['batch_id'].tolist()
    selected_delete = st.sidebar.selectbox("Select batch to delete:", all_batch_ids)
    if st.sidebar.button("Delete Selected Batch"):
        file_path = os.path.join("roasts", selected_delete)
        deleted = False
        if os.path.exists(file_path):
            os.remove(file_path)
            deleted = True
        else:
            matched = glob.glob(os.path.join("roasts", f"*{selected_delete}*"))
            if matched:
                os.remove(matched[0])
                deleted = True
        if deleted:
            st.sidebar.success(f"✅ Deleted file: {selected_delete}")
            with st.spinner("Updating data after deletion..."):
                success, _ = run_process_data(show_message=True)
                if success:
                    refresh_data_and_rerun()
                else:
                    st.sidebar.error("Data update failed. Please check manually.")
        else:
            st.sidebar.error(f"❌ File not found for {selected_delete}")
else:
    st.sidebar.info("No batches available to delete.")

# ----------------------------------------------------------------------
# 天气推荐模块
st.sidebar.markdown("---")
st.sidebar.markdown("## 🌦️ Weather‑Based Recommendation")
if data_loaded:
    today_humidity = st.sidebar.number_input(
        "Today's humidity (%)", min_value=0.0, max_value=100.0,
        value=70.0, step=1.0
    )
    if st.sidebar.button("Get Recommended Start Gas"):
        good_batches = master_sum[
            (master_sum['deviation'] < master_sum['deviation'].quantile(0.3)) &
            (master_sum['avg_humidity'].notna())
            ]
        if len(good_batches) >= 10:
            slope, intercept, _, _, _ = stats.linregress(
                good_batches['avg_humidity'],
                good_batches['start_gas_num']
            )
            recommended = intercept + slope * today_humidity
            recommended = max(20, min(100, recommended))
            st.sidebar.success(f"✅ Recommended start gas: **{recommended:.1f}%**")
            st.sidebar.caption(f"Sensitivity: {slope:.2f}% per 1% humidity.")
            st.session_state['humidity_slope'] = slope
        else:
            st.sidebar.warning(f"Only {len(good_batches)} batches with weather data. Need ≥10.")
else:
    st.sidebar.warning("Data not loaded, weather recommendation unavailable.")

# ----------------------------------------------------------------------
# AI 预测器（基础版）
st.sidebar.markdown("---")
st.sidebar.markdown("## 🤖 AI Predictor (Start Gas, Gas, Deviation)")

if st.sidebar.button("🔄 Train Models"):
    if not data_loaded:
        st.sidebar.error("Data not loaded. Cannot train models.")
    else:
        result = subprocess.run([sys.executable, "train_model.py"], capture_output=True, text=True)
        if result.returncode == 0:
            st.sidebar.success("Models trained!")
        else:
            st.sidebar.error("Training failed.")
            st.sidebar.code(result.stderr)

if data_loaded:
    pred_humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 70.0)
    pred_temp = st.sidebar.number_input("Temperature (°C)", -10.0, 40.0, 15.0)
    if 'coffee_type' in master_sum.columns:
        coffee_vals = sorted(master_sum['coffee_type'].dropna().unique())
        if 'Unknown' not in coffee_vals:
            coffee_vals.append('Unknown')
        pred_coffee = st.sidebar.selectbox("Coffee Type", coffee_vals)
    else:
        pred_coffee = 'Unknown'
    if 'machine_id' in master_sum.columns:
        machine_vals = sorted(master_sum['machine_id'].dropna().unique())
        if 'Unknown' not in machine_vals:
            machine_vals.append('Unknown')
        pred_machine = st.sidebar.selectbox("Machine", machine_vals)
    else:
        pred_machine = 'Unknown'

    if st.sidebar.button("Predict"):
        model_files = ['model_start.pkl', 'model_gas.pkl', 'model_dev.pkl', 'model_features.pkl']
        if not all(os.path.exists(f) for f in model_files):
            st.sidebar.warning("Models not found. Please click 'Train Models' first.")
        else:
            model_start = joblib.load('model_start.pkl')
            model_gas = joblib.load('model_gas.pkl')
            model_dev = joblib.load('model_dev.pkl')
            feature_cols = joblib.load('model_features.pkl')

            feat_dict = {}
            for f in feature_cols:
                if f == 'avg_humidity':
                    feat_dict[f] = pred_humidity
                elif f == 'avg_temp':
                    feat_dict[f] = pred_temp
                elif f == 'coffee_code':
                    coffee_map = {'Paz': 0, 'FLORA': 1, 'Other': 2, 'Unknown': 3}
                    feat_dict[f] = coffee_map.get(pred_coffee, 3)
                elif f == 'machine_code':
                    machine_map = {'Giesen 15': 0, 'Giesen 30A': 1, 'Unknown': 2}
                    feat_dict[f] = machine_map.get(pred_machine, 2)
                else:
                    feat_dict[f] = 0

            X_pred = pd.DataFrame([feat_dict])[feature_cols]
            start_pred = model_start.predict(X_pred)[0]
            gas_pred = model_gas.predict(X_pred)[0]
            dev_pred = model_dev.predict(X_pred)[0]

            st.sidebar.markdown("---")
            st.sidebar.markdown("### ✅ Prediction Results")
            st.sidebar.write(f"**Recommended Start Gas:** {start_pred:.1f}%")
            st.sidebar.write(f"**Predicted Total Gas:** {gas_pred:.0f} %·s")
            st.sidebar.write(f"**Predicted Curve Deviation:** {dev_pred:.0f} ℃·s")
else:
    st.sidebar.warning("Data not loaded, AI predictor disabled.")

# ----------------------------------------------------------------------
# 视图选择（仅在数据加载成功时显示）
if data_loaded:
    view = st.sidebar.radio(
        "Select View",
        [
            "Single Batch Comparison",
            "Statistical Analysis by Start Gas",
            "Full Roast Energy Analysis",
            "Moisture & Density Analysis",
            "AI Roast Prediction"
        ]
    )

    # ============================= 视图 1 =============================
    if view == "Single Batch Comparison":
        batches = master_sum['batch_id'].tolist()
        if not batches:
            st.warning("No batches available after filtering.")
            st.stop()
        selected = st.sidebar.selectbox("Select batch to compare:", batches)
        current_sum = master_sum[master_sum['batch_id'] == selected].iloc[0]
        current_ts = master_ts[master_ts['batch_id'] == selected].sort_values('time_sec')

        # 动态获取该批次的咖啡种类，并尝试加载对应的基准曲线
        coffee_type = current_sum.get('coffee_type', 'Other')
        safe_name = coffee_type.replace(' ', '_').replace('/', '_')
        ref_ts_path = f'processed_ref_{safe_name}.csv'
        ref_sum_path = f'ref_summary_{safe_name}.csv'
        # 如果专用基准文件不存在，则使用默认
        if not os.path.exists(ref_ts_path):
            ref_ts_path = 'processed_ref.csv'
            ref_sum_path = 'ref_summary.csv'
        try:
            ref_ts_dynamic = pd.read_csv(ref_ts_path)
            ref_sum_dynamic = pd.read_csv(ref_sum_path).iloc[0]
        except Exception as e:
            st.error(f"Failed to load reference data for coffee type '{coffee_type}': {e}")
            st.stop()

        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.subheader("Temperature Curve Comparison")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ref_ts_dynamic['time_sec'], y=ref_ts_dynamic['beantemp'],
                                     name='Standard', line=dict(color='lightblue', dash='dot')))
            fig.add_trace(go.Scatter(x=current_ts['time_sec'], y=current_ts['beantemp'],
                                     name='Current', line=dict(color='darkblue')))
            fig.update_layout(xaxis_title="Time (seconds)", yaxis_title="Bean Temperature (°C)",
                              hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            html_str = pio.to_html(fig, include_plotlyjs='cdn')
            st.download_button(label="📥 Download Chart", data=html_str,
                               file_name="temperature_curve.html", mime="text/html")

        with col_right:
            st.subheader("📊 Energy Efficiency Metrics")
            total_gas_ref = ref_sum_dynamic['total_gas']
            total_gas_cur = current_sum['total_gas']
            energy_dev = ((total_gas_cur - total_gas_ref) / total_gas_ref) * 100 if total_gas_ref else 0
            st.metric("Total Energy Deviation", f"{energy_dev:+.1f}%", delta=f"{energy_dev:+.1f}%",
                      delta_color="inverse")

            st.markdown("**Phase-wise Gas Consumption vs Standard**")
            phases = ['dry', 'mail', 'dev']
            phase_names = ['Drying', 'Maillard', 'Development']
            for phase, name in zip(phases, phase_names):
                ref_gas = ref_sum_dynamic[f'{phase}_gas']
                cur_gas = current_sum[f'{phase}_gas']
                dev = ((cur_gas - ref_gas) / ref_gas) * 100 if ref_gas else 0
                st.write(f"{name}: {dev:+.1f}%")

            ref_eff = ref_sum_dynamic['total_efficiency']
            cur_eff = current_sum['total_efficiency']
            eff_dev = ((cur_eff - ref_eff) / ref_eff) * 100 if ref_eff else 0
            st.metric("Total Thermal Efficiency (°C/100%·s)", f"{cur_eff:.2f}", delta=f"{eff_dev:+.1f}%")

            st.markdown("**Phase-wise Thermal Efficiency**")
            for phase, name in zip(phases, phase_names):
                ref_eff_phase = ref_sum_dynamic[f'{phase}_efficiency']
                cur_eff_phase = current_sum[f'{phase}_efficiency']
                eff_dev_phase = ((cur_eff_phase - ref_eff_phase) / ref_eff_phase) * 100 if ref_eff_phase else 0
                st.write(f"{name}: {cur_eff_phase:.2f} ({eff_dev_phase:+.1f}%)")

            dev_val = current_sum['deviation']
            st.metric("Curve Deviation (℃·s)", f"{dev_val:.0f}")

            # 排名
            master_sum['dev_rank'] = master_sum['deviation'].rank(ascending=True, method='min')
            master_sum['gas_rank'] = master_sum['total_gas'].rank(ascending=True, method='min')
            total_batches = len(master_sum)
            dev_rank = int(master_sum[master_sum['batch_id'] == selected]['dev_rank'].iloc[0])
            gas_rank = int(master_sum[master_sum['batch_id'] == selected]['gas_rank'].iloc[0])


            def get_rating(rank, total):
                if rank <= total * 0.3:
                    return "Excellent", "green"
                elif rank <= total * 0.7:
                    return "Average", "orange"
                else:
                    return "Poor", "red"


            dev_rating, dev_color = get_rating(dev_rank, total_batches)
            gas_rating, gas_color = get_rating(gas_rank, total_batches)
            st.markdown("### Batch Performance Ranking")
            st.markdown(
                f"Curve Deviation Rank: **<span style='color:{dev_color}'>{dev_rank}/{total_batches}</span>** ({dev_rating})",
                unsafe_allow_html=True)
            st.markdown(
                f"Energy Consumption Rank: **<span style='color:{gas_color}'>{gas_rank}/{total_batches}</span>** ({gas_rating})",
                unsafe_allow_html=True)

            # 建议
            st.markdown("### Suggestions")
            advice = []
            if current_sum['dev_gas'] > master_sum['dev_gas'].mean():
                advice.append("• Development phase gas above average. Consider reducing gas earlier.")
            if current_sum['dry_efficiency'] < master_sum['dry_efficiency'].mean():
                advice.append("• Drying phase efficiency below average. Check charge temperature or airflow.")
            if current_sum['deviation'] > master_sum['deviation'].median():
                advice.append("• Curve deviation larger than typical. Try to follow standard profile more closely.")
            if not advice:
                advice.append("✅ This batch is performing well in all aspects.")
            for a in advice:
                st.write(a)

            if 'avg_humidity' in current_sum.index and pd.notna(current_sum['avg_humidity']):
                st.markdown("### Weather on Roast Day")
                st.write(f"💧 Humidity: {current_sum['avg_humidity']:.1f}%")
                st.write(f"🌡️ Temperature: {current_sum['avg_temp']:.1f}°C")
                diff = today_humidity - current_sum['avg_humidity']
                if abs(diff) > 5 and 'humidity_slope' in st.session_state:
                    adjust = st.session_state['humidity_slope'] * diff
                    st.info(f"💡 Humidity difference {diff:+.1f}%. Consider adjusting start gas by {adjust:+.1f}%.")

    # ============================= 视图 2 =============================
    elif view == "Statistical Analysis by Start Gas":
        st.subheader("📈 Statistical Analysis by Start Gas Value")
        if not available_gas:
            st.warning("No valid start gas values found.")
            st.stop()
        selected_gas = st.sidebar.selectbox("Select Start Gas (%) to analyze:", available_gas)
        group_df = master_sum[master_sum['start_gas_int'] == selected_gas].copy()
        st.write(f"**{len(group_df)} batches found with start gas = {selected_gas}%**")
        with st.expander("Show batch IDs"):
            st.write(group_df['batch_id'].tolist())

        st.markdown("### Summary Statistics")
        metrics = ['total_gas', 'dry_gas', 'mail_gas', 'dev_gas',
                   'total_efficiency', 'dry_efficiency', 'mail_efficiency', 'dev_efficiency', 'deviation']
        rows = []
        for m in metrics:
            vals = group_df[m].dropna()
            if vals.empty:
                continue
            baseline = ref_sum[m]
            pct = ((vals.mean() - baseline) / baseline * 100) if baseline != 0 else np.nan
            rows.append({'Metric': m, 'Mean': vals.mean(), 'Std': vals.std(),
                         'Min': vals.min(), 'Max': vals.max(),
                         'Baseline': baseline, 'Mean vs Baseline (%)': pct})
        st.dataframe(pd.DataFrame(rows).round(2), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Box(y=group_df['total_gas'], name=f'{selected_gas}% batches', boxmean='sd'))
            fig.add_hline(y=ref_sum['total_gas'], line_dash='dash', line_color='red', annotation_text='Baseline')
            fig.update_layout(title='Total Gas Distribution')
            st.plotly_chart(fig)
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Box(y=group_df['deviation'], name=f'{selected_gas}% batches', boxmean='sd'))
            fig.add_hline(y=ref_sum['deviation'], line_dash='dash', line_color='red', annotation_text='Baseline')
            fig.update_layout(title='Curve Deviation Distribution')
            st.plotly_chart(fig)

        st.markdown("### Average Temperature Curve")
        time_uniform = np.arange(0, 801)
        curves = []
        for bid in group_df['batch_id']:
            ts = master_ts[master_ts['batch_id'] == bid].sort_values('time_sec')
            f = interp1d(ts['time_sec'], ts['beantemp'], kind='linear', fill_value='extrapolate')
            curves.append(f(time_uniform))
        if curves:
            mean_c = np.mean(curves, axis=0)
            std_c = np.std(curves, axis=0)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_uniform, y=mean_c, name='Average'))
            fig.add_trace(
                go.Scatter(x=time_uniform, y=mean_c + std_c, mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=time_uniform, y=mean_c - std_c, mode='lines', fill='tonexty',
                                     fillcolor='rgba(0,0,255,0.2)', name='±1 Std'))
            fig.add_trace(go.Scatter(x=ref_ts['time_sec'], y=ref_ts['beantemp'], name='Baseline',
                                     line=dict(color='red', dash='dash')))
            st.plotly_chart(fig)

    # ============================= 视图 3 =============================
    elif view == "Full Roast Energy Analysis":
        st.subheader("🔥 Full Roast Energy Analysis")
        st.markdown("Gas usage is shown as **%·s** (gas setting × seconds). Not m³ or kWh.")
        st.info("Lower-left quadrant = best (low gas, low deviation).")

        fig_overview = go.Figure()
        fig_overview.add_trace(go.Scatter(
            x=master_sum["total_gas"], y=master_sum["deviation"],
            mode="markers", text=master_sum["batch_id"], name="Roasts",
            marker=dict(size=9, color="blue", opacity=0.65)
        ))
        fig_overview.update_layout(xaxis_title="Total Gas Used (%·s)", yaxis_title="Curve Deviation (°C·s)")
        st.plotly_chart(fig_overview, use_container_width=True)

        selected_energy_batch = st.selectbox("Select a batch for detailed gas-curve analysis:",
                                             master_sum["batch_id"].tolist())
        selected_summary = master_sum[master_sum["batch_id"] == selected_energy_batch].iloc[0]
        selected_ts = master_ts[master_ts["batch_id"] == selected_energy_batch].sort_values("time_sec")
        ref_ts_sorted = ref_ts.sort_values("time_sec")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total Gas Used (%·s)", f"{selected_summary['total_gas']:.0f}")
        with col_b:
            st.metric("Curve Deviation (°C·s)", f"{selected_summary['deviation']:.0f}")
        with col_c:
            st.metric("Starting Gas (%)",
                      f"{selected_summary['start_gas']:.0f}%" if pd.notna(selected_summary['start_gas']) else "Unknown")

        st.markdown("### Gas Curve Across Roast")
        fig_gas = go.Figure()
        fig_gas.add_trace(go.Scatter(x=ref_ts_sorted['time_sec'] / 60, y=ref_ts_sorted['gascontrol'],
                                     name="Reference gas", line=dict(color="red", dash="dash")))
        fig_gas.add_trace(go.Scatter(x=selected_ts['time_sec'] / 60, y=selected_ts['gascontrol'],
                                     name="Selected roast gas", line=dict(color="blue")))
        fig_gas.update_layout(xaxis_title="Time (minutes)", yaxis_title="Gas control (%)")
        st.plotly_chart(fig_gas, use_container_width=True)

        st.markdown("### Cumulative Gas Usage")
        selected_ts['cumulative_gas'] = selected_ts['gascontrol'].cumsum()
        ref_ts_sorted['cumulative_gas'] = ref_ts_sorted['gascontrol'].cumsum()
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=ref_ts_sorted['time_sec'] / 60, y=ref_ts_sorted['cumulative_gas'],
                                     name="Reference cumulative", line=dict(color="red", dash="dash")))
        fig_cum.add_trace(go.Scatter(x=selected_ts['time_sec'] / 60, y=selected_ts['cumulative_gas'],
                                     name="Selected roast cumulative", line=dict(color="blue")))
        fig_cum.update_layout(xaxis_title="Time (minutes)", yaxis_title="Cumulative Gas (%·s)")
        st.plotly_chart(fig_cum, use_container_width=True)

        st.markdown("### Gas Usage by Time Block")
        time_blocks = [(0, 300, "0–5 min"), (300, 600, "5–10 min"), (600, 900, "10–15 min"),
                       (900, selected_ts['time_sec'].max() + 1, "15+ min")]
        block_rows = []
        for start, end, label in time_blocks:
            sel_block = selected_ts[(selected_ts['time_sec'] >= start) & (selected_ts['time_sec'] < end)]
            ref_block = ref_ts_sorted[(ref_ts_sorted['time_sec'] >= start) & (ref_ts_sorted['time_sec'] < end)]
            sel_gas = sel_block['gascontrol'].sum()
            ref_gas = ref_block['gascontrol'].sum()
            diff = sel_gas - ref_gas
            pct = diff / ref_gas * 100 if ref_gas else np.nan
            block_rows.append(
                {"Time Block": label, "Selected Gas (%·s)": sel_gas, "Reference Gas (%·s)": ref_gas, "Difference": diff,
                 "Difference (%)": pct})
        st.dataframe(pd.DataFrame(block_rows).round(2), use_container_width=True)

        st.markdown("### Extra Gas vs Reference")
        selected_gas_interp = interp1d(selected_ts['time_sec'], selected_ts['gascontrol'], kind='linear',
                                       fill_value='extrapolate')
        gas_diff = selected_gas_interp(ref_ts_sorted['time_sec']) - ref_ts_sorted['gascontrol']
        fig_diff = go.Figure()
        fig_diff.add_trace(
            go.Scatter(x=ref_ts_sorted['time_sec'] / 60, y=gas_diff, mode='lines', name='Gas difference'))
        fig_diff.add_hline(y=0, line_dash='dash')
        fig_diff.update_layout(xaxis_title="Time (minutes)", yaxis_title="Gas Difference (% points)")
        st.plotly_chart(fig_diff, use_container_width=True)

    # ============================= 视图 4 =============================
    elif view == "Moisture & Density Analysis":
        st.subheader("💧 Moisture & Density Analysis")
        moisture_file = "moisture_density.csv"
        default_columns = ["measurement_id", "measurement_date", "coffee_name", "crop_year", "container",
                           "moisture_pct", "density_g_l", "notes"]

        if os.path.exists(moisture_file):
            moisture_df = pd.read_csv(moisture_file)
        else:
            moisture_df = pd.DataFrame(columns=default_columns)

        for col in default_columns:
            if col not in moisture_df.columns:
                moisture_df[col] = ""
        moisture_df = moisture_df.replace(["None", "none", ""], np.nan)
        moisture_df = moisture_df.dropna(subset=["measurement_date", "moisture_pct", "density_g_l"], how="all").copy()

        if moisture_df.empty:
            moisture_df = pd.DataFrame(columns=default_columns)
        else:
            if "measurement_id" not in moisture_df.columns or moisture_df["measurement_id"].isna().all():
                moisture_df["measurement_id"] = range(1, len(moisture_df) + 1)
        moisture_df.to_csv(moisture_file, index=False)

        st.markdown("## 1. Add New Measurement")
        with st.form("add_moisture_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                m_date = st.date_input("Measurement date", value=date.today())
                coffee_name = st.selectbox("Coffee", ["Passeio / Paz Coffee"])
            with col2:
                crop_year = st.text_input("Crop year", "2025")
                container = st.text_input("Container", "Container 1")
            with col3:
                moisture_pct = st.number_input("Moisture (%)", 0.0, 30.0, 10.0, 0.1)
                density_g_l = st.number_input("Density (g/L)", 0.0, 1000.0, 700.0, 1.0)
            notes = st.text_area("Notes")
            if st.form_submit_button("Save measurement"):
                next_id = int(moisture_df["measurement_id"].max()) + 1 if not moisture_df.empty else 1
                new_row = pd.DataFrame(
                    [{"measurement_id": next_id, "measurement_date": m_date.isoformat(), "coffee_name": coffee_name,
                      "crop_year": crop_year, "container": container, "moisture_pct": moisture_pct,
                      "density_g_l": density_g_l, "notes": notes}])
                moisture_df = pd.concat([moisture_df, new_row], ignore_index=True)
                moisture_df.to_csv(moisture_file, index=False)
                st.success("Saved!")
                st.rerun()

        st.markdown("## 2. Current Dataset")
        if moisture_df.empty:
            st.info("No data yet.")
            st.stop()
        moisture_df["measurement_date"] = pd.to_datetime(moisture_df["measurement_date"])
        edited_df = st.data_editor(moisture_df, use_container_width=True, num_rows="dynamic")
        if st.button("💾 Save edited table"):
            edited_df.to_csv(moisture_file, index=False)
            st.success("Saved!")
            st.rerun()

        st.markdown("## 3. Delete Measurement")
        delete_options = [f"ID {row['measurement_id']} | {row['measurement_date'].date()} | {row['coffee_name']}" for
                          _, row in moisture_df.iterrows()]
        selected_del = st.selectbox("Select to delete:", delete_options)
        if st.button("🗑️ Delete"):
            del_id = int(selected_del.split("|")[0].replace("ID", "").strip())
            moisture_df = moisture_df[moisture_df["measurement_id"] != del_id]
            moisture_df.to_csv(moisture_file, index=False)
            st.success("Deleted!")
            st.rerun()

        st.markdown("## 4. Latest Physical Data")
        latest = moisture_df.sort_values("measurement_date").iloc[-1]
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Latest Moisture (%)", f"{latest['moisture_pct']:.1f}%")
        with col_b:
            st.metric("Latest Density (g/L)", f"{latest['density_g_l']:.0f} g/L")
        with col_c:
            st.metric("Date", latest["measurement_date"].date().isoformat())

        st.markdown("## 5. Link to Roasts")
        analysis_df = master_sum.copy()
        analysis_df["linked_moisture_pct"] = latest["moisture_pct"]
        analysis_df["linked_density_g_l"] = latest["density_g_l"]
        st.dataframe(
            analysis_df[["batch_id", "total_gas", "deviation", "linked_moisture_pct", "linked_density_g_l"]].head(50))

        st.markdown("## 6. Correlation (if enough variation)")
        valid_corr = analysis_df[["total_gas", "linked_moisture_pct", "linked_density_g_l"]].dropna()
        if len(valid_corr) >= 3 and valid_corr["linked_density_g_l"].nunique() > 1:
            corr_dens = valid_corr["linked_density_g_l"].corr(valid_corr["total_gas"])
            corr_mois = valid_corr["linked_moisture_pct"].corr(valid_corr["total_gas"])
            st.metric("Density vs Total Gas Corr", f"{corr_dens:.3f}")
            st.metric("Moisture vs Total Gas Corr", f"{corr_mois:.3f}")

    # ============================= 视图 5 =============================
    elif view == "AI Roast Prediction":
        st.subheader("🤖 AI Roast Prediction (Early Curve Based)")
        st.markdown("Uses the first X minutes of the roast to predict final curve deviation and total gas usage.")
        try:
            from sklearn.ensemble import RandomForestRegressor, IsolationForest
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_absolute_error, r2_score
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
        except ImportError:
            st.error("scikit-learn not installed. Run: pip install scikit-learn")
            st.stop()

        checkpoint_min = st.slider("Use first X minutes to predict:", 3, 10, 5)
        checkpoint_sec = checkpoint_min * 60


        def safe_interp(df, x_col, y_col, target_x):
            df = df[[x_col, y_col]].dropna().sort_values(x_col)
            if len(df) < 2: return np.nan
            if target_x < df[x_col].min() or target_x > df[x_col].max(): return np.nan
            try:
                return float(interp1d(df[x_col], df[y_col], kind='linear', fill_value='extrapolate')(target_x))
            except:
                return np.nan


        def sum_gas_block(df, start, end):
            block = df[(df['time_sec'] >= start) & (df['time_sec'] < end)]
            return block['gascontrol'].sum() if not block.empty else np.nan


        rows = []
        for bid, ts in master_ts.groupby("batch_id"):
            ts = ts.sort_values('time_sec').dropna(subset=['time_sec', 'gascontrol', 'beantemp'])
            if ts.empty: continue
            early = ts[ts['time_sec'] <= checkpoint_sec]
            if early.empty: continue
            temp_start = safe_interp(ts, 'time_sec', 'beantemp', 0)
            temp_check = safe_interp(ts, 'time_sec', 'beantemp', checkpoint_sec)
            gas_start = safe_interp(ts, 'time_sec', 'gascontrol', 0)
            gas_check = safe_interp(ts, 'time_sec', 'gascontrol', checkpoint_sec)
            temp_rise = temp_check - temp_start if pd.notna(temp_start) and pd.notna(temp_check) else np.nan
            temp_slope = temp_rise / checkpoint_sec if pd.notna(temp_rise) else np.nan
            gas_diff = early['gascontrol'].diff().abs()
            row = {
                "batch_id": bid,
                "gas_start": gas_start,
                "gas_at_checkpoint": gas_check,
                "gas_until_checkpoint": early['gascontrol'].sum(),
                "avg_gas_until_checkpoint": early['gascontrol'].mean(),
                "max_gas_until_checkpoint": early['gascontrol'].max(),
                "min_gas_until_checkpoint": early['gascontrol'].min(),
                "std_gas_until_checkpoint": early['gascontrol'].std(),
                "gas_change_count": (gas_diff > 1).sum(),
                "gas_change_amount": gas_diff.sum(),
                "temp_start": temp_start,
                "temp_at_checkpoint": temp_check,
                "temp_rise_until_checkpoint": temp_rise,
                "temp_slope_until_checkpoint": temp_slope,
                "avg_temp_until_checkpoint": early['beantemp'].mean(),
                "max_temp_until_checkpoint": early['beantemp'].max(),
                "min_temp_until_checkpoint": early['beantemp'].min(),
                "std_temp_until_checkpoint": early['beantemp'].std(),
                "gas_0_1min": sum_gas_block(ts, 0, 60),
                "gas_1_3min": sum_gas_block(ts, 60, 180),
                "gas_3_5min": sum_gas_block(ts, 180, 300),
                "gas_5_7min": sum_gas_block(ts, 300, 420),
                "gas_7_10min": sum_gas_block(ts, 420, 600),
            }
            rows.append(row)
        ai_df = pd.DataFrame(rows)
        if ai_df.empty:
            st.warning("No data available for AI features.")
            st.stop()
        ai_df = ai_df.merge(master_sum[['batch_id', 'total_gas', 'deviation', 'start_gas']], on='batch_id', how='inner')
        ai_df['gas_start'] = ai_df['gas_start'].fillna(ai_df['start_gas'])
        feature_cols = [c for c in ai_df.columns if c not in ['batch_id', 'total_gas', 'deviation', 'start_gas']]

        model_df = ai_df[['batch_id'] + feature_cols + ['total_gas', 'deviation']].dropna()
        if len(model_df) < 8:
            st.warning(f"Only {len(model_df)} batches available. Need at least 8 for reliable ML.")
            st.stop()

        X = model_df[feature_cols]
        y_dev = model_df['deviation']
        y_gas = model_df['total_gas']

        if len(model_df) >= 20:
            X_train, X_test, y_dev_train, y_dev_test, y_gas_train, y_gas_test = train_test_split(
                X, y_dev, y_gas, test_size=0.25, random_state=42)
        else:
            X_train, X_test, y_dev_train, y_dev_test, y_gas_train, y_gas_test = X, X, y_dev, y_dev, y_gas, y_gas

        dev_model = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2)
        gas_model = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2)
        dev_model.fit(X_train, y_dev_train)
        gas_model.fit(X_train, y_gas_train)

        dev_mae = mean_absolute_error(y_dev_test, dev_model.predict(X_test))
        gas_mae = mean_absolute_error(y_gas_test, gas_model.predict(X_test))
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Deviation MAE (°C·s)", f"{dev_mae:.0f}")
            st.metric("Deviation R²", f"{r2_score(y_dev_test, dev_model.predict(X_test)):.2f}")
        with col2:
            st.metric("Total Gas MAE (%·s)", f"{gas_mae:.0f}")
            st.metric("Total Gas R²", f"{r2_score(y_gas_test, gas_model.predict(X_test)):.2f}")

        st.markdown("## Predict a Specific Batch")
        sel_batch = st.selectbox("Select batch", model_df['batch_id'].tolist())
        sel_row = model_df[model_df['batch_id'] == sel_batch].iloc[0]
        sel_X = pd.DataFrame([sel_row[feature_cols].values], columns=feature_cols)
        pred_dev = dev_model.predict(sel_X)[0]
        pred_gas = gas_model.predict(sel_X)[0]
        actual_dev = sel_row['deviation']
        actual_gas = sel_row['total_gas']
        st.metric("Predicted Curve Deviation", f"{pred_dev:.0f}", delta=f"{pred_dev - actual_dev:+.0f}")
        st.metric("Predicted Total Gas", f"{pred_gas:.0f}", delta=f"{pred_gas - actual_gas:+.0f}")

        st.markdown("## Feature Importance")
        imp_df = pd.DataFrame({"feature": feature_cols, "importance": dev_model.feature_importances_}).sort_values(
            "importance", ascending=False).head(10)
        fig_imp = go.Figure(go.Bar(x=imp_df["importance"], y=imp_df["feature"], orientation='h'))
        fig_imp.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("## Anomaly Detection (Isolation Forest)")
        if len(model_df) >= 10:
            iso = IsolationForest(contamination=min(0.15, 5 / len(model_df)), random_state=42)
            labels = iso.fit_predict(X)
            anomaly_scores = iso.decision_function(X)
            anomaly_df = model_df.copy()
            anomaly_df['anomaly'] = labels
            anomaly_df['anomaly_score'] = anomaly_scores

            anomaly_only = anomaly_df[anomaly_df['anomaly'] == -1]
            if len(anomaly_only) > 0:
                st.dataframe(anomaly_only[['batch_id', 'total_gas', 'deviation', 'anomaly', 'anomaly_score']],
                             use_container_width=True)
                st.caption(f"Found {len(anomaly_only)} anomalous batch(es). Lower anomaly score = more unusual.")
            else:
                st.success("No anomalous batches detected.")

            fig_anomaly = go.Figure()
            fig_anomaly.add_trace(go.Scatter(
                x=anomaly_df[anomaly_df['anomaly'] == 1]['total_gas'],
                y=anomaly_df[anomaly_df['anomaly'] == 1]['deviation'],
                mode='markers', name='Normal', marker=dict(color='blue', size=8)
            ))
            fig_anomaly.add_trace(go.Scatter(
                x=anomaly_df[anomaly_df['anomaly'] == -1]['total_gas'],
                y=anomaly_df[anomaly_df['anomaly'] == -1]['deviation'],
                mode='markers', name='Anomaly', marker=dict(color='red', size=10, symbol='x')
            ))
            fig_anomaly.update_layout(xaxis_title="Total Gas Used (%·s)", yaxis_title="Curve Deviation (°C·s)")
            st.plotly_chart(fig_anomaly, use_container_width=True)
        else:
            st.warning("Need ≥10 batches for anomaly detection.")

else:
    st.stop()