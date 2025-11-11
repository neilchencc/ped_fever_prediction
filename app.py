import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import joblib

# ---------------------------------------------------
# Title & Introduction
# ---------------------------------------------------
st.title("📈 Body Temperature Analysis Tool (Last 24h Prediction)")

st.markdown("""
**App Description:**  
This app uses historical body temperature records from **08:00 of the previous day to 08:00 of the last day** 
to predict whether a fever may occur in the coming days.

**Input Options:**  
1. Upload a CSV file with three columns: `Date`, `Time`, `Temperature`  
2. Manual entry: edit temperatures directly in the table below.
""")

# ----------------------
# Input Method
# ----------------------
input_method = st.radio("Select input method:", ["Upload CSV file", "Manual Entry"])
df = pd.DataFrame(columns=["Date", "Time", "Temperature"])

# ----------------------
# CSV Upload
# ----------------------
if input_method == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload CSV with columns: Date, Time, Temperature", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = ["Date", "Time", "Temperature"][:len(df.columns)]
            df.columns = [c.strip() for c in df.columns]

            # 嘗試轉換資料，若有錯誤則刪除該筆
            def safe_parse_datetime(row):
                try:
                    return datetime.strptime(str(int(row["Date"])) + f"{int(row['Time']):04d}", "%Y%m%d%H%M")
                except Exception:
                    return np.nan

            df["DateTime"] = df.apply(safe_parse_datetime, axis=1)
            df = df.dropna(subset=["DateTime", "Temperature"])  # 移除時間或溫度無效者
            df = df[pd.to_numeric(df["Temperature"], errors="coerce").notnull()]  # 移除非數字溫度
            df["Temperature"] = df["Temperature"].astype(float)
            df = df.sort_values("DateTime").reset_index(drop=True)

        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            df = pd.DataFrame(columns=["Date", "Time", "Temperature"])

# ----------------------
# Manual Entry
# ----------------------
elif input_method == "Manual Entry":
    st.subheader("Manual Data Entry (editable table)")
    
    day1_times = [f"{h:02d}:00" for h in range(8,24)]
    day2_times = [f"{h:02d}:00" for h in range(0,8)]
    all_times = [("Day1", t) for t in day1_times] + [("Day2", t) for t in day2_times]

    manual_df = pd.DataFrame(all_times, columns=["Date", "Time"])
    manual_df["Temperature"] = np.nan

    edited_df = st.data_editor(manual_df, num_rows="dynamic", use_container_width=True)
    edited_df = edited_df.dropna(subset=["Temperature"])
    if not edited_df.empty:
        df = edited_df.copy()
        base_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        df["DateTime"] = df.apply(
            lambda row: base_date + timedelta(days=int(row["Date"][-1])-1,
                                              hours=int(row["Time"][:2]),
                                              minutes=int(row["Time"][3:])), axis=1)
        df = df.sort_values("DateTime").reset_index(drop=True)

# ----------------------
# Proceed if data exists
# ----------------------
if not df.empty:
    last_date = df["DateTime"].dt.date.max()
    end_time = datetime.combine(last_date, datetime.min.time()) + timedelta(hours=8)
    start_time = end_time - timedelta(hours=24)

    df_24h = df[(df["DateTime"] >= start_time) & (df["DateTime"] <= end_time)].copy()
    df_24h = df_24h.reset_index(drop=True)

    if df_24h.empty:
        st.warning("No data available in the last 24 hours (08:00 → 08:00).")
    else:
        df_24h["Hours"] = (df_24h["DateTime"] - df_24h["DateTime"].min()).dt.total_seconds() / 3600

        # 計算特徵（僅供模型使用，不顯示）
        max_bt = df_24h["Temperature"].max()
        min_bt = df_24h["Temperature"].min()
        mean_bt = df_24h["Temperature"].mean()
        std_bt = df_24h["Temperature"].std()
        X = df_24h["Hours"].values.reshape(-1,1)
        y = df_24h["Temperature"].values
        model_lr = LinearRegression().fit(X,y)
        slope = model_lr.coef_[0]
        last_time = df_24h["Hours"].max()
        last_8h = df_24h[df_24h["Hours"] >= last_time-8]
        max_last8 = last_8h["Temperature"].max()
        range_bt = max_bt - min_bt
        diff_last8_allmax = max_last8 - max_bt

        features = [max_bt, min_bt, mean_bt, std_bt, slope, range_bt, max_last8, diff_last8_allmax]

        # 模型預測
        try:
            scaler = joblib.load("scaler.pkl")
            svm_model = joblib.load("svm_model.pkl")
            features_scaled = scaler.transform(np.array(features).reshape(1,-1))
            if hasattr(svm_model, "predict_proba"):
                pred_prob = svm_model.predict_proba(features_scaled)[0][1]
            else:
                pred_prob = svm_model.decision_function(features_scaled)[0]

            threshold = 0.5
            st.subheader("🤖 Prediction Result")
            if pred_prob >= threshold:
                st.success(f"Prediction: Fever likely (Score/Probability={pred_prob:.3f} ≥ {threshold})")
            else:
                st.info(f"Prediction: No fever expected (Score/Probability={pred_prob:.3f} < {threshold})")

        except FileNotFoundError as e:
            st.error(f"Missing model file: {e.filename}")
        except Exception as e:
            st.error(f"Error loading scaler or model: {e}")

        # ----------------------
        # Data Preview (Last 24h)
        # ----------------------
        st.write("### 🧾 Data Preview (Last 24h)")
        df_preview = df_24h.copy()
        df_preview["Date"] = df_preview["DateTime"].dt.strftime("%Y-%m-%d")
        df_preview["Time"] = df_preview["DateTime"].dt.strftime("%H:%M")
        st.dataframe(df_preview[["Date", "Time", "Temperature"]])

        # ----------------------
        # Temperature Trend Plot
        # ----------------------
        st.subheader("📉 Temperature Trend (Last 24h)")
        fig, ax = plt.subplots()
        ax.plot(df_24h["DateTime"], df_24h["Temperature"], marker='o', label="Temperature")
        ax.axhline(y=38, color='darkred', linestyle='--', linewidth=2, label="Fever Threshold (38°C)")
        ax.set_ylim(35, 43)
        ax.set_xlabel("Time")
        ax.set_ylabel("Temperature (°C)")
        plt.xticks(rotation=45, ha='right')  # 逆時針45度旋轉時間
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

else:
    st.info("⬆️ Please upload a CSV file or fill in temperatures manually to begin analysis.")














