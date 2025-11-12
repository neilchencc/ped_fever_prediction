import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import joblib
from dateutil import parser
import chardet

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

**Note:**  
Date formats can be `20251110`, `2025/11/10`, `Nov 10, 2025`, `2025-11-10`, or `11/10`.  
Time formats can be `17:00`, `05:00 pm`, `1700`, `0`, `5`, `130`, etc.
""")

# ----------------------
# Input Method
# ----------------------
input_method = st.radio("Select input method:", ["Upload CSV file", "Manual Entry"])
df = pd.DataFrame(columns=["Date", "Time", "Temperature"])

# ----------------------
# Helper Function for Robust Date/Time Parsing
# ----------------------
def parse_datetime(date_str, time_str):
    time_str = str(time_str).strip()

    if time_str in ["", "nan", "NaN"]:
        time_str = "00:00"
    elif time_str.isdigit():
        time_str = time_str.zfill(4)
        time_str = time_str[:2] + ":" + time_str[2:]

    try:
        dt_date = parser.parse(str(date_str), dayfirst=False, fuzzy=True)
    except Exception:
        raise ValueError(f"Unrecognized date format: {date_str}")

    try:
        dt_time = parser.parse(time_str, fuzzy=True).time()
    except Exception:
        raise ValueError(f"Unrecognized time format: {time_str}")

    return datetime.combine(dt_date.date(), dt_time)

# ----------------------
# CSV Upload
# ----------------------
if input_method == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload CSV with at least three columns: Date, Time, Temperature", type=["csv"])
    if uploaded_file is not None:
        try:
            # 偵測編碼
            uploaded_file.seek(0)
            rawdata = uploaded_file.read()
            result = chardet.detect(rawdata)
            encoding = result['encoding']

            if encoding is None:
                encoding = "big5"

            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding)

            if df.empty:
                st.error("Uploaded CSV is empty.")
            else:
                # 只保留前三欄並重新命名
                df = df.iloc[:, :3]
                df.columns = ["Date", "Time", "Temperature"]
                df = df.dropna(subset=["Temperature"])  # 刪除溫度空值列

                # 清理數據
                df["Date"] = df["Date"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
                df["Time"] = df["Time"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

                # 解析日期時間
                df["DateTime"] = df.apply(lambda row: parse_datetime(row["Date"], row["Time"]), axis=1)
                df = df.sort_values("DateTime").reset_index(drop=True)

        except pd.errors.EmptyDataError:
            st.error("Uploaded CSV is empty or unreadable.")
        except UnicodeDecodeError:
            st.error("Failed to decode CSV. Please make sure it's encoded in UTF-8, GBK, or Big5.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# ----------------------
# Manual Entry
# ----------------------
elif input_method == "Manual Entry":
    st.subheader("Manual Data Entry (editable table)")
    
    day1_times = [f"{h:02d}:00" for h in range(8,24)]
    day2_times = [f"{h:02d}:00" for h in range(0,8)]
    all_times = [("Day1", t) for t in day1_times] + [("Day2", t) for t in day2_times]

    manual_df = pd.DataFrame(all_times, columns=["Day", "Time"])
    manual_df["Temperature"] = np.nan

    edited_df = st.data_editor(manual_df, num_rows="dynamic", use_container_width=True)
    edited_df = edited_df.dropna(subset=["Temperature"])

    if not edited_df.empty:
        df = edited_df.copy()
        today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        df["DateTime"] = df.apply(lambda row: (
            today - timedelta(days=1) if row["Day"]=="Day1" else today
        ) + timedelta(hours=int(row["Time"][:2]), minutes=int(row["Time"][3:])), axis=1)
        df = df.sort_values("DateTime").reset_index(drop=True)

# ----------------------
# Proceed if Data Exists
# ----------------------
if not df.empty:
    last_date = df["DateTime"].dt.date.max()
    end_time = datetime.combine(last_date, datetime.min.time()) + timedelta(hours=8)
    start_time = end_time - timedelta(hours=24)
    df_24h = df[(df["DateTime"] >= start_time) & (df["DateTime"] <= end_time)].copy().reset_index(drop=True)

    if df_24h.empty:
        st.warning("No data available in the last 24 hours (08:00 → 08:00).")
    else:
        df_24h["Hours"] = (df_24h["DateTime"] - df_24h["DateTime"].min()).dt.total_seconds()/3600

        # ---------------------- Features ----------------------
        max_bt = df_24h["Temperature"].max()
        min_bt = df_24h["Temperature"].min()
        mean_bt = df_24h["Temperature"].mean()
        std_bt = df_24h["Temperature"].std()

        X = df_24h["Hours"].values.reshape(-1,1)
        y = df_24h["Temperature"].values
        slope = LinearRegression().fit(X, y).coef_[0]

        last_time = df_24h["Hours"].max()
        last_8h = df_24h[df_24h["Hours"] >= last_time-8]
        max_last8 = last_8h["Temperature"].max()
        range_bt = max_bt - min_bt
        diff_last8_allmax = max_last8 - max_bt

        features = [max_bt, min_bt, mean_bt, std_bt, slope, range_bt, max_last8, diff_last8_allmax]

        # ---------------------- Model Prediction ----------------------
        try:
            scaler = joblib.load("scaler.pkl")
            svm_model = joblib.load("svm_model.pkl")
            features_scaled = scaler.transform(np.array(features).reshape(1,-1))

            st.subheader("🤖 Prediction Result")
            if hasattr(svm_model, "predict_proba"):
                pred_prob = svm_model.predict_proba(features_scaled)[0][1]
            else:
                pred_prob = svm_model.decision_function(features_scaled)[0]

            threshold = 0.5
            if pred_prob >= threshold:
                st.success(f"Prediction: Fever likely (Score/Probability={pred_prob:.3f} ≥ {threshold})")
            else:
                st.info(f"Prediction: No fever expected (Score/Probability={pred_prob:.3f} < {threshold})")

        except FileNotFoundError as e:
            st.error(f"Missing model file: {e.filename}")
        except Exception as e:
            st.error(f"Error loading scaler or model: {e}")

        # ---------------------- Data Preview ----------------------
        st.subheader("🧾 Data Preview (Last 24h)")
        df_preview = df_24h.copy()
        df_preview["Date"] = df_preview["DateTime"].dt.strftime("%Y-%m-%d")
        df_preview["Time"] = df_preview["DateTime"].dt.strftime("%H:%M")
        st.dataframe(df_preview[["Date", "Time", "Temperature"]])

        # ---------------------- Temperature Trend ----------------------
        st.subheader("📉 Temperature Trend (Last 24h)")
        fig, ax = plt.subplots()
        ax.plot(df_24h["DateTime"], df_24h["Temperature"], marker='o', label="Temperature")
        ax.axhline(y=38, color='darkred', linestyle='--', linewidth=2, label="Fever Threshold (38°C)")
        ax.set_ylim(35, 43)
        ax.set_xlabel("Time")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(True)
        ax.legend()
        plt.xticks(rotation=45, ha='left')
        st.pyplot(fig)

else:
    st.info("⬆️ Please upload a CSV file or fill in temperatures manually to begin analysis.")




