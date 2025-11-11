import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
from dateutil import parser
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------
st.set_page_config(page_title="Body Temperature Analysis", layout="wide")

st.title("🌡️ Body Temperature Analysis and Fever Prediction (Last 24 Hours)")

st.markdown("""
This app analyzes your **body temperature data** from a CSV file and predicts if a fever may occur in the near future.  
It supports Chinese and English CSV files and various date/time formats.  

**Accepted formats:**
- **Date:** `2025/11/09`, `2025-11-09`, `20251109`, etc.  
- **Time:** `08:00`, `1541`, `115`, `15`, `上午8:00`, etc.  
""")

# ---------------------------------------------------
# File Upload
# ---------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"])

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def normalize_str(s):
    """Normalize mixed-format date/time strings."""
    s = str(s)
    s = s.replace("：", ":").replace("，", ",").replace("／", "/")
    s = s.replace("上午", "AM ").replace("下午", "PM ").replace("早上", "AM ").replace("晚上", "PM ")
    s = s.replace("年", "-").replace("月", "-").replace("日", " ")
    return s.strip()

def parse_date_time_str(date_val, time_val):
    """Parse flexible date/time formats into datetime objects."""
    if pd.isna(date_val) and pd.isna(time_val):
        return pd.NaT
    ds = str(date_val).strip() if not pd.isna(date_val) else ""
    ts = str(time_val).strip() if not pd.isna(time_val) else ""
    ds_norm = normalize_str(ds)
    ts_norm = normalize_str(ts)

    # Case 1: Numeric date (YYYYMMDD) and numeric time (HHMM)
    if re.fullmatch(r"\d{8}", ds_norm):
        try:
            d = datetime.strptime(ds_norm, "%Y%m%d")
            if re.fullmatch(r"\d{1,4}", ts_norm):
                t_padded = ts_norm.zfill(4)
                hh = int(t_padded[:2])
                mm = int(t_padded[2:])
                return datetime.combine(d.date(), datetime.min.time().replace(hour=hh, minute=mm))
        except Exception:
            pass

    # Case 2: Numeric time with non-numeric date
    if re.fullmatch(r"\d{1,4}", ts_norm):
        try:
            t_padded = ts_norm.zfill(4)
            hh = int(t_padded[:2])
            mm = int(t_padded[2:])
            d = parser.parse(ds_norm, fuzzy=True)
            return datetime.combine(d.date(), datetime.min.time().replace(hour=hh, minute=mm))
        except Exception:
            pass

    # Case 3: General parser
    try:
        return parser.parse(f"{ds_norm} {ts_norm}", fuzzy=True)
    except Exception:
        return pd.NaT


# ---------------------------------------------------
# Process Uploaded File
# ---------------------------------------------------
if uploaded_file is not None:
    try_encodings = ["utf-8-sig", "utf-8", "big5", "cp950"]
    df_read = None

    for enc in try_encodings:
        try:
            uploaded_file.seek(0)
            df_read = pd.read_csv(uploaded_file, encoding=enc, sep=None, engine="python")
            break
        except Exception:
            continue

    if df_read is None:
        st.error("❌ Failed to read the CSV file. Please re-save it as UTF-8 or Big5.")
    else:
        df_read.columns = [str(c).strip() for c in df_read.columns]

        # Auto-detect columns
        def detect_columns(columns):
            date_keywords = ["date", "日期", "day"]
            time_keywords = ["time", "時間", "時刻"]
            temp_keywords = ["temp", "temperature", "體溫", "溫度", "bt"]

            date_col = time_col = temp_col = None
            lowered = [c.lower() for c in columns]

            for i, col in enumerate(lowered):
                if date_col is None and any(kw in col for kw in date_keywords):
                    date_col = columns[i]
                if time_col is None and any(kw in col for kw in time_keywords):
                    time_col = columns[i]
                if temp_col is None and any(kw in col for kw in temp_keywords):
                    temp_col = columns[i]

            # Fallbacks
            if date_col is None and len(columns) >= 1:
                date_col = columns[0]
            if time_col is None and len(columns) >= 2:
                time_col = columns[1]
            if temp_col is None and len(columns) >= 3:
                temp_col = columns[2]

            return date_col, time_col, temp_col

        date_col, time_col, temp_col = detect_columns(df_read.columns)

        if date_col is None or time_col is None or temp_col is None:
            st.error("❌ Could not detect 'Date', 'Time', or 'Temperature' columns.")
        else:
            df = df_read[[date_col, time_col, temp_col]].rename(
                columns={date_col: "Date", time_col: "Time", temp_col: "Temperature"}
            )

            # Convert to datetime
            df["DateTime"] = df.apply(lambda r: parse_date_time_str(r["Date"], r["Time"]), axis=1)
            df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")

            # Drop invalid rows
            df = df.dropna(subset=["DateTime", "Temperature"])
            df = df.sort_values("DateTime").reset_index(drop=True)

            if df.empty:
                st.error("⚠️ No valid data found after cleaning. Check your CSV format.")
            else:
                # Format display
                df["DateStr"] = df["DateTime"].dt.strftime("%b %d, %Y")
                df["TimeStr"] = df["DateTime"].dt.strftime("%I:%M %p")

                preview_df = df[["DateStr", "TimeStr", "Temperature"]].rename(
                    columns={"DateStr": "Date", "TimeStr": "Time"}
                )

                # -----------------------
                # Display Data Preview
                # -----------------------
                st.subheader("📋 Data Preview")
                st.dataframe(preview_df, hide_index=True, use_container_width=True)

                # -----------------------
                # Plot Temperature Trend
                # -----------------------
                st.subheader("📈 Temperature Trend (Last 24 Hours)")
                plt.figure(figsize=(10, 4))
                plt.plot(df["DateTime"], df["Temperature"], marker="o", linestyle="-", color="steelblue")
                plt.axhline(y=38.0, color="red", linestyle="--", label="Fever threshold (38°C)")
                plt.xlabel("Time")
                plt.ylabel("Temperature (°C)")
                plt.title("Body Temperature Trend")
                plt.xticks(rotation=45)
                plt.legend()
                plt.grid(True)
                st.pyplot(plt)

                # -----------------------
                # Simple Linear Prediction
                # -----------------------
                if len(df) >= 3:
                    st.subheader("🤖 Fever Prediction (1-hour Trend)")
                    df["Timestamp"] = df["DateTime"].astype(np.int64) // 10**9
                    X = df[["Timestamp"]]
                    y = df["Temperature"]
                    model = LinearRegression().fit(X, y)
                    future_time = df["Timestamp"].max() + 3600  # +1 hour
                    predicted_temp = model.predict([[future_time]])[0]
                    st.write(f"Predicted temperature in 1 hour: **{predicted_temp:.2f}°C**")
                    if predicted_temp >= 38:
                        st.warning("⚠️ High risk of fever within the next hour.")
                    else:
                        st.success("✅ Low fever risk within the next hour.")

else:
    st.info("⬆️ Please upload a CSV file to begin analysis.")











