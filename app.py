import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import joblib
from dateutil import parser
import re
import matplotlib.dates as mdates

# -----------------------
# Helper functions
# -----------------------
def detect_columns(columns):
    """Return (date_col, time_col, temp_col) by fuzzy matching column names (supports English & common Chinese)."""
    date_keywords = ["date", "日期", "day"]
    time_keywords = ["time", "時間", "時刻", "時", "時分"]
    temp_keywords = ["temp", "temperature", "體溫", "溫度", "t"]

    date_col = time_col = temp_col = None
    lowered = [str(c).strip().lower() for c in columns]

    for i, col in enumerate(lowered):
        for kw in date_keywords:
            if kw in col:
                date_col = columns[i]
                break
        for kw in time_keywords:
            if kw in col and time_col is None:
                time_col = columns[i]
                break
        for kw in temp_keywords:
            if kw in col and temp_col is None:
                temp_col = columns[i]
                break

    # fallback to first/second/third if not found
    if date_col is None and len(columns) >= 1:
        date_col = columns[0]
    if time_col is None and len(columns) >= 2:
        time_col = columns[1]
    if temp_col is None and len(columns) >= 3:
        temp_col = columns[2]

    return date_col, time_col, temp_col

def normalize_chinese_datetime_text(s: str) -> str:
    """Normalize common Chinese/time variants to improve parsing success."""
    if pd.isna(s):
        return ""
    s = str(s).strip()
    # Replace fullwidth colon/digits
    s = s.replace("：", ":").replace("；", ";")
    # Convert Chinese AM/PM
    s = s.replace("上午", "AM ").replace("下午", "PM ").replace("早上", "AM ").replace("晚上", "PM ")
    s = s.replace("中午", "12:00 PM ")
    # Remove Chinese words like '日','號' (keep numbers), keep '月' as separator
    s = re.sub(r"[日號]", " ", s)
    # Replace '年'/'月' with separators if present like "2025年11月12日"
    s = s.replace("年", "-").replace("月", "-")
    # Remove extraneous whitespace and punctuation
    s = re.sub(r"[／/]+", "-", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def try_parse_datetime(date_str, time_str):
    """Attempt several strategies to parse date/time into a datetime. Return pd.NaT on failure."""
    try:
        ds = normalize_chinese_datetime_text(date_str)
        ts = normalize_chinese_datetime_text(time_str)

        # If time looks like HHMM (e.g., 0830), insert colon
        m = re.fullmatch(r"(\d{3,4})", ts)
        if m:
            t = m.group(1)
            if len(t) == 3:
                ts = f"{t[0]}:{t[1:]}"
            else:
                ts = f"{t[:2]}:{t[2:]}"

        # Some CSVs have combined datetime in one column
        combined = f"{ds} {ts}".strip()
        if combined:
            try:
                return parser.parse(combined, fuzzy=True)
            except Exception:
                pass

        # Try parsing date first then attach parsed time if possible
        try:
            d = parser.parse(ds, fuzzy=True, default=datetime.now())
            try:
                t = parser.parse(ts, fuzzy=True, default=d)
                # If time string had only time, parser.parse returns today's date — replace date portion with d's date
                if any(ch.isdigit() for ch in ts):
                    return datetime.combine(d.date(), t.time())
                return d
            except Exception:
                return d
        except Exception:
            pass

    except Exception:
        pass

    return pd.NaT

# -----------------------
# Streamlit UI (English)
# -----------------------
st.set_page_config(page_title="Body Temperature Analysis (Last 24h)", layout="wide")
st.title("📈 Body Temperature Analysis Tool (Last 24h Prediction)")
st.markdown(
    "Upload a CSV file or enter data manually. The app uses temperature records from **08:00 of the previous day to 08:00 of the current day** "
    "to predict whether a fever may occur."
)

input_method = st.radio("Select input method:", ["Upload CSV file", "Manual Entry"])
df = pd.DataFrame(columns=["Date", "Time", "Temperature"])

# -----------------------
# CSV Upload handling
# -----------------------
if input_method == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload CSV (columns: Date, Time, Temperature). Chinese content is allowed.", type=["csv"])
    if uploaded_file is not None:
        # Try common encodings
        try_encodings = ["utf-8-sig", "utf-8", "big5", "cp950"]
        df_read = None
        for enc in try_encodings:
            try:
                uploaded_file.seek(0)
                df_read = pd.read_csv(uploaded_file, encoding=enc)
                break
            except Exception:
                continue

        if df_read is None:
            st.error("Failed to read CSV file with common encodings (utf-8, utf-8-sig, big5, cp950). Please re-save with UTF-8 or Big5 encoding.")
        else:
            # Normalize columns and detect which are date/time/temp
            df_read.columns = [c.strip() for c in df_read.columns]
            date_col, time_col, temp_col = detect_columns(list(df_read.columns))

            # Ensure columns exist
            if date_col is None or time_col is None or temp_col is None:
                st.error("Could not detect Date / Time / Temperature columns. Please ensure your CSV has these columns (names can be in English or Chinese).")
            else:
                # Keep only relevant columns and rename
                df = df_read[[date_col, time_col, temp_col]].rename(columns={date_col: "Date", time_col: "Time", temp_col: "Temperature"}).copy()

                # Parse datetimes with robust routines
                df["DateTime"] = df.apply(lambda r: try_parse_datetime(r["Date"], r["Time"]), axis=1)

                # Remove rows with invalid DateTime or Temperature
                df = df.dropna(subset=["DateTime", "Temperature"])
                df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
                df = df.dropna(subset=["Temperature"])
                df["Temperature"] = df["Temperature"].astype(float)

                # Sort by DateTime
                df = df.sort_values("DateTime").reset_index(drop=True)

# -----------------------
# Manual entry
# -----------------------
elif input_method == "Manual Entry":
    st.subheader("Manual Data Entry (editable table)")
    st.caption("Default times: Day1 08:00 → 23:00 ; Day2 00:00 → 07:00. Edit or add rows as needed.")
    day1_times = [f"{h:02d}:00" for h in range(8, 24)]
    day2_times = [f"{h:02d}:00" for h in range(0, 8)]
    all_times = [("Day1", t) for t in day1_times] + [("Day2", t) for t in day2_times]

    manual_df = pd.DataFrame(all_times, columns=["Date", "Time"])
    manual_df["Temperature"] = np.nan

    edited_df = st.data_editor(manual_df, num_rows="dynamic", use_container_width=True)
    edited_df = edited_df.dropna(subset=["Temperature"])
    if not edited_df.empty:
        # Map Day1/Day2 into concrete datetimes using today's date as anchor
        base_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        def day_time_to_dt(row):
            day = row["Date"]
            t = row["Time"]
            try:
                day_offset = 0 if "1" in str(day) else 1
            except Exception:
                day_offset = 0
            # attempt parse the time only
            dt_time = try_parse_datetime(base_date.strftime("%Y-%m-%d"), t)
            if pd.isna(dt_time):
                # fallback: simple parse HH:MM
                parts = str(t).strip().split(":")
                h = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
                m = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                return base_date + timedelta(days=day_offset, hours=h, minutes=m)
            else:
                # set date to base_date + offset
                return datetime.combine((base_date + timedelta(days=day_offset)).date(), dt_time.time())

        edited_df["DateTime"] = edited_df.apply(day_time_to_dt, axis=1)
        edited_df["Temperature"] = pd.to_numeric(edited_df["Temperature"], errors="coerce")
        edited_df = edited_df.dropna(subset=["Temperature"])
        df = edited_df.rename(columns={"Date": "Date", "Time": "Time", "Temperature": "Temperature"})[["Date", "Time", "Temperature", "DateTime"]].copy()
        df = df.sort_values("DateTime").reset_index(drop=True)

# -----------------------
# Proceed if dataframe not empty
# -----------------------
if not df.empty and "DateTime" in df.columns and not df["DateTime"].isna().all():
    # determine the 08:00 previous-day -> 08:00 last-day window based on latest datetime in data
    last_date = df["DateTime"].dt.date.max()
    end_time = datetime.combine(last_date, datetime.min.time()) + timedelta(hours=8)
    start_time = end_time - timedelta(hours=24)

    df_24h = df[(df["DateTime"] >= start_time) & (df["DateTime"] <= end_time)].copy()
    df_24h = df_24h.reset_index(drop=True)

    if df_24h.empty:
        st.warning("No data available in the last 24 hours (08:00 → 08:00).")
    else:
        # feature calculations (kept internal)
        df_24h["Hours"] = (df_24h["DateTime"] - df_24h["DateTime"].min()).dt.total_seconds() / 3600
        max_bt = df_24h["Temperature"].max()
        min_bt = df_24h["Temperature"].min()
        mean_bt = df_24h["Temperature"].mean()
        std_bt = df_24h["Temperature"].std()
        X = df_24h["Hours"].values.reshape(-1, 1)
        y = df_24h["Temperature"].values
        try:
            model_lr = LinearRegression().fit(X, y)
            slope = model_lr.coef_[0]
        except Exception:
            slope = 0.0
        last_time = df_24h["Hours"].max()
        last_8h = df_24h[df_24h["Hours"] >= last_time - 8]
        max_last8 = last_8h["Temperature"].max() if not last_8h.empty else np.nan
        range_bt = max_bt - min_bt
        diff_last8_allmax = (max_last8 - max_bt) if pd.notna(max_last8) else np.nan
        features = [max_bt, min_bt, mean_bt, std_bt, slope, range_bt, max_last8, diff_last8_allmax]

        # prediction (attempt)
        try:
            scaler = joblib.load("scaler.pkl")
            svm_model = joblib.load("svm_model.pkl")
            features_scaled = scaler.transform(np.array(features).reshape(1, -1))
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

        # -----------------------
        # Data Preview (Date, Time, Temperature)
        # Date: "Nov 12, 2025"
        # Time: "08:00 AM"
        # -----------------------
        st.write("### 🧾 Data Preview (Last 24h)")
        df_preview = df_24h.copy()
        df_preview["Date"] = df_preview["DateTime"].dt.strftime("%b %d, %Y")   # e.g., Nov 12, 2025
        df_preview["Time"] = df_preview["DateTime"].dt.strftime("%I:%M %p")    # e.g., 08:00 AM
        st.dataframe(df_preview[["Date", "Time", "Temperature"]], use_container_width=True)

        # -----------------------
        # Temperature Trend Plot
        # -----------------------
        st.subheader("📉 Temperature Trend (Last 24h)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_24h["DateTime"], df_24h["Temperature"], marker="o", linewidth=1.5, label="Temperature")
        ax.axhline(y=38, color="darkred", linestyle="--", linewidth=2, label="Fever Threshold (38°C)")
        ax.set_ylim(34.5, 43)
        ax.set_xlabel("Time")
        ax.set_ylabel("Temperature (°C)")
        # x-axis formatting: show month-day and time in friendly format
        formatter = mdates.DateFormatter("%b %d %I:%M %p")  # e.g., Nov 12 08:00 AM
        ax.xaxis.set_major_formatter(formatter)
        # auto-locate ticks
        locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
        ax.xaxis.set_major_locator(locator)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
else:
    st.info("⬆️ Please upload a CSV file or fill in temperatures manually to begin analysis.")














