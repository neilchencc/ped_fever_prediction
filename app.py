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
st.title("📈 Body Temperature Analysis Tool (CSV Upload / Manual Entry + Normalization + Prediction)")

st.markdown("""
**App Description:**  
This app uses historical body temperature records from **08:00 to 08:00 the following day** 
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
        df = pd.read_csv(uploaded_file)
        df.columns = ["Date", "Time", "Temperature"][:len(df.columns)]
        df.columns = [c.strip() for c in df.columns]
        df["DateTime"] = df.apply(
            lambda row: datetime.strptime(str(int(row["Date"])) + f"{int(row['Time']):04d}", "%Y%m%d%H%M"),
            axis=1
        )
        df = df.sort_values("DateTime").reset_index(drop=True)

# ----------------------
# Manual Entry as DataFrame
# ----------------------
elif input_method == "Manual Entry":
    st.subheader("Manual Data Entry (editable table)")

    # Create all time points from 08:00 Day1 → 07:00 Day2
    day1_times = [f"{h:02d}:00" for h in range(8,24)]
    day2_times = [f"{h:02d}:00" for h in range(0,8)]
    all_times = [("Day1", t) for t in day1_times] + [("Day2", t) for t in day2_times]

    manual_df = pd.DataFrame(all_times, columns=["Date", "Time"])
    manual_df["Temperature"] = np.nan  # empty temperature column

    # Editable table
    edited_df = st.data_editor(manual_df, num_rows="dynamic", use_container_width=True)

    # Remove empty temperature rows
    edited_df = edited_df.dropna(subset=["Temperature"])
    if not edited_df.empty:
        df = edited_df.copy()
        # Map Day1/Day2 to datetime
        base_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        df["DateTime"] = df.apply(lambda row: base_date + timedelta(days=int(row["Date"][-1])-1,
                                                                hours=int(row["Time"][:2]),
                                                                minutes=int(row["Time"][3:])), axis=1)
        df = df.sort_values("DateTime").reset_index(drop=True)

# ----------------------
# Proceed if dataframe not empty
# ----------------------
if not df.empty:
    # ----------------------
    # Prediction
    # ----------------------
    st.subheader("🤖 Prediction Result")
    try:
        df["Hours"] = (df["DateTime"] - df["DateTime"].min()).dt.total_seconds()/3600

        max_bt = df["Temperature"].max()
        min_bt = df["Temperature"].min()
        mean_bt = df["Temperature"].mean()
        std_bt = df["Temperature"].std()
        X = df["Hours"].values.reshape(-1,1)
        y = df["Temperature"].values
        model_lr = LinearRegression().fit(X,y)
        slope = model_lr.coef_[0]
        last_time = df["Hours"].max()
        last_8h = df[df["Hours"] >= last_time-8]
        max_last8 = last_8h["Temperature"].max()
        range_bt = max_bt - min_bt
        diff_last8_allmax = max_last8 - max_bt

        features = [max_bt, min_bt, mean_bt, std_bt, slope, range_bt, max_last8, diff_last8_allmax]

        # Load models
        scaler = joblib.load("scaler.pkl")
        svm_model = joblib.load("svm_model.pkl")
        features_scaled = scaler.transform(np.array(features).reshape(1,-1))

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
        st.error(f"Error during prediction: {e}")

    # ----------------------
    # Data Preview
    # ----------------------
    st.write("### 🧾 Data Preview")
    st.dataframe(df)

    # ----------------------
    # Statistical Summary
    # ----------------------
    st.subheader("📊 Statistical Summary")
    feature_names = [
        "Maximum (max)", "Minimum (min)", "Average (mean)", "Standard Deviation (std)",
        "Slope", "Max - Min", "Max of Last 8 Hours", "Last 8h Max - Overall Max"
    ]
    features_values = [max_bt, min_bt, mean_bt, std_bt, slope, range_bt, max_last8, diff_last8_allmax]
    result_table = pd.DataFrame({"Feature": feature_names, "Value":[f"{v:.4f}" for v in features_values]})
    st.table(result_table)

    # ----------------------
    # Temperature Trend Plot
    # ----------------------
    st.subheader("📉 Temperature Trend")
    fig, ax = plt.subplots()
    ax.plot(df["DateTime"], df["Temperature"], marker='o', label="Temperature")
    ax.axhline(y=38, color='darkred', linestyle='--', linewidth=2, label="Fever Threshold (38°C)")
    ax.set_ylim(35, 43)
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

else:
    st.info("⬆️ Please upload a CSV file or fill in temperatures manually to begin analysis.")








