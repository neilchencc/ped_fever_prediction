import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
import joblib  # for loading pkl models

# ---------------------------------------------------
# 📘 Streamlit Title
# ---------------------------------------------------
st.title("📈 Body Temperature Analysis Tool (CSV Upload + Normalization + Prediction)")

# ---------------------------------------------------
# 📂 Upload CSV File
# ---------------------------------------------------
uploaded_file = st.file_uploader("Please upload a CSV file with three columns: Date, Time, Temperature", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Rename columns to standard names regardless of input
    expected_cols = ["Date", "Time", "Temperature"]
    df.columns = expected_cols[:len(df.columns)]
    df.columns = [c.strip() for c in df.columns]

    # Create DateTime column
    df["DateTime"] = df.apply(
        lambda row: datetime.strptime(str(int(row["Date"])) + f"{int(row['Time']):04d}", "%Y%m%d%H%M"),
        axis=1
    )

    df = df.sort_values("DateTime").reset_index(drop=True)

    st.write("### 🧾 Data Preview:")
    st.dataframe(df)

    # ---------------------------------------------------
    # 🧮 Data Range Selection
    # ---------------------------------------------------
    unique_dates = sorted(df["Date"].unique())
    if len(unique_dates) < 2:
        st.error("⚠️ Not enough data. Please include at least two different dates.")
    else:
        second_last_date = unique_dates[-2]
        last_date = unique_dates[-1]

        start_time = datetime.strptime(str(second_last_date) + "0800", "%Y%m%d%H%M")
        end_time = datetime.strptime(str(last_date) + "2359", "%Y%m%d%H%M")

        df_range = df[(df["DateTime"] >= start_time) & (df["DateTime"] <= end_time)]

        if df_range.empty:
            st.warning("⚠️ No data found in the specified time range.")
        else:
            st.write(f"### ⏱ Analysis Range: {start_time} – {end_time}")
            st.dataframe(df_range)

            # ---------------------------------------------------
            # 🧩 Feature Engineering
            # ---------------------------------------------------
            t0 = df_range["DateTime"].min()
            df_range["Hours"] = (df_range["DateTime"] - t0).dt.total_seconds() / 3600

            max_bt = df_range["Temperature"].max()
            min_bt = df_range["Temperature"].min()
            mean_bt = df_range["Temperature"].mean()
            std_bt = df_range["Temperature"].std()

            X = df_range["Hours"].values.reshape(-1, 1)
            y = df_range["Temperature"].values
            model_lr = LinearRegression().fit(X, y)
            slope = model_lr.coef_[0]

            last_time = df_range["Hours"].max()
            last_8h = df_range[df_range["Hours"] >= last_time - 8]
            max_last8 = last_8h["Temperature"].max()

            range_bt = max_bt - min_bt
            diff_last8_allmax = max_last8 - max_bt

            # Build feature list
            features = [max_bt, min_bt, mean_bt, std_bt, slope, range_bt, max_last8, diff_last8_allmax]
            feature_names = [
                "Maximum (max)", "Minimum (min)", "Average (mean)", "Standard Deviation (std)",
                "Slope", "Max - Min", "Max of Last 8 Hours", "Last 8h Max - Overall Max"
            ]

            result_table = pd.DataFrame({
                "Feature": feature_names,
                "Value": [f"{v:.4f}" for v in features]
            })
            st.subheader("📊 Statistical Summary")
            st.table(result_table)

            # ---------------------------------------------------
            # 🤖 Prediction
            # ---------------------------------------------------
            st.subheader("🤖 Prediction Result")

            try:
                # Load scaler and model
                scaler = joblib.load("scaler.pkl")
                svm_model = joblib.load("svm_model.pkl")

                # Normalize features
                features_array = np.array(features).reshape(1, -1)
                features_scaled = scaler.transform(features_array)

                # Model prediction
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
                st.error(f"Missing required model file: {e.filename}")
            except Exception as e:
                st.error(f"Error during model loading or prediction: {e}")

            # ---------------------------------------------------
            # 📉 Temperature Chart (Y-axis fixed between 35~43°C)
            # ---------------------------------------------------
            st.subheader("📉 Temperature Trend")

            fig, ax = plt.subplots()
            ax.plot(df_range["DateTime"], df_range["Temperature"], marker='o')
            ax.axhline(y=38, color='darkred', linestyle='--', linewidth=2, label="Fever Threshold (38°C)")  # 🔥 Dark red line
            ax.set_ylim(35, 43)
            ax.set_xlabel("Time")
            ax.set_ylabel("Temperature (°C)")
            ax.grid(True)
            st.pyplot(fig)

else:
    st.info("⬆️ Please upload a CSV file to begin analysis.")




