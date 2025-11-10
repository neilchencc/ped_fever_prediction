import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import joblib  # 載入 pkl 模型

st.title("📈 體溫紀錄分析工具（CSV 上傳 + 預測）")

uploaded_file = st.file_uploader("請上傳包含 Date, Time, BT 欄位的 CSV 檔案", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]

    df["DateTime"] = df.apply(
        lambda row: datetime.strptime(str(int(row["Date"])) + f"{int(row['Time']):04d}", "%Y%m%d%H%M"),
        axis=1
    )
    df = df.sort_values("DateTime").reset_index(drop=True)

    st.write("### 🧾 原始資料預覽：")
    st.dataframe(df)

    unique_dates = sorted(df["Date"].unique())
    if len(unique_dates) < 2:
        st.error("⚠️ 資料不足，請至少包含兩個不同日期。")
    else:
        second_last_date = unique_dates[-2]
        last_date = unique_dates[-1]

        start_time = datetime.strptime(str(second_last_date) + "0800", "%Y%m%d%H%M")
        end_time = datetime.strptime(str(last_date) + "2359", "%Y%m%d%H%M")

        df_range = df[(df["DateTime"] >= start_time) & (df["DateTime"] <= end_time)]

        if df_range.empty:
            st.warning("⚠️ 此時間區間內沒有資料。")
        else:
            st.write(f"### ⏱ 分析範圍：{start_time} ～ {end_time}")
            st.dataframe(df_range)

            t0 = df_range["DateTime"].min()
            df_range["Hours"] = (df_range["DateTime"] - t0).dt.total_seconds() / 3600

            max_bt = df_range["BT"].max()
            min_bt = df_range["BT"].min()
            mean_bt = df_range["BT"].mean()
            std_bt = df_range["BT"].std()

            X = df_range["Hours"].values.reshape(-1, 1)
            y = df_range["BT"].values
            model_lr = LinearRegression().fit(X, y)
            slope = model_lr.coef_[0]

            last_time = df_range["Hours"].max()
            last_8h = df_range[df_range["Hours"] >= last_time - 8]
            max_last8 = last_8h["BT"].max()

            range_bt = max_bt - min_bt
            diff_last8_allmax = max_last8 - max_bt

            # 顯示統計結果表
            features = [max_bt, min_bt, mean_bt, std_bt, slope, range_bt, max_last8, diff_last8_allmax]
            feature_names = [
                "最大值 (max)", "最小值 (min)", "平均值 (mean)", "標準差 (std)",
                "斜率 (slope)", "max - min", "最後8小時的 max", "最後8小時 max - 全部 max"
            ]

            result_table = pd.DataFrame({
                "指標": feature_names,
                "數值": [f"{v:.4f}" for v in features]
            })
            st.subheader("📊 統計結果")
            st.table(result_table)

            # 預測部分
            st.subheader("🤖 預測結果")

            try:
                svm_model = joblib.load("svm_model.pkl")  # 請確定此檔案跟app.py同目錄

                # SVM 預測，注意輸入需是2D array
                pred_prob = svm_model.predict_proba([features])[0][1] if hasattr(svm_model, "predict_proba") else svm_model.decision_function([features])[0]
                # 這裡我們用 decision_function（無機率的話）或 predict_proba 第二欄機率

                threshold = 0.5
                if pred_prob >= threshold:
                    st.success(f"預測結果：未來會發燒 (機率/分數={pred_prob:.3f} ≥ {threshold})")
                else:
                    st.info(f"預測結果：未來不會發燒 (機率/分數={pred_prob:.3f} < {threshold})")

            except FileNotFoundError:
                st.error("找不到 svm_model.pkl 模型檔，請確認檔案存在於同目錄下。")
            except Exception as e:
                st.error(f"載入或預測時發生錯誤：{e}")

            st.subheader("📉 體溫變化圖")
            st.line_chart(df_range.set_index("DateTime")["BT"])

else:
    st.info("⬆️ 請上傳一個 CSV 檔以開始分析。")

