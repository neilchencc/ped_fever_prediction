import streamlit as st
import pandas as pd

st.title("📂 CSV 檔案上傳與顯示")

# 上傳檔案元件
uploaded_file = st.file_uploader("請選擇一個 CSV 檔案", type=["csv"])

if uploaded_file is not None:
    # 讀取 CSV 檔案
    df = pd.read_csv(uploaded_file)
    
    st.success("✅ 檔案上傳成功！")
    st.write("### 檔案內容預覽：")
    
    # 顯示前幾筆資料
    st.dataframe(df.head())

    # 顯示檔案資訊
    st.write("資料筆數：", df.shape[0])
    st.write("欄位數量：", df.shape[1])
else:
    st.info("請上傳一個 CSV 檔案以開始。")
