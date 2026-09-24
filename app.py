import streamlit as st
import pandas as pd
import numpy as np
import joblib

saved = joblib.load("model.pkl")
model = saved["model"]
scaler = saved["scaler"]
threshold = saved["threshold"]

st.set_page_config(page_title="Fraud Detection",layout="centered")
st.title("🔍 Fraud Detection System")

st.header("Input Feature :-")

st.sidebar.subheader("Precision / Recall at different thresholds")
table = pd.read_csv("threshold_table.csv")
st.sidebar.dataframe(table, hide_index=True)
st.sidebar.line_chart(table.set_index("Threshold"))

V14 = st.slider("V14",min_value=-10.0,max_value=10.0,value=0.0)
V10 = st.slider("V10",min_value=-10.0,max_value=10.0,value=0.0)
V12 = st.slider("V12",min_value=-10.0,max_value=10.0,value=0.0)
V17 = st.slider("V17",min_value=-10.0,max_value=10.0,value=0.0)
V4 = st.slider("V4",min_value=-10.0,max_value=10.0,value=0.0)
V11 = st.slider("V11",min_value=-10.0,max_value=10.0,value=0.0)

if st.button("Predict"):

    input_data = pd.DataFrame(np.zeros((1,30)), columns=list(scaler.feature_names_in_))

    input_data["V14"] = V14
    input_data["V10"] = V10
    input_data["V12"] = V12
    input_data["V17"] = V17
    input_data["V4"]  = V4
    input_data["V11"] = V11

    input_scaled = scaler.transform(input_data)

    prob = model.predict_proba(input_scaled)[0][1]
    prediction = int(prob >= threshold)

    st.subheader("Result")
    st.progress(int(prob *100))

    if prediction == 1 :
        st.error(f"🚨 High Risk Transaction ({prob:.2f})")
    else:
        st.success(f"✅ Safe Transaction ({prob:.2f})")