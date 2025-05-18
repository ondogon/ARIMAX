# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
import joblib

# 모델과 스케일러 로딩
model = load_model('arimax_lstm_model.keras')
scaler = joblib.load('scaler1.pkl')

st.title("🌱 ARIMAX LSTM 기반 SOD / CAT 활성도 예측")

# 사용자 입력
uv_type = st.selectbox("자외선 타입 선택", ["UVA", "UVB"])
uv_type_value = 0 if uv_type == "UVA" else 1

uv_time = st.slider("자외선 조사 시간 (분)", 0, 40, step=5)

# 초기 시계열 입력 (정규화)
init_sod = scaler.transform(pd.DataFrame([[0, 0, 0.8]], columns=["UV_time", "Pred_SOD", "Pred_CAT"]))[0][2]
init_cat = scaler.transform(pd.DataFrame([[0, 0, 0.3]], columns=["UV_time", "Pred_SOD", "Pred_CAT"]))[0][2]

uv_time_scaled = scaler.transform(
    pd.DataFrame([[uv_time, 0, 0]], columns=["UV_time", "Pred_SOD", "Pred_CAT"])
)[0][0]

# 입력 시퀀스 생성 및 예측
sequence = [[uv_type_value, uv_time_scaled, init_sod, init_cat]] * 5
sod_preds, cat_preds = [], []

for _ in range(25):  # 총 50분 예측
    x_input = np.array(sequence[-5:]).reshape(1, 5, 4)
    pred = model.predict(x_input, verbose=0)[0]
    sod_preds.append(pred[0])
    cat_preds.append(pred[1])
    sequence.append([uv_type_value, uv_time_scaled, pred[0], pred[1]])

# 역정규화
sod_real = [scaler.inverse_transform([[0, 0, s]])[0][2] for s in sod_preds]
cat_real = [scaler.inverse_transform([[0, 0, c]])[0][2] for c in cat_preds]
times = list(range(0, 50, 2))

# Plotly 그래프
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=times, y=sod_real, mode='lines+markers', name='SOD 활성도',
    line=dict(color='blue'), hovertemplate='SOD: %{y:.3f}<br>시간: %{x}분'
))

fig.add_trace(go.Scatter(
    x=times, y=cat_real, mode='lines+markers', name='CAT 활성도',
    line=dict(color='red'), hovertemplate='CAT: %{y:.3f}<br>시간: %{x}분'
))

fig.update_layout(
    title="자외선 제거 후 경과 시간에 따른 SOD / CAT 활성도 예측",
    xaxis_title="경과 시간 (분)",
    yaxis_title="효소 활성도",
    hovermode="closest"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("💡 마우스를 그래프의 점 위에 올리면 해당 시점의 활성도 수치를 확인할 수 있어요.")
