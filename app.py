import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="GDOES择优溅射深度仿真", layout="wide")
st.title("GDOES深度剖析仿真｜择优溅射效应")
st.markdown("MRI-CRAS简易模拟，可导出CSV深度数据用于Origin绘图")

# 侧边栏参数
with st.sidebar:
    st.header("仿真参数设置")
    total_depth = st.number_input("总溅射深度 (nm)", min_value=10.0, max_value=500.0, value=100.0, step=1.0)
    step_depth = st.number_input("深度步长 (nm)", min_value=0.05, max_value=5.0, value=0.2, step=0.05)
    mix_length = st.number_input("离子混合长度 (nm)", min_value=0.1, max_value=20.0, value=3.0, step=0.1)
    prefer_factor = st.slider("择优溅射因子", min_value=0.5, max_value=1.5, value=1.0, step=0.05,
        help="<1：A更容易被溅射；>1：B更容易被溅射；=1无择优溅射")
    interface_pos = st.number_input("界面位置 (nm)", min_value=5.0, max_value=495.0, value=50.0, step=1.0)

run_sim = st.button("▶ 开始仿真绘图")

# 仿真主函数
def gdoes_sim(depth_array, interface, Lmix, k_sput):
    ideal_A = np.where(depth_array < interface, 1.0, 0.0)
    sigma = Lmix / 2.355
    # 高斯卷积模拟离子混合
    def gauss(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    mixed_A = np.zeros_like(depth_array)
    for i, x in enumerate(depth_array):
        weight = gauss(depth_array, x, sigma)
        weight = weight / np.sum(weight)
        mixed_A[i] = np.sum(ideal_A * weight)
    # 择优溅射修正
    signal_A = mixed_A ** k_sput
    signal_B = 1.0 - signal_A
    return signal_A, signal_B, ideal_A

if run_sim:
    with st.spinner("仿真计算中..."):
        depth_arr = np.arange(0, total_depth, step_depth)
        sigA, sigB, idealA = gdoes_sim(depth_arr, interface_pos, mix_length, prefer_factor)
        df_out = pd.DataFrame({
            "Depth_nm": depth_arr,
            "Ideal_A": idealA,
            "Signal_A": sigA,
            "Signal_B": sigB
        })
        # Plotly绘图
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_out["Depth_nm"], y=df_out["Ideal_A"], name="理想A组分", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=df_out["Depth_nm"], y=df_out["Signal_A"], name="仿真A信号"))
        fig.add_trace(go.Scatter(x=df_out["Depth_nm"], y=df_out["Signal_B"], name="仿真B信号"))
        fig.update_layout(xaxis_title="溅射深度 nm", yaxis_title="归一化组分",
            height=600, legend_title="组分")
        st.plotly_chart(fig, use_container_width=True)
        # CSV下载按钮
        csv = df_out.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("📥 下载仿真CSV数据(可导入Origin)", csv, file_name="gdoes_sim.csv", mime="text/csv")
        st.caption("简易MRI模型，仅用于本科课题趋势仿真，定量还需要继续完善模型")