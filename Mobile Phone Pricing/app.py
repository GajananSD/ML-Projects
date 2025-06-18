import streamlit as st
import numpy as np
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Mobile Price Range Predictor")

# Input fields
battery_power = st.number_input("Battery Power (mAh)", min_value=501, max_value=1998, step=50)
blue = st.selectbox("Bluetooth", [0, 1])
clock_speed = st.number_input("Clock Speed (GHz)", min_value=0.5, max_value=3.0, step=0.1, format="%.1f")
dual_sim = st.selectbox("Dual SIM", [0, 1])
fc = st.slider("Front Camera (MP)", min_value=0, max_value=50)
four_g = st.selectbox("4G", [0, 1])
int_memory = st.slider("Internal Memory (GB)", min_value=2, max_value=128)
m_dep = st.number_input("Mobile Depth (cm)", min_value=0.1, max_value=2.0, step=0.1, format="%.1f")
mobile_wt = st.slider("Mobile Weight (grams)", min_value=80, max_value=300)
n_cores = st.slider("Number of Cores", min_value=1, max_value=16)
pc = st.slider("Primary Camera (MP)", min_value=0, max_value=50)
px_height = st.number_input("Pixel Height", min_value=0, max_value=2200)
px_width = st.number_input("Pixel Width", min_value=500, max_value=3200)
ram = st.number_input("RAM (MB)", min_value=256, max_value=16384)
sc_h = st.slider("Screen Height (cm)", min_value=5, max_value=19)
sc_w = st.slider("Screen Width (cm)", min_value=2, max_value=18)
talk_time = st.slider("Talk Time (hrs)", min_value=2, max_value=20)
three_g = st.selectbox("3G", [0, 1])
touch_screen = st.selectbox("Touch Screen", [0, 1])
wifi = st.selectbox("WiFi", [0, 1])

pixel_density = px_width * px_height
screen_area = sc_h * sc_w
core_efficiency = clock_speed * n_cores

if st.button("Predict Price Range"):
    input_data = np.array([[battery_power, blue, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt, pc, ram,
                            talk_time, three_g,	touch_screen, wifi,	pixel_density, screen_area, core_efficiency]])
    
    prediction = model.predict(input_data)

    label_map = {0: "Low cost (0)", 1: "Medium cost (1)", 2: "High cost (2)", 3: "Very High cost (3)"}
    st.success(f"Predicted Price Range: {label_map[prediction[0]]}")