import streamlit as st
import pandas as pd
import pickle

with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

df = pd.read_csv("dataset_processed.csv")

# Input fields
make_model_map = df.groupby("make")["model"].unique().apply(list).to_dict()
cylinders_options = sorted(df['cylinders'].dropna().unique())
fuel_options = df['fuel'].unique()
trim_options = df['trim'].unique()
body_options = df['body'].unique()
doors_options = sorted(df['doors'].dropna().unique())
drivetrain_options = df['drivetrain'].unique()
valvecount_options = sorted(df['valve_count'].dropna().unique())
fuelsystem_options = df['fuel_system'].unique()
valvetrain_options = df['valve_train'].unique()
aspiration_options = df['aspiration'].unique()
transmission_options = df['transmission_type'].unique()
gears_options = sorted(df['gears'].dropna().unique())
extcolor_options = df['ext_color'].unique()
intcolor_options = df['int_color'].unique()

st.set_page_config(layout="wide")
st.title("Vehicle Price Predictor")


col1, col2, col3 = st.columns(3)

with col1:
    make_options = list(make_model_map.keys())
    make = st.selectbox("Make", make_options, key="make_select")
    model_options = make_model_map.get(make, [])
    model = st.selectbox("Model", model_options, key="model_select")
    year = st.slider("Year", min_value=2021, max_value=2025, value=2024, key="year_slider")
    cylinders = st.selectbox("Cylinders", cylinders_options, key="cylinders_select")
    fuel = st.selectbox("Fuel Type", fuel_options, key="fuel_select")
    mileage = st.number_input("Mileage (miles/l)", min_value=0.0, max_value=30.0, value=10.0, step=0.1, format="%.1f", key="mileage_input")
    
with col2:
    trim = st.selectbox("Trim", trim_options, key="trim_select")
    body = st.selectbox("Body Style", body_options, key="body_select")
    doors = st.selectbox("Doors", doors_options, key="doors_select")
    drivetrain = st.selectbox("Drivetrain", drivetrain_options, key="drivetrain_select")
    valve_count = st.selectbox("Valve Count", valvecount_options, key="valve_count_select")
    fuel_system = st.selectbox("Fuel System", fuelsystem_options, key="fuel_system_select")

with col3:
    valve_train = st.selectbox("Valve Train", valvetrain_options, key="valve_train_select")
    aspiration = st.selectbox("Aspiration", aspiration_options, key="aspiration_select")
    transmission_type = st.selectbox("Transmission Type", transmission_options, key="transmission_select")
    gears = st.selectbox("Number of Gears", gears_options, key="gears_select")
    ext_color = st.selectbox("Exterior Color", extcolor_options, key="ext_color_select")
    int_color = st.selectbox("Interior Color", intcolor_options, key="int_color_select")


if st.button("Predict Price", key="predict_button"):
    new_vehicle = pd.DataFrame([{
        'make': make,
        'model': model,
        'year': year,
        'cylinders': cylinders,
        'fuel': fuel,
        'mileage': mileage,
        'trim': trim,
        'body': body,
        'doors': doors,
        'drivetrain': drivetrain,
        'valve_count': valve_count,
        'fuel_system': fuel_system,
        'valve_train': valve_train,
        'aspiration': aspiration,
        'transmission_type': transmission_type,
        'gears': gears,
        'ext_color': ext_color,
        'int_color': int_color
    }])

    try:
        predicted_price = loaded_model.predict(new_vehicle)
        st.success(f"Predicted Price: ${predicted_price[0]:,.2f}")
        
        st.subheader("Input Specifications")
        st.dataframe(new_vehicle)
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")