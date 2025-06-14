
import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌸")

# Title
st.title("🌸 Iris Flower Species Predictor")
st.write("Enter the measurements below to predict the Iris species:")

# Load model
try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ model.pkl not found!")
    st.stop()

# Input form
with st.form("iris_form"):
    sepal_length = st.number_input("Sepal length (cm)", min_value=0.0, format="%.2f")
    sepal_width = st.number_input("Sepal width (cm)", min_value=0.0, format="%.2f")
    petal_length = st.number_input("Petal length (cm)", min_value=0.0, format="%.2f")
    petal_width = st.number_input("Petal width (cm)", min_value=0.0, format="%.2f")

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    species = ['Setosa', 'Versicolor', 'Virginica'][prediction]
    st.success(f"🌺 Predicted Iris species: **{species}**")
