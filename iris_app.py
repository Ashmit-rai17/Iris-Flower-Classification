import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('../models/iris_rf_model.joblib')

st.title('Iris Flower Species Predictor')
st.write('Enter the features of the iris flower to predict its species:')

# Input fields for features
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=0.2)

features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button('Predict'):
    prediction = model.predict(features)
    st.success(f'Predicted Species: {prediction[0]}') 