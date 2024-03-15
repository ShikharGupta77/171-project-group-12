import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import joblib

def process_input(age, sleep_duration, quality_of_sleep, physical_activity_level, stress_level, heart_rate, daily_steps):
    data = pd.DataFrame({
        'Age': [age],
        'Sleep Duration': [sleep_duration],
        'Quality of Sleep': [quality_of_sleep],
        'Physical Activity Level': [physical_activity_level],
        'Stress Level': [stress_level],
        'Heart Rate': [heart_rate],
        'Daily Steps': [daily_steps],
    })
    return data

def predict_sleep_disorder(age, sleep_duration, quality_of_sleep, physical_activity_level, stress_level, heart_rate, daily_steps):
    data = np.array([[age, sleep_duration, quality_of_sleep, physical_activity_level, stress_level, heart_rate, daily_steps]])
    combined_data = np.vstack((X_test, data))
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(combined_data)
    scaled_test_value = combined_data_scaled[-1, :]  
    scaled_test_value_reshaped = scaled_test_value.reshape(1, -1)
    predicted_sleep_disorder = svm_model.predict(scaled_test_value_reshaped)
    return predicted_sleep_disorder

def main():
    st.title('Sleep Disorder Prediction')
    st.write('Enter the following information to predict sleep disorder:')

    age = st.slider('Age', min_value=1, max_value=100, value=44)
    sleep_duration = st.slider('Sleep Duration (hours)', min_value=1.0, max_value=24.0, value=6.4)
    quality_of_sleep = st.slider('Quality of Sleep (1-10)', min_value=1, max_value=10, value=6)
    physical_activity_level = st.slider('Physical Activity Level (How many minutes you were active)', min_value=1, max_value=240, value=45)
    stress_level = st.slider('Stress Level (1-10)', min_value=1, max_value=10, value=7)
    heart_rate = st.slider('Heart Rate', min_value=20, max_value=200, value=72)
    daily_steps = st.slider('Daily Steps', min_value=0, max_value=20000, value=6000)

    if st.button('Predict'):
        prediction = predict_sleep_disorder(age, sleep_duration, quality_of_sleep, physical_activity_level, stress_level, heart_rate, daily_steps)
        st.write('Predicted Sleep Disorder:', prediction)

if __name__ == '__main__':
    svm_model = joblib.load('svm_linear_model.pkl')  
    X_test = joblib.load('x_test.pkl')
    main()
