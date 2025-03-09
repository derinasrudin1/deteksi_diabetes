import pickle
import numpy as np
import streamlit as st

# Load model dan scaler
with open("diabetes_model_1.sav", "rb") as file:
    diabetes_model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Create title

st.title("Diabetes Prediction Web App with Support Vector Machine (SVM) Algorithm")
st.write("")
st.write("")

col1, col2 = st.columns(2)

# Get user input
with col1:
    Pregnancies = st.text_input("Pregnancies")
    Glucose = st.text_input("Glucose")
    BloodPressure = st.text_input("BloodPressure")
    SkinThickness = st.text_input("SkinThickness")
with col2:
    Insulin = st.text_input("Insulin")
    BMI = st.text_input("BMI")
    DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction")
    Age = st.text_input("Age")

st.write("")
st.write("")

# Create a button for Prediction
if st.button("Diabetes Test Result"):
    try:
        # Konversi input ke float dan ubah ke numpy array
        user_input = np.array(
            [
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age),
            ]
        ).reshape(1, -1)

        # Standarisasi input sebelum prediksi
        user_input = scaler.transform(user_input)

        # Prediksi dengan model
        diabetes_prediction = diabetes_model.predict(user_input)

        # Output hasil prediksi
        if diabetes_prediction[0] == 1:
            st.success("Pasien terkena diabetes")
        else:
            st.warning("Pasien tidak terkena diabetes")
    except ValueError:
        st.error("Harap masukkan angka yang valid!")

st.write("")
st.write("")

with st.chat_message("user"):
    st.write("Hello 👋")
    st.write(
        "My name is Deri Nasrudin, I am a student at Universitas Dian Nusantara majoring in Computer Science."
    )
    st.markdown(
        """
    <style>
    .footer {
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: grey;
        
    }
    </style>
    <div class="footer">
        © 2025 Aplikasi Prediksi Diabetes | <a href="https://github.com/derinasrudin1" target="_blank">GitHub</a> | 
        <a href="https://linkedin.com/in/username" target="_blank">LinkedIn</a>
    </div>
    """,
        unsafe_allow_html=True,
    )
