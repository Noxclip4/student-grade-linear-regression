import streamlit as st
import pandas as pd
import joblib

st.title("Prediksi Nilai Akhir Siswa (G3)")
st.write("Model Linear Regression")

model = joblib.load("student_grade_lr.joblib")

G1 = st.slider("Nilai G1", 0, 20, 10)
G2 = st.slider("Nilai G2", 0, 20, 10)
studytime = st.selectbox("Waktu Belajar (1–4)", [1,2,3,4])
failures = st.number_input("Jumlah Gagal", 0, 5, 0)
absences = st.number_input("Jumlah Absen", 0, 100, 0)
internet = st.selectbox("Akses Internet", ["yes", "no"])
higher = st.selectbox("Ingin Kuliah?", ["yes", "no"])
schoolsup = st.selectbox("Dukungan Sekolah", ["yes", "no"])

input_df = pd.DataFrame([{
    "G1": G1,
    "G2": G2,
    "studytime": studytime,
    "failures": failures,
    "absences": absences,
    "internet": internet,
    "higher": higher,
    "schoolsup": schoolsup
}])

if st.button("Prediksi"):
    pred = model.predict(input_df)[0]
    st.success(f"Prediksi Nilai Akhir (G3): {pred:.2f}")
