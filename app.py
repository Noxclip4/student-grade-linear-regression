import streamlit as st
import pandas as pd
import joblib

# ======================
# Page config
# ======================
st.set_page_config(
    page_title="Prediksi Nilai Akhir Siswa (G3)",
    layout="centered"
)

# ======================
# Header
# ======================
st.title("📘 Prediksi Nilai Akhir Siswa (G3)")
st.caption("Deployment Machine Learning — Linear Regression (Streamlit)")

st.markdown(
    """
Aplikasi ini memprediksi **nilai akhir siswa (G3)** menggunakan model **Linear Regression**.
Skala nilai pada dataset adalah **0–20**.

**Cara pakai:**
1) Masukkan nilai & faktor siswa  
2) Klik **Prediksi Nilai Akhir**  
3) Lihat hasil prediksi dan interpretasinya
"""
)

# ======================
# Load model
# ======================
@st.cache_resource
def load_model():
    return joblib.load("student_grade_lr.joblib")

try:
    model = load_model()
except Exception as e:
    st.error("Model tidak bisa dibaca. Pastikan file `student_grade_lr.joblib` ada di folder yang sama dengan `app.py`.")
    st.exception(e)
    st.stop()

# ======================
# Sidebar info (optional)
# ======================
with st.sidebar:
    st.subheader("ℹ️ Info Singkat")
    st.write(
        """
**Target:** G3 (nilai akhir)  
**Fitur input:** G1, G2, studytime, failures, absences, internet, higher, schoolsup  
**Catatan:** Input kategori memakai `yes/no` sesuai dataset.
"""
    )
    st.write("---")
    st.write("Tip: Coba ubah **failures** dan lihat efeknya pada prediksi.")

# ======================
# Input form
# ======================
st.subheader("📝 Input Data Siswa")

col1, col2 = st.columns(2)

with col1:
    G1 = st.slider(
        "G1 — Nilai periode 1",
        0, 20, 10,
        help="Nilai siswa pada periode/semester pertama (skala 0–20)."
    )
    studytime = st.selectbox(
        "Studytime — Waktu belajar (1–4)",
        [1, 2, 3, 4],
        help="Kategori waktu belajar mingguan. 1=sedikit, 4=banyak."
    )
    absences = st.number_input(
        "Absences — Jumlah absen",
        min_value=0, max_value=100, value=0,
        help="Jumlah ketidakhadiran siswa selama periode data."
    )
    internet = st.selectbox(
        "Internet — Akses internet di rumah",
        ["yes", "no"],
        help="Apakah siswa punya akses internet di rumah (yes/no)."
    )

with col2:
    G2 = st.slider(
        "G2 — Nilai periode 2",
        0, 20, 10,
        help="Nilai siswa pada periode/semester kedua (skala 0–20)."
    )
    failures = st.number_input(
        "Failures — Jumlah kegagalan akademik",
        min_value=0, max_value=5, value=0,
        help="Jumlah kegagalan akademik sebelumnya (mis. pernah tidak lulus / mengulang). Nilai tinggi biasanya menurunkan prediksi."
    )
    higher = st.selectbox(
        "Higher — Ingin melanjutkan kuliah?",
        ["yes", "no"],
        help="Apakah siswa berencana melanjutkan pendidikan ke jenjang lebih tinggi (yes/no)."
    )
    schoolsup = st.selectbox(
        "Schoolsup — Dukungan sekolah?",
        ["yes", "no"],
        help="Apakah siswa mendapat dukungan belajar tambahan dari sekolah (yes/no)."
    )

# ======================
# Domain sanity warnings
# ======================
warnings = []
if failures >= 3:
    warnings.append("Failures tinggi (≥3). Ini biasanya sangat menurunkan nilai akhir.")
if absences >= 30:
    warnings.append("Absences tinggi (≥30). Ketidakhadiran tinggi umumnya berdampak negatif pada nilai.")
if (G2 < G1) and (G1 >= 15):
    warnings.append("G2 lebih rendah dari G1. Bisa jadi performa menurun pada periode 2.")

if warnings:
    st.warning("⚠️ Peringatan input:\n- " + "\n- ".join(warnings))

# ======================
# Build input dataframe
# ======================
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

st.markdown("#### Preview input yang dikirim ke model")
st.dataframe(input_df, use_container_width=True)

# ======================
# Predict
# ======================
st.write("")

if st.button("🎯 Prediksi Nilai Akhir", use_container_width=True):
    try:
        pred = float(model.predict(input_df)[0])
    except Exception as e:
        st.error("Terjadi error saat melakukan prediksi. Pastikan kolom input sesuai saat training.")
        st.exception(e)
        st.stop()

    # Clip to valid grade range (0–20)
    pred_clipped = max(0.0, min(20.0, pred))

    st.success(f"📊 Prediksi Nilai Akhir (G3): **{pred_clipped:.2f}** (skala 0–20)")

    # Simple interpretation bucket
    if pred_clipped < 8:
        label = "🔴 Rendah"
        note = "Disarankan dukungan belajar tambahan (remedial, pendampingan, konseling belajar)."
    elif pred_clipped < 14:
        label = "🟡 Sedang"
        note = "Cukup baik, tetapi peningkatan bisa fokus pada konsistensi belajar dan mengurangi absen."
    else:
        label = "🟢 Tinggi"
        note = "Performa akademik baik. Jaga konsistensi dan kebiasaan belajar."

    st.info(f"**Kategori:** {label}\n\n**Catatan:** {note}")

    with st.expander("📌 Kenapa hasilnya bisa begitu? (penjelasan awam)"):
        st.markdown(
            f"""
Model **Linear Regression** membuat prediksi dari kombinasi beberapa faktor.

**Yang biasanya paling kuat:**
- **G1 & G2**: nilai sebelumnya biasanya sangat berhubungan dengan nilai akhir.
- **Failures**: semakin tinggi kegagalan akademik, umumnya prediksi nilai akhir turun.
- **Absences**: absen tinggi sering berkaitan dengan nilai lebih rendah.

Model tidak “mengerti” manusia seperti guru, tapi belajar pola dari data:
input yang mirip → output prediksi yang mirip.
"""
        )

# ======================
# Explain features & quick table
# ======================
with st.expander("📚 Penjelasan Fitur (apa arti tiap input?)"):
    st.markdown(
        """
- **G1**: nilai periode/semester 1 (0–20)  
- **G2**: nilai periode/semester 2 (0–20)  
- **G3**: nilai akhir (target yang diprediksi) (0–20)  
- **studytime**: kategori waktu belajar (1–4)  
- **failures**: jumlah kegagalan akademik sebelumnya (mis. pernah tidak lulus / mengulang)  
- **absences**: jumlah ketidakhadiran  
- **internet**: akses internet di rumah (yes/no)  
- **higher**: rencana melanjutkan kuliah (yes/no)  
- **schoolsup**: dukungan belajar tambahan dari sekolah (yes/no)

**Kenapa skala 0–20?**  
Karena dataset menggunakan sistem penilaian skala 0–20. Jika ingin “dibayangkan” ke 0–100, perkiraan kasar: `nilai_100 ≈ nilai_20 × 5`.
"""
    )

with st.expander("📊 Ringkasan Arah Pengaruh (versi mudah)"):
    st.table(pd.DataFrame({
        "Faktor": ["G1", "G2", "Studytime", "Failures", "Absences", "Internet", "Higher", "School support"],
        "Arah umum": ["Naik", "Naik (kuat)", "Naik (kecil)", "Turun (kuat)", "Turun", "Bervariasi", "Cenderung naik", "Bervariasi"],
        "Penjelasan singkat": [
            "Nilai awal biasanya berlanjut",
            "Nilai periode 2 sangat dekat dengan nilai akhir",
            "Lebih banyak belajar sering membantu",
            "Riwayat gagal biasanya menurunkan performa",
            "Sering absen membuat ketinggalan materi",
            "Bisa membantu belajar, bisa juga distraksi",
            "Motivasi lanjut studi sering terkait usaha belajar",
            "Kadang diberi ke siswa yang butuh bantuan"
        ]
    }))

st.write("---")
st.caption("Catatan: Arah pengaruh bisa berbeda tergantung pola data. Untuk versi lebih ilmiah, koefisien model dapat ditampilkan bila disimpan saat training.")
