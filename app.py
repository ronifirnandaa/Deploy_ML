import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Prediksi Diabetes",
    page_icon="🩺",
    layout="centered"
)

# ---------------------------
# Load Model & Scaler
# ---------------------------
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------------------
# Load Dataset Referensi
# ---------------------------
@st.cache_data
def load_reference_data():
    df = pd.read_csv("diabetes.csv")
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    return X, y

X_ref, y_ref = load_reference_data()

# ---------------------------
# Main Title
# ---------------------------
st.title("Prediksi Diabetes Menggunakan SVM")
st.write(
    "Aplikasi ini digunakan untuk memprediksi kemungkinan diabetes "
    "berdasarkan data medis pasien."
)

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("Informasi Model")

# === AKURASI MODEL ===
MODEL_ACCURACY = 0.7338

st.sidebar.metric(
    label="Akurasi Model",
    value=f"{MODEL_ACCURACY*100:.2f}%"
)

st.sidebar.markdown("""
**Algoritma** : Support Vector Machine (SVM)  
**Dataset** : Pima Indians Diabetes  

**Kelas Output**:
- 0 → Tidak Diabetes  
- 1 → Diabetes
""")

st.sidebar.markdown("---")

input_type = st.sidebar.radio(
    "Metode Input Data",
    ["Input Manual", "Upload CSV"]
)

# ===========================
# SESSION STATE
# ===========================
if "input_data" not in st.session_state:
    st.session_state.input_data = X_ref.iloc[0].values

# ===========================
# INPUT MANUAL
# ===========================
if input_type == "Input Manual":

    st.subheader("Input Data Pasien")

    col_input, col_gen = st.columns([3, 1])

    # ===== GENERATOR =====
    with col_gen:
        st.markdown("### Generate Data")
        gen_label = st.selectbox(
            "Label Data",
            [0, 1],
            format_func=lambda x: "Tidak Diabetes (0)" if x == 0 else "Diabetes (1)"
        )

        if st.button("Generate Acak"):
            idx = np.random.choice(y_ref[y_ref == gen_label].index)
            st.session_state.input_data = X_ref.loc[idx].values
            st.success("Data berhasil digenerate")

    # ===== INPUT FORM =====
    with col_input:
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input(
                "Jumlah Kehamilan", 0, 20,
                int(st.session_state.input_data[0])
            )
            glucose = st.number_input(
                "Kadar Glukosa", 0, 200,
                int(st.session_state.input_data[1])
            )
            blood_pressure = st.number_input(
                "Tekanan Darah (mm Hg)", 0, 150,
                int(st.session_state.input_data[2])
            )
            skin_thickness = st.number_input(
                "Ketebalan Lipatan Kulit (mm)", 0, 100,
                int(st.session_state.input_data[3])
            )

        with col2:
            insulin = st.number_input(
                "Kadar Insulin (mu U/ml)", 0, 900,
                int(st.session_state.input_data[4])
            )
            bmi = st.number_input(
                "BMI", 0.0, 70.0,
                float(st.session_state.input_data[5])
            )
            dpf = st.number_input(
                "Diabetes Pedigree Function", 0.0, 3.0,
                float(st.session_state.input_data[6])
            )
            age = st.number_input(
                "Usia", 0, 120,
                int(st.session_state.input_data[7])
            )

    data = np.array([[pregnancies, glucose, blood_pressure,
                      skin_thickness, insulin, bmi, dpf, age]])

# ===========================
# UPLOAD CSV
# ===========================
elif input_type == "Upload CSV":

    st.subheader("Upload Data Pasien (CSV)")

    uploaded_file = st.file_uploader(
        "Upload file CSV (1 baris, 8 fitur)",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        data = df.values
    else:
        data = None

# ---------------------------
# PREDIKSI
# ---------------------------
st.markdown("---")

if st.button("Prediksi") and data is not None:
    data_scaled = scaler.transform(data)
    prediction = svm_model.predict(data_scaled)[0]

    st.subheader("Hasil Prediksi")

    if prediction == 1:
        st.error("⚠️ Pasien diprediksi **MENDERITA DIABETES**")
    else:
        st.success("✅ Pasien diprediksi **TIDAK MENDERITA DIABETES**")
