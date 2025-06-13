
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns

st.set_page_config(
    page_title="Dashboard Prediksi Stroke",
    page_icon="🧠"
)

st.title("Stroke Risk Prediction App - Naive Bayes Classifier")
st.write("Model ini menggunakan pendekatan Naive Bayes untuk memprediksi apakah seseorang berisiko terkena stroke, Masukkan data pasien untuk memprediksi kemungkinan stroke.")
st.markdown("---")
st.subheader("Evaluasi Model")

model = joblib.load("naive_bayes_stroke_model.pkl")
features = joblib.load("model_features.pkl")
data = pd.read_csv("healthcare-dataset-stroke-data.csv")

data.drop(columns=["id"], inplace=True)
data.dropna(inplace=True)
data["gender"] = data["gender"].map({"Male": 0, "Female": 1, "Other": 2})
data["Residence_type"] = data["Residence_type"].map({"Urban": 1, "Rural": 0})
data = pd.get_dummies(data, drop_first=True)

for col in features:
    if col not in data.columns:
        data[col] = 0

X = data[features]
y = data["stroke"]
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
roc_auc = roc_auc_score(y, y_prob)

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Accuracy", f"{acc:.2f}")
with col2: st.metric("Precision", f"{prec:.2f}")
with col3: st.metric("Recall", f"{rec:.2f}")
with col4: st.metric("ROC AUC", f"{roc_auc:.2f}")

plot_option = st.selectbox("Pilih grafik untuk ditampilkan:", ["Pilih", "ROC AUC Curve", "Confusion Matrix"])
if plot_option == "ROC AUC Curve":
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    st.pyplot(fig)
elif plot_option == "Confusion Matrix":
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

st.markdown("---")
st.subheader("Prediksi Risiko Stroke")

with st.form("stroke_prediction_form"):
    gender = st.selectbox("Jenis Kelamin", options=["Male", "Female", "Other"])
    age = st.number_input("Umur", min_value=0, max_value=120, value=45)
    hypertension = st.selectbox("Hipertensi", options=["Tidak", "Ya"])
    heart_disease = st.selectbox("Penyakit Jantung", options=["Tidak", "Ya"])
    ever_married = st.selectbox("Pernah Menikah?", options=["No", "Yes"])
    work_type = st.selectbox("Jenis Pekerjaan", options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Tipe Tempat Tinggal", options=["Urban", "Rural"])
    avg_glucose = st.number_input("Rata-rata Glukosa", min_value=0.0, max_value=300.0, value=100.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0)
    smoking_status = st.selectbox("Status Merokok", options=["never smoked", "formerly smoked", "smokes", "Unknown"])
    submit = st.form_submit_button("Prediksi Risiko")

def map_inputs():
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    married_map = {"Yes": 1, "No": 0}
    residence_map = {"Urban": 1, "Rural": 0}
    base = dict.fromkeys(features, 0)
    base.update({
        "age": age,
        "hypertension": 1 if hypertension == "Ya" else 0,
        "heart_disease": 1 if heart_disease == "Ya" else 0,
        "avg_glucose_level": avg_glucose,
        "bmi": bmi,
        "gender": gender_map[gender],
        "ever_married_Yes": 1 if ever_married == "Yes" else 0,
        f"work_type_{work_type}": 1,
        "Residence_type": residence_map[residence_type],
        f"smoking_status_{smoking_status}": 1
    })
    return pd.DataFrame([base])

if submit:
    try:
        input_df = map_inputs()
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        st.subheader("Hasil Prediksi")
        if pred == 1:
            st.error(f"Pasien diprediksi BERISIKO STROKE ⚠️\n\nProbabilitas: **{prob:.2f}**")
        else:
            st.success(f"Pasien diprediksi TIDAK berisiko stroke ✅\n\nProbabilitas: **{prob:.2f}**")
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
