# -*- coding: utf-8 -*-
"""
Dashboard Data Mining - Agro Environmental Stress Prediction
Jalankan: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Data Mining Dashboard", layout="wide")

# ==================== LOAD & PROCESS ====================
@st.cache_data
def load_data():
    path = './datasets/niladriroy0/agro-environmental-dataset/versions/1/agro_environmental_dataset.csv'
    df_raw = pd.read_csv(path)

    kolom_dibuang = [
        'location_id', 'failure_flag', 'suitability_score',
        'soil_ph', 'ph_stress_flag', 'nitrogen_ppm',
        'phosphorus_ppm', 'potassium_ppm', 'nutrient_balance'
    ]
    df = df_raw.drop(columns=kolom_dibuang).dropna()

    kolom_kat = ['soil_type', 'moisture_regime', 'thermal_regime', 'plant_category']
    for col in kolom_kat:
        df[col] = df[col].astype(str).str.lower().str.strip()
    df = df.drop_duplicates()

    kolom_sensor = ['soil_moisture_pct', 'soil_temp_c', 'air_temp_c', 'light_intensity_par']
    for col in kolom_sensor:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    return df_raw, df

@st.cache_data
def evaluate_model(df):
    kolom_kat = ['soil_type', 'moisture_regime', 'thermal_regime', 'plant_category']
    X = df.drop(columns=['stress_level'])
    y = df['stress_level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[kolom_kat] = encoder.fit_transform(X_train[kolom_kat])
    X_test[kolom_kat] = encoder.transform(X_test[kolom_kat])

    kolom_num = [c for c in X_train.columns if c not in kolom_kat]
    scaler = StandardScaler()
    X_train[kolom_num] = scaler.fit_transform(X_train[kolom_num])
    X_test[kolom_num] = scaler.transform(X_test[kolom_num])

    model = joblib.load('./models/decision_tree_tuned.pkl')
    y_pred = model.predict(X_test)

    return y_test, y_pred, model, encoder, scaler

# ==================== LOAD ====================
df_raw, df_clean = load_data()
y_test, y_pred, model, encoder, scaler = evaluate_model(df_clean)

# ==================== HEADER ====================
st.title("Dashboard Data Mining - Prediksi Stres Tanaman")
st.caption("Dataset: Agro-Environmental | Model: Decision Tree")
st.divider()

# ==================== ROW 1: DATASET OVERVIEW ====================
st.subheader("Ringkasan Dataset")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Baris (Raw)", f"{df_raw.shape[0]:,}")
c2.metric("Baris (Clean)", f"{df_clean.shape[0]:,}")
c3.metric("Fitur", df_clean.shape[1] - 1)
c4.metric("Data Train", f"{len(y_test) * 4:,}")
c5.metric("Data Test", f"{len(y_test):,}")

# ==================== ROW 2: MODEL METRICS + CONFUSION MATRIX ====================
st.divider()
st.subheader("Evaluasi Model Decision Tree (Tuned)")

col_metric, col_cm = st.columns([1, 1.5])

with col_metric:
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    m1, m2 = st.columns(2)
    m1.metric("Accuracy", f"{acc:.4f}")
    m2.metric("Precision", f"{prec:.4f}")
    m3, m4 = st.columns(2)
    m3.metric("Recall", f"{rec:.4f}")
    m4.metric("F1-Score", f"{f1:.4f}")

    st.markdown("**Hyperparameter:**")
    params = model.get_params()
    st.code(f"criterion  : {params['criterion']}\n"
            f"max_depth  : {params['max_depth']}\n"
            f"min_split  : {params['min_samples_split']}\n"
            f"min_leaf   : {params['min_samples_leaf']}\n"
            f"tree_depth : {model.get_depth()}\n"
            f"n_leaves   : {model.get_n_leaves()}", language=None)

with col_cm:
    labels = ['0 (Normal)', '1 (Ringan)', '2 (Berat)']
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title('Confusion Matrix - Decision Tree', fontweight='bold', pad=12)
    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Aktual')
    plt.tight_layout()
    st.pyplot(fig)

# ==================== ROW 3: CLASSIFICATION REPORT + DISTRIBUSI ====================
st.divider()
col_report, col_dist = st.columns(2)

with col_report:
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    report_df = pd.DataFrame(report).T.round(4)
    st.dataframe(report_df, width=600)

with col_dist:
    st.subheader("Distribusi Kelas Target")
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    df_clean['stress_level'].value_counts().sort_index().plot(kind='bar', ax=ax2, color=colors)
    ax2.set_xticklabels(['0 (Normal)', '1 (Ringan)', '2 (Berat)'], rotation=0)
    ax2.set_ylabel("Jumlah")
    ax2.set_title("Distribusi Stress Level", fontweight='bold')
    for i, v in enumerate(df_clean['stress_level'].value_counts().sort_index()):
        ax2.text(i, v + 500, f'{v:,}', ha='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig2)

# ==================== ROW 4: PREVIEW DATA ====================
st.divider()
st.subheader("Preview Data (10 baris pertama)")
st.dataframe(df_clean.head(10), width=1400)

# ==================== ROW 5: TEST MODEL ====================
st.divider()
st.subheader("🧪 Test Model")
st.caption("Pilih mode pengujian: Single Test untuk input manual, atau Batch Test untuk menguji beberapa data sekaligus dari dataset.")

kolom_kat = ['soil_type', 'moisture_regime', 'thermal_regime', 'plant_category']

tab_single, tab_batch = st.tabs(["🔬 Single Test", "📊 Batch Test"])

# -------------------- SINGLE TEST --------------------
with tab_single:
    st.markdown("#### Input Data Manual")
    st.caption("Masukkan parameter lingkungan untuk memprediksi level stres tanaman.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🌱 Properti Tanah**")
        soil_type = st.selectbox("Tipe Tanah", sorted(df_clean['soil_type'].unique()), key="single_soil")
        bulk_density = st.number_input("Bulk Density", 0.5, 2.5, 1.3, 0.1, key="single_bd")
        organic_matter = st.number_input("Organic Matter (%)", 0.5, 20.0, 3.0, 0.5, key="single_om")
        cation_ec = st.number_input("Cation Exchange Capacity", 1.0, 50.0, 15.0, 1.0, key="single_cec")
        salinity = st.number_input("Salinity EC", 0.01, 2.0, 0.4, 0.05, key="single_sal")
        buffering = st.number_input("Buffering Capacity", 0.1, 1.0, 0.6, 0.05, key="single_buf")

    with col2:
        st.markdown("**💧 Kelembaban**")
        soil_moisture = st.number_input("Soil Moisture (%)", 0.0, 80.0, 35.0, 1.0, key="single_sm")
        moisture_dry = st.number_input("Moisture Limit Dry", 1.0, 50.0, 15.0, 1.0, key="single_md")
        moisture_wet = st.number_input("Moisture Limit Wet", 10.0, 80.0, 40.0, 1.0, key="single_mw")
        moisture_regime = st.selectbox("Moisture Regime", sorted(df_clean['moisture_regime'].unique()), key="single_mr")

    with col3:
        st.markdown("**🌡️ Suhu & Cahaya**")
        soil_temp = st.number_input("Soil Temp (C)", 5.0, 45.0, 25.0, 0.5, key="single_st")
        air_temp = st.number_input("Air Temp (C)", 5.0, 50.0, 28.0, 0.5, key="single_at")
        thermal_regime = st.selectbox("Thermal Regime", sorted(df_clean['thermal_regime'].unique()), key="single_tr")
        light = st.number_input("Light Intensity PAR", 50.0, 2500.0, 600.0, 50.0, key="single_li")
        plant_cat = st.selectbox("Plant Category", sorted(df_clean['plant_category'].unique()), key="single_pc")

    if st.button("🔍 Prediksi Stress Level", type="primary", use_container_width=True, key="btn_single"):
        input_data = pd.DataFrame([{
            'soil_type': soil_type, 'bulk_density': bulk_density,
            'organic_matter_pct': organic_matter, 'cation_exchange_capacity': cation_ec,
            'salinity_ec': salinity, 'buffering_capacity': buffering,
            'soil_moisture_pct': soil_moisture, 'moisture_limit_dry': moisture_dry,
            'moisture_limit_wet': moisture_wet, 'moisture_regime': moisture_regime,
            'soil_temp_c': soil_temp, 'air_temp_c': air_temp,
            'thermal_regime': thermal_regime, 'light_intensity_par': light,
            'plant_category': plant_cat,
        }])

        input_data[kolom_kat] = encoder.transform(input_data[kolom_kat])
        kolom_num = [c for c in input_data.columns if c not in kolom_kat]
        input_data[kolom_num] = scaler.transform(input_data[kolom_num])

        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        confidence = proba[pred] * 100

        stress_map = {0: ("Normal", "success"), 1: ("Ringan", "warning"), 2: ("Berat", "error")}
        label, alert = stress_map[pred]

        st.divider()
        getattr(st, alert)(f"**Hasil Prediksi: Stress Level {pred} ({label}) | Confidence: {confidence:.2f}%**")

        st.markdown("**Probabilitas tiap kelas:**")
        prob_cols = st.columns(3)
        for i, (lbl, p) in enumerate(zip(['Normal (0)', 'Ringan (1)', 'Berat (2)'], proba)):
            prob_cols[i].metric(lbl, f"{p*100:.2f}%")

# -------------------- BATCH TEST --------------------
with tab_batch:
    st.markdown("#### Batch Test - Random Data dari Dataset")
    st.caption("Ambil data secara acak dari dataset untuk menguji performa model pada banyak sampel sekaligus.")

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        n_samples = st.slider("Jumlah sampel", min_value=5, max_value=100, value=20, step=5, key="batch_n")
    with col_cfg2:
        random_seed = st.number_input("Random Seed (untuk reproduksi)", min_value=0, max_value=9999, value=42, step=1, key="batch_seed")

    if st.button("🚀 Jalankan Batch Test", type="primary", use_container_width=True, key="btn_batch"):
        # Sample random data from cleaned dataset
        sample_size = min(n_samples, len(df_clean))
        df_sample = df_clean.sample(n=sample_size, random_state=int(random_seed)).copy()
        y_actual = df_sample['stress_level'].values

        # Prepare features
        X_sample = df_sample.drop(columns=['stress_level']).copy()
        X_sample_encoded = X_sample.copy()
        X_sample_encoded[kolom_kat] = encoder.transform(X_sample_encoded[kolom_kat])
        kolom_num = [c for c in X_sample_encoded.columns if c not in kolom_kat]
        X_sample_encoded[kolom_num] = scaler.transform(X_sample_encoded[kolom_num])

        # Predict
        y_batch_pred = model.predict(X_sample_encoded)
        y_batch_proba = model.predict_proba(X_sample_encoded)

        # ---- Metrics Summary ----
        st.divider()
        st.markdown("### 📈 Hasil Batch Test")

        batch_acc = accuracy_score(y_actual, y_batch_pred)
        batch_prec = precision_score(y_actual, y_batch_pred, average='weighted', zero_division=0)
        batch_rec = recall_score(y_actual, y_batch_pred, average='weighted', zero_division=0)
        batch_f1 = f1_score(y_actual, y_batch_pred, average='weighted', zero_division=0)
        n_correct = (y_actual == y_batch_pred).sum()
        n_wrong = sample_size - n_correct

        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        mc1.metric("Total Sampel", sample_size)
        mc2.metric("✅ Benar", n_correct)
        mc3.metric("❌ Salah", n_wrong)
        mc4.metric("Accuracy", f"{batch_acc:.4f}")
        mc5.metric("Precision", f"{batch_prec:.4f}")
        mc6.metric("F1-Score", f"{batch_f1:.4f}")

        # ---- Confusion Matrix & Distribution ----
        col_bcm, col_bdist = st.columns(2)

        with col_bcm:
            st.markdown("**Confusion Matrix (Batch)**")
            labels_batch = ['0 (Normal)', '1 (Ringan)', '2 (Berat)']
            cm_batch = confusion_matrix(y_actual, y_batch_pred, labels=[0, 1, 2])
            fig_bcm, ax_bcm = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm_batch, annot=True, fmt='d', cmap='Purples', cbar=False,
                        xticklabels=labels_batch, yticklabels=labels_batch, ax=ax_bcm)
            ax_bcm.set_title('Confusion Matrix - Batch Test', fontweight='bold', pad=12)
            ax_bcm.set_xlabel('Prediksi')
            ax_bcm.set_ylabel('Aktual')
            plt.tight_layout()
            st.pyplot(fig_bcm)

        with col_bdist:
            st.markdown("**Perbandingan Aktual vs Prediksi**")
            compare_df = pd.DataFrame({
                'Aktual': pd.Series(y_actual).value_counts().reindex([0, 1, 2], fill_value=0),
                'Prediksi': pd.Series(y_batch_pred).value_counts().reindex([0, 1, 2], fill_value=0),
            })
            compare_df.index = ['0 (Normal)', '1 (Ringan)', '2 (Berat)']
            fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
            compare_df.plot(kind='bar', ax=ax_bar, color=['#3498db', '#e74c3c'], edgecolor='white')
            ax_bar.set_title('Distribusi Aktual vs Prediksi', fontweight='bold')
            ax_bar.set_xlabel('Stress Level')
            ax_bar.set_ylabel('Jumlah')
            ax_bar.set_xticklabels(compare_df.index, rotation=0)
            ax_bar.legend(loc='upper right')
            plt.tight_layout()
            st.pyplot(fig_bar)

        # ---- Detail Table ----
        st.markdown("### 📋 Detail Hasil per Sampel")

        stress_label_map = {0: 'Normal', 1: 'Ringan', 2: 'Berat'}
        result_df = X_sample.copy()
        result_df.insert(0, 'No', range(1, sample_size + 1))
        result_df['Aktual'] = [f"{v} ({stress_label_map[v]})" for v in y_actual]
        result_df['Prediksi'] = [f"{v} ({stress_label_map[v]})" for v in y_batch_pred]
        result_df['Confidence (%)'] = [f"{y_batch_proba[i][y_batch_pred[i]] * 100:.2f}" for i in range(sample_size)]
        result_df['Status'] = ['✅ Benar' if a == p else '❌ Salah' for a, p in zip(y_actual, y_batch_pred)]

        # Color the status column
        def highlight_status(row):
            if row['Status'] == '✅ Benar':
                return ['background-color: #03ad06'] * len(row)
            else:
                return ['background-color: #b80611'] * len(row)

        st.dataframe(
            result_df.style.apply(highlight_status, axis=1),
            use_container_width=True,
            height=min(400, 35 * sample_size + 38)
        )
