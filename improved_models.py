# -*- coding: utf-8 -*-
"""
PERBAIKAN MODEL v2 — TARGET ACCURACY > 80%
============================================
Perbaikan tambahan:
1. Feature Engineering: fitur turunan dari domain knowledge pertanian
2. Decision Tree: tuning lebih luas + criterion entropy
3. Naive Bayes: PowerTransformer agar distribusi lebih Gaussian
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# TAHAP 1: MUAT DAN BERSIHKAN DATA
# ============================================================
print("=" * 70)
print("  MEMUAT DAN MEMBERSIHKAN DATA")
print("=" * 70)

path_file = './datasets/niladriroy0/agro-environmental-dataset/versions/1/agro_environmental_dataset.csv'
df = pd.read_csv(path_file)
print(f"Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

kolom_dibuang = [
    'location_id', 'failure_flag', 'suitability_score',
    'soil_ph', 'ph_stress_flag', 'nitrogen_ppm',
    'phosphorus_ppm', 'potassium_ppm', 'nutrient_balance'
]
df_selected = df.drop(columns=kolom_dibuang)
df_selected = df_selected.dropna()

kolom_kategorikal = ['soil_type', 'moisture_regime', 'thermal_regime', 'plant_category']
for col in kolom_kategorikal:
    df_selected[col] = df_selected[col].astype(str).str.lower().str.strip()

df_selected = df_selected.drop_duplicates()

kolom_numerik_sensor = ['soil_moisture_pct', 'soil_temp_c', 'air_temp_c', 'light_intensity_par']
for col in kolom_numerik_sensor:
    Q1 = df_selected[col].quantile(0.25)
    Q3 = df_selected[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_selected = df_selected[(df_selected[col] >= lower_bound) & (df_selected[col] <= upper_bound)]

print(f"Data setelah cleaning: {df_selected.shape[0]} baris")

# ============================================================
# TAHAP 2: FEATURE ENGINEERING (KUNCI PENINGKATAN PERFORMA!)
# ============================================================
print("\n" + "=" * 70)
print("  FEATURE ENGINEERING (FITUR TURUNAN BARU)")
print("=" * 70)

# Moisture features - seberapa jauh kondisi dari batas optimal
df_selected['moisture_range'] = df_selected['moisture_limit_wet'] - df_selected['moisture_limit_dry']

df_selected['moisture_position'] = (
    (df_selected['soil_moisture_pct'] - df_selected['moisture_limit_dry']) /
    df_selected['moisture_range'].replace(0, 1)
)

df_selected['moisture_below_dry'] = (
    df_selected['moisture_limit_dry'] - df_selected['soil_moisture_pct']
).clip(lower=0)

df_selected['moisture_above_wet'] = (
    df_selected['soil_moisture_pct'] - df_selected['moisture_limit_wet']
).clip(lower=0)

df_selected['moisture_deviation'] = np.where(
    df_selected['soil_moisture_pct'] < df_selected['moisture_limit_dry'],
    df_selected['moisture_limit_dry'] - df_selected['soil_moisture_pct'],
    np.where(
        df_selected['soil_moisture_pct'] > df_selected['moisture_limit_wet'],
        df_selected['soil_moisture_pct'] - df_selected['moisture_limit_wet'],
        0
    )
)

# Temperature features
df_selected['temp_diff'] = df_selected['soil_temp_c'] - df_selected['air_temp_c']
df_selected['temp_avg'] = (df_selected['soil_temp_c'] + df_selected['air_temp_c']) / 2

# Soil quality index
df_selected['soil_quality'] = (
    df_selected['organic_matter_pct'] * df_selected['cation_exchange_capacity'] /
    (df_selected['bulk_density'] * df_selected['salinity_ec']).replace(0, 1)
)

# Interaction: moisture position x buffering capacity
df_selected['moisture_buffering'] = df_selected['moisture_position'] * df_selected['buffering_capacity']

fitur_baru = [
    'moisture_range', 'moisture_position', 'moisture_below_dry',
    'moisture_above_wet', 'moisture_deviation', 'temp_diff',
    'temp_avg', 'soil_quality', 'moisture_buffering'
]
print(f"Fitur baru ditambahkan: {len(fitur_baru)}")
for f in fitur_baru:
    print(f"  + {f}")
print(f"Total fitur sekarang: {df_selected.shape[1] - 1} (sebelumnya 15)")

# ============================================================
# TAHAP 3: SPLIT -> ENCODE -> SCALE (TANPA DATA LEAKAGE)
# ============================================================
print("\n" + "=" * 70)
print("  SPLIT -> ENCODE -> SCALE (BEBAS DATA LEAKAGE)")
print("=" * 70)

X = df_selected.drop(columns=['stress_level'])
y = df_selected['stress_level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Data Latih : {X_train.shape[0]} baris, {X_train.shape[1]} fitur")
print(f"Data Uji   : {X_test.shape[0]} baris")

# Encoding
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train[kolom_kategorikal] = encoder.fit_transform(X_train[kolom_kategorikal])
X_test[kolom_kategorikal] = encoder.transform(X_test[kolom_kategorikal])

# Scaling
kolom_numerik = [col for col in X_train.columns if col not in kolom_kategorikal]
scaler = StandardScaler()
X_train[kolom_numerik] = scaler.fit_transform(X_train[kolom_numerik])
X_test[kolom_numerik] = scaler.transform(X_test[kolom_numerik])

print("Encoding & Scaling selesai (fit pada TRAINING saja)")

# ============================================================
# TAHAP 4: DECISION TREE DENGAN TUNING
# ============================================================
print("\n" + "=" * 70)
print("  MODEL 1: DECISION TREE + FEATURE ENGINEERING + TUNING")
print("=" * 70)

print("\n[A] Baseline (tanpa tuning)...")
dt_baseline = DecisionTreeClassifier(random_state=42)
dt_baseline.fit(X_train, y_train)
dt_bl_pred = dt_baseline.predict(X_test)
dt_bl_acc = accuracy_score(y_test, dt_bl_pred)
print(f"    Accuracy: {dt_bl_acc:.5f} | Depth: {dt_baseline.get_depth()} | Leaves: {dt_baseline.get_n_leaves()}")

print("\n[B] GridSearchCV tuning... (mohon tunggu)")
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 15, 20, 25, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
}

dt_grid = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=dt_param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=1,
    verbose=1
)
dt_grid.fit(X_train, y_train)

dt_best = dt_grid.best_estimator_
dt_pred = dt_best.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

print(f"\n    Best Params : {dt_grid.best_params_}")
print(f"    CV Score    : {dt_grid.best_score_:.5f}")
print(f"    Depth/Leaves: {dt_best.get_depth()}/{dt_best.get_n_leaves()}")

print("\n--- Hasil Evaluasi Decision Tree (Tuned + Feature Eng.) ---")
print(f"Accuracy  : {dt_acc:.5f}")
print(f"Precision : {precision_score(y_test, dt_pred, average='weighted'):.5f}")
print(f"Recall    : {recall_score(y_test, dt_pred, average='weighted'):.5f}")
print(f"F1-Score  : {f1_score(y_test, dt_pred, average='weighted'):.5f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, dt_pred))
print("\nClassification Report:")
print(classification_report(y_test, dt_pred,
      target_names=['0 (Normal)', '1 (Ringan)', '2 (Berat)']))
print(f"Peningkatan dari baseline: {dt_bl_acc:.5f} -> {dt_acc:.5f} ({(dt_acc-dt_bl_acc)*100:+.2f}%)")

# ============================================================
# TAHAP 5: NAIVE BAYES DENGAN POWER TRANSFORM + TUNING
# ============================================================
print("\n" + "=" * 70)
print("  MODEL 2: NAIVE BAYES + POWER TRANSFORM + TUNING")
print("=" * 70)

# Untuk Naive Bayes, gunakan PowerTransformer agar distribusi lebih Gaussian
print("\n[A] Menyiapkan data khusus NB dengan PowerTransformer...")
pt = PowerTransformer(method='yeo-johnson')
X_train_nb = X_train.copy()
X_test_nb = X_test.copy()
X_train_nb[kolom_numerik] = pt.fit_transform(X_train[kolom_numerik])
X_test_nb[kolom_numerik] = pt.transform(X_test[kolom_numerik])

print("\n[B] Baseline Naive Bayes...")
nb_baseline = GaussianNB()
nb_baseline.fit(X_train_nb, y_train)
nb_bl_pred = nb_baseline.predict(X_test_nb)
nb_bl_acc = accuracy_score(y_test, nb_bl_pred)
print(f"    Accuracy: {nb_bl_acc:.5f}")

print("\n[C] GridSearchCV tuning var_smoothing... (mohon tunggu)")
nb_param_grid = {
    'var_smoothing': np.logspace(-12, 0, 50)
}
nb_grid = GridSearchCV(
    estimator=GaussianNB(),
    param_grid=nb_param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=1,
    verbose=1
)
nb_grid.fit(X_train_nb, y_train)

nb_best = nb_grid.best_estimator_
nb_pred = nb_best.predict(X_test_nb)
nb_acc = accuracy_score(y_test, nb_pred)

print(f"\n    Best var_smoothing: {nb_grid.best_params_['var_smoothing']:.6e}")
print(f"    CV Score          : {nb_grid.best_score_:.5f}")

print("\n--- Hasil Evaluasi Naive Bayes (Tuned + Feature Eng. + PowerTransform) ---")
print(f"Accuracy  : {nb_acc:.5f}")
print(f"Precision : {precision_score(y_test, nb_pred, average='weighted'):.5f}")
print(f"Recall    : {recall_score(y_test, nb_pred, average='weighted'):.5f}")
print(f"F1-Score  : {f1_score(y_test, nb_pred, average='weighted'):.5f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, nb_pred))
print("\nClassification Report:")
print(classification_report(y_test, nb_pred,
      target_names=['0 (Normal)', '1 (Ringan)', '2 (Berat)']))
print(f"Peningkatan dari baseline: {nb_bl_acc:.5f} -> {nb_acc:.5f} ({(nb_acc-nb_bl_acc)*100:+.2f}%)")

# ============================================================
# TAHAP 6: RINGKASAN
# ============================================================
print("\n" + "=" * 70)
print("  RINGKASAN AKHIR")
print("=" * 70)

results = pd.DataFrame({
    'Model': [
        'DT Baseline', 'DT Tuned+FE',
        'NB Baseline', 'NB Tuned+PT+FE'
    ],
    'Accuracy': [dt_bl_acc, dt_acc, nb_bl_acc, nb_acc],
    'Precision': [
        precision_score(y_test, dt_bl_pred, average='weighted'),
        precision_score(y_test, dt_pred, average='weighted'),
        precision_score(y_test, nb_bl_pred, average='weighted'),
        precision_score(y_test, nb_pred, average='weighted')
    ],
    'F1-Score': [
        f1_score(y_test, dt_bl_pred, average='weighted'),
        f1_score(y_test, dt_pred, average='weighted'),
        f1_score(y_test, nb_bl_pred, average='weighted'),
        f1_score(y_test, nb_pred, average='weighted')
    ]
})
print(results.round(5).to_string(index=False))

print("\n" + "=" * 70)
print("  SELESAI")
print("=" * 70)
