# ==========================================================================
# CELL BARU: Ganti cell "Data Transformation" dan "Machine Learning"
# di Data_Mining.ipynb dengan kode di bawah ini.
#
# INSTRUKSI:
# 1. Di notebook, HAPUS cell "Data Transformation" (cell ke-6)
# 2. HAPUS cell "Machine Learning" pertama (cell ke-7, yang DT + NB)
# 3. Tambahkan 3 cell baru di posisi yang sama, isinya dari kode di bawah
# ==========================================================================


# ===================== CELL BARU 1: Feature Engineering =====================
# Paste kode di bawah ini ke cell baru setelah cell "Data Pre-processing"

# ASUMSI: df_selected sudah ada dari tahap cleaning sebelumnya

print("=" * 60)
print("  TAHAP 4: FEATURE ENGINEERING")
print("=" * 60)

# Fitur turunan dari domain knowledge pertanian
# 1. Moisture features
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

import numpy as np
df_selected['moisture_deviation'] = np.where(
    df_selected['soil_moisture_pct'] < df_selected['moisture_limit_dry'],
    df_selected['moisture_limit_dry'] - df_selected['soil_moisture_pct'],
    np.where(
        df_selected['soil_moisture_pct'] > df_selected['moisture_limit_wet'],
        df_selected['soil_moisture_pct'] - df_selected['moisture_limit_wet'],
        0
    )
)

# 2. Temperature features
df_selected['temp_diff'] = df_selected['soil_temp_c'] - df_selected['air_temp_c']
df_selected['temp_avg'] = (df_selected['soil_temp_c'] + df_selected['air_temp_c']) / 2

# 3. Soil quality index
df_selected['soil_quality'] = (
    df_selected['organic_matter_pct'] * df_selected['cation_exchange_capacity'] /
    (df_selected['bulk_density'] * df_selected['salinity_ec']).replace(0, 1)
)

# 4. Interaction feature
df_selected['moisture_buffering'] = df_selected['moisture_position'] * df_selected['buffering_capacity']

print(f"Fitur baru: moisture_range, moisture_position, moisture_below_dry,")
print(f"            moisture_above_wet, moisture_deviation, temp_diff,")
print(f"            temp_avg, soil_quality, moisture_buffering")
print(f"Total fitur: {df_selected.shape[1] - 1} kolom (sebelumnya 15)")
print(f"Dimensi data: {df_selected.shape}")


# =============== CELL BARU 2: Data Transformation (Diperbaiki) ===============
# Paste kode di bawah ini ke cell baru berikutnya

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

print("=" * 60)
print("  TAHAP 5: SPLIT -> ENCODE -> SCALE (BEBAS DATA LEAKAGE)")
print("=" * 60)

# PERBAIKAN KRITIS: Split DULU, baru transform
X = df_selected.drop(columns=['stress_level'])
y = df_selected['stress_level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Data Latih : {X_train.shape[0]} baris, {X_train.shape[1]} fitur")
print(f"Data Uji   : {X_test.shape[0]} baris")

# Encoding - fit pada TRAINING saja
kolom_kategorikal = ['soil_type', 'moisture_regime', 'thermal_regime', 'plant_category']
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train[kolom_kategorikal] = encoder.fit_transform(X_train[kolom_kategorikal])
X_test[kolom_kategorikal] = encoder.transform(X_test[kolom_kategorikal])

# Scaling - fit pada TRAINING saja
kolom_numerik = [col for col in X_train.columns if col not in kolom_kategorikal]
scaler = StandardScaler()
X_train[kolom_numerik] = scaler.fit_transform(X_train[kolom_numerik])
X_test[kolom_numerik] = scaler.transform(X_test[kolom_numerik])

print("\nEncoder & Scaler di-fit pada DATA TRAINING saja (no data leakage)")
print("Transformasi selesai. Data siap untuk modeling.")


# ============== CELL BARU 3: Decision Tree & Naive Bayes (Tuned) ==============
# Paste kode di bawah ini ke cell baru berikutnya

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  TAHAP 6: EKSPERIMEN MODELING (DECISION TREE & NAIVE BAYES)")
print("=" * 60)

# Fungsi visualisasi confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['0 (Normal)', '1 (Ringan)', '2 (Berat)'],
                yticklabels=['0 (Normal)', '1 (Ringan)', '2 (Berat)'])
    plt.title(title, fontweight='bold', pad=15)
    plt.xlabel('Prediksi (Predicted Class)', labelpad=10)
    plt.ylabel('Aktual (Actual Class)', labelpad=10)
    plt.tight_layout()
    plt.show()

# ==========================================
# MODEL 1: DECISION TREE (TUNED)
# ==========================================
print("\n>>> Memproses Decision Tree dengan GridSearchCV...")
print("    Mencari hyperparameter terbaik... (mohon tunggu)\n")

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

print(f"\nParameter Terbaik: {dt_grid.best_params_}")
print(f"CV Score Terbaik : {dt_grid.best_score_:.5f}")

print("\n--- Hasil Evaluasi Decision Tree (Tuned) ---")
print(f"Accuracy  : {accuracy_score(y_test, dt_pred):.5f}")
print(f"Precision : {precision_score(y_test, dt_pred, average='weighted'):.5f}")
print(f"Recall    : {recall_score(y_test, dt_pred, average='weighted'):.5f}")
print(f"F1-Score  : {f1_score(y_test, dt_pred, average='weighted'):.5f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, dt_pred))

# Visualisasi
plot_confusion_matrix(y_test, dt_pred, 'Confusion Matrix - Decision Tree (Tuned)')

print("-" * 60)

# ==========================================
# MODEL 2: NAIVE BAYES (TUNED)
# ==========================================
print("\n>>> Memproses Naive Bayes dengan GridSearchCV...")
print("    Mencari var_smoothing terbaik... (mohon tunggu)\n")

nb_param_grid = {
    'var_smoothing': np.logspace(-12, -1, 30)
}

nb_grid = GridSearchCV(
    estimator=GaussianNB(),
    param_grid=nb_param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=1,
    verbose=1
)
nb_grid.fit(X_train, y_train)

nb_best = nb_grid.best_estimator_
nb_pred = nb_best.predict(X_test)

print(f"\nParameter Terbaik: {nb_grid.best_params_}")
print(f"CV Score Terbaik : {nb_grid.best_score_:.5f}")

print("\n--- Hasil Evaluasi Naive Bayes (Tuned) ---")
print(f"Accuracy  : {accuracy_score(y_test, nb_pred):.5f}")
print(f"Precision : {precision_score(y_test, nb_pred, average='weighted'):.5f}")
print(f"Recall    : {recall_score(y_test, nb_pred, average='weighted'):.5f}")
print(f"F1-Score  : {f1_score(y_test, nb_pred, average='weighted'):.5f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, nb_pred))

# Visualisasi
plot_confusion_matrix(y_test, nb_pred, 'Confusion Matrix - Naive Bayes (Tuned)')

# ==========================================
# RINGKASAN
# ==========================================
print("\n" + "=" * 60)
print("  RINGKASAN PERBANDINGAN")
print("=" * 60)

results = pd.DataFrame({
    'Model': ['Decision Tree (Tuned)', 'Naive Bayes (Tuned)'],
    'Accuracy': [
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, nb_pred)
    ],
    'Precision': [
        precision_score(y_test, dt_pred, average='weighted'),
        precision_score(y_test, nb_pred, average='weighted')
    ],
    'Recall': [
        recall_score(y_test, dt_pred, average='weighted'),
        recall_score(y_test, nb_pred, average='weighted')
    ],
    'F1-Score': [
        f1_score(y_test, dt_pred, average='weighted'),
        f1_score(y_test, nb_pred, average='weighted')
    ]
})
print(results.round(5).to_string(index=False))
print("\n" + "=" * 60)
print("  EKSPERIMEN SELESAI")
print("=" * 60)

# ==========================================
# SIMPAN MODEL DECISION TREE KE FILE .PKL
# ==========================================


# Simpan juga encoder dan scaler (dibutuhkan saat prediksi data baru)
joblib.dump(encoder, './models/ordinal_encoder.pkl')
joblib.dump(scaler, './models/standard_scaler.pkl')
print(f"Encoder disimpan ke: ./models/ordinal_encoder.pkl")
print(f"Scaler disimpan ke : ./models/standard_scaler.pkl")
