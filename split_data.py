# -*- coding: utf-8 -*-
"""
Script untuk memisahkan data train dan test ke folder yang berbeda.
Menjalankan pipeline preprocessing yang sama lalu menyimpan hasilnya.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Baca Dataset
# ==========================================
path_file = './datasets/niladriroy0/agro-environmental-dataset/versions/1/agro_environmental_dataset.csv'
df = pd.read_csv(path_file)
print(f"Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

# ==========================================
# 2. Data Selection (Drop kolom tidak relevan)
# ==========================================
kolom_dibuang = [
    'location_id', 'failure_flag', 'suitability_score',
    'soil_ph', 'ph_stress_flag', 'nitrogen_ppm',
    'phosphorus_ppm', 'potassium_ppm', 'nutrient_balance'
]
df_selected = df.drop(columns=kolom_dibuang)

# ==========================================
# 3. Data Cleaning
# ==========================================
# Missing values
df_selected = df_selected.dropna()

# Inkonsistensi teks
kolom_kategorikal = ['soil_type', 'moisture_regime', 'thermal_regime', 'plant_category']
for col in kolom_kategorikal:
    df_selected[col] = df_selected[col].astype(str).str.lower().str.strip()

# Duplikat
df_selected = df_selected.drop_duplicates()

# Outlier (IQR)
kolom_numerik_sensor = ['soil_moisture_pct', 'soil_temp_c', 'air_temp_c', 'light_intensity_par']
for col in kolom_numerik_sensor:
    Q1 = df_selected[col].quantile(0.25)
    Q3 = df_selected[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_selected = df_selected[(df_selected[col] >= lower_bound) & (df_selected[col] <= upper_bound)]

print(f"Data setelah cleaning: {df_selected.shape[0]} baris")

# ==========================================
# 4. Data Transformation
# ==========================================
df_transformed = df_selected.copy()

# Label Encoding
le = LabelEncoder()
for col in kolom_kategorikal:
    df_transformed[col] = le.fit_transform(df_transformed[col])

# Standard Scaling
kolom_numerik = [col for col in df_transformed.columns if col not in kolom_kategorikal and col != 'stress_level']
scaler = StandardScaler()
df_transformed[kolom_numerik] = scaler.fit_transform(df_transformed[kolom_numerik])

# ==========================================
# 5. Split dan Simpan ke Folder Terpisah
# ==========================================
X = df_transformed.drop(columns=['stress_level'])
y = df_transformed['stress_level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Gabungkan X dan y kembali untuk disimpan sebagai CSV lengkap
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Buat folder
train_dir = os.path.join('.', 'data', 'train')
test_dir = os.path.join('.', 'data', 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Simpan file CSV
train_path = os.path.join(train_dir, 'train_data.csv')
test_path = os.path.join(test_dir, 'test_data.csv')

train_data.to_csv(train_path, index=False)
test_data.to_csv(test_path, index=False)

print(f"\n{'='*60}")
print(f"  DATA BERHASIL DIPISAHKAN KE FOLDER BERBEDA")
print(f"{'='*60}")
print(f"Data Training : {train_data.shape[0]} baris -> {train_path}")
print(f"Data Testing  : {test_data.shape[0]} baris -> {test_path}")
print(f"{'='*60}")
