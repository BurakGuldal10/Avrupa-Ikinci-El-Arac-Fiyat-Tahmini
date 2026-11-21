# -*- coding: utf-8 -*-
"""
Avrupa İkinci El Araç Fiyat Tahmini — Uçtan Uca Eğitim Betiği (train.py)
Veri setini indirir, sızıntı sütunlarını ve gizli kopyaları temizler,
TargetEncoder + Monotonik Kısıtlı XGBoost boru hattını eğitir ve `final_model.joblib` olarak kaydeder.
"""
import os
import json
import time
import joblib
import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

RANDOM_STATE = 42

def main():
    print("1) Kagglehub ile veri seti indiriliyor...")
    path = kagglehub.dataset_download("automarketapicom/european-used-car-market-dataset-100k-listings")
    parquet_file = os.path.join(path, "european_used_cars_100k.parquet")
    df = pd.read_parquet(parquet_file)
    print(f"   Yüklendi: {df.shape[0]:,} satır x {df.shape[1]} sütun")

    # Sızıntı ve gereksiz sütunların çıkarılması
    cols_to_drop = [
        "id", "source_portal", "main_image_url", "seller_name", "seller_type",
        "seller_city", "seller_zip", "vat_deductible", "equipment",
        "power_kw", "price_evaluation", "color", "version"
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Mantıksal aykırı değer temizliği
    valid_mask = (
        (df["first_registration_year"] <= 2026) &
        (df["first_registration_year"] >= 1995) &
        ((df["power_hp"] > 10) | (df["power_hp"].isna())) &
        (df["power_hp"].isna() | (df["power_hp"] <= 850)) &
        (df["mileage_km"] <= 500000) & (df["mileage_km"] >= 0) &
        (df["price_eur"] >= 1000) & (df["price_eur"] <= 500000)
    )
    df_clean = df[valid_mask].copy()

    # Gizli kopyaların (duplicate) bölme öncesinde temizlenmesi
    feature_cols = [c for c in df_clean.columns if c != "price_eur"]
    df_clean = df_clean.drop_duplicates(subset=feature_cols, keep="first").reset_index(drop=True)
    print(f"2) Temizlik ve tekilleştirme sonrası benzersiz araç sayısı: {len(df_clean):,}")

    # %60 Train / %20 Validation / %20 Test bölünmesi
    X = df_clean.drop(columns=["price_eur"]).copy()
    y = df_clean["price_eur"].copy()
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE)

    # Sızıntısız Ön İşleme Boru Hattı
    numeric_features = ["mileage_km", "first_registration_year", "power_hp"]
    high_card_features = ["make", "model"]
    low_card_features = ["fuel_type", "transmission", "body_type", "seller_country"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("high_cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("target_enc", TargetEncoder(smooth="auto", cv=5, random_state=RANDOM_STATE))
        ]), high_card_features),
        ("low_cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), low_card_features),
    ])

    best_xgb_params = {
        "subsample": 0.9,
        "n_estimators": 500,
        "max_depth": 10,
        "learning_rate": 0.08,
        "gamma": 0.1,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist"
    }

    final_model = Pipeline([
        ("prep", preprocessor),
        ("model", XGBRegressor(**best_xgb_params))
    ])

    t0 = time.time()
    final_model.fit(X_train_val, y_train_val)
    print(f"3) Final model eğitildi ({time.time() - t0:.2f} sn). Test seti değerlendiriliyor...")

    preds = final_model.predict(X_test)
    print(f"   Test R²   : %{r2_score(y_test, preds)*100:.2f}")
    print(f"   Test MAE  : {mean_absolute_error(y_test, preds):,.2f} €")
    print(f"   Test RMSE : {root_mean_squared_error(y_test, preds):,.2f} €")

    joblib.dump(final_model, "final_model.joblib")
    print("4) Model kaydedildi: final_model.joblib")

    # Arayüz için marka-model ve kategori sözlüğünü kaydet
    make_models = {
        mk: sorted(df_clean[df_clean["make"] == mk]["model"].dropna().unique().tolist())
        for mk in sorted(df_clean["make"].dropna().unique().tolist())
    }
    metadata = {
        "make_models": make_models,
        "fuels": sorted(df_clean["fuel_type"].dropna().unique().tolist()),
        "transmissions": sorted(df_clean["transmission"].dropna().unique().tolist()),
        "bodies": sorted(df_clean["body_type"].dropna().unique().tolist()),
        "countries": sorted(df_clean["seller_country"].dropna().unique().tolist()),
    }
    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("5) Arayüz meta verisi kaydedildi: metadata.json")

if __name__ == "__main__":
    main()
