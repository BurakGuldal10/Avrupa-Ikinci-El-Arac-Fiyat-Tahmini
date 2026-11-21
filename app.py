# -*- coding: utf-8 -*-
"""
Avrupa İkinci El Araç Fiyat Tahmin Sistemi — Gradio Kullanıcı Arayüzü (app.py)
BTK Akademi Uçtan Uca Makine Öğrenmesi Projesi (Aşama 13 & 14)
"""
import os
import json
import joblib
import pandas as pd
import gradio as gr

MODEL_PATH = os.environ.get("MODEL_PATH", "final_model.joblib")
META_PATH = os.environ.get("META_PATH", "metadata.json")

DEFAULT_METADATA = {
    "make_models": {
        "BMW": ["X1", "X3", "X5", "320", "520", "118", "M3"],
        "Mercedes-Benz": ["A 180", "C 200", "E 220", "GLC 220", "CLA 200"],
        "Volkswagen": ["Golf", "Passat", "Tiguan", "Polo", "T-Roc", "ID.4"],
        "Audi": ["A3", "A4", "A6", "Q3", "Q5", "Q7"],
        "Nissan": ["Qashqai", "Juke", "Micra", "X-Trail", "Leaf"],
        "MINI": ["Cooper", "Countryman", "Clubman", "One"],
        "Skoda": ["Octavia", "Superb", "Kodiaq", "Kamiq", "Fabia"],
        "Toyota": ["Corolla", "Yaris", "RAV4", "C-HR", "Aygo"],
        "MG": ["ZS", "HS", "MG4", "EHS"],
        "Renault": ["Clio", "Megane", "Captur", "Kadjar", "Austral"],
        "Peugeot": ["208", "308", "2008", "3008", "5008"],
        "Ford": ["Focus", "Fiesta", "Kuga", "Puma", "Mondeo"],
    },
    "fuels": ["Diesel", "Benzin", "Elektro", "Hybrid (Benzin/Elektro)", "Hybrid (Diesel/Elektro)", "LPG"],
    "transmissions": ["Automatik", "Schaltgetriebe", "Halbautomatik"],
    "bodies": ["SUV", "Limousine", "Kombi", "Kleinwagen", "Cabrio", "Sportwagen/Coupé", "Van/Kleinbus"],
    "countries": ["DE", "AT", "IT", "NL", "ES", "FR", "BE", "LU"]
}

if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        META = json.load(f)
else:
    META = DEFAULT_METADATA

MAKE_MODELS = META["make_models"]
UNIQUE_MAKES = sorted(list(MAKE_MODELS.keys()))
UNIQUE_FUELS = META["fuels"]
UNIQUE_TRANS = META["transmissions"]
UNIQUE_BODIES = META["bodies"]
UNIQUE_COUNTRIES = META["countries"]

loaded_model = None
if os.path.exists(MODEL_PATH):
    loaded_model = joblib.load(MODEL_PATH)

def update_models_dropdown(selected_make):
    models = MAKE_MODELS.get(selected_make, [])
    default_val = models[0] if models else ""
    return gr.Dropdown(choices=models, value=default_val)

def predict_car_price(make, model, first_registration_year, mileage_km, power_hp,
                      fuel_type, transmission, body_type, seller_country):
    # 1. Hata ve Eksik Giriş Kontrolü (Yönerge Bölüm 6.1)
    if not make or not model:
        return "⚠️ Hata: Lütfen geçerli bir Araç Markası ve Modeli seçiniz."
    if mileage_km is None or power_hp is None or first_registration_year is None:
        return "⚠️ Hata: Yıl, Kilometre ve Motor Gücü alanları boş bırakılamaz."
    if mileage_km < 0 or mileage_km > 500000:
        return "⚠️ Uyarı: Kilometre değeri 0 ile 500.000 km arasında olmalıdır (Model eğitim sınırları)."
    if power_hp < 11 or power_hp > 850:
        return "⚠️ Uyarı: Motor gücü 11 HP ile 850 HP arasında olmalıdır."
    if first_registration_year < 1995 or first_registration_year > 2026:
        return "⚠️ Uyarı: Model yılı 1995 ile 2026 arasında olmalıdır."

    warnings = []
    if first_registration_year >= 2024 and mileage_km > 250000:
        warnings.append("ℹ️ Not: Yeni model yılına göre oldukça yüksek bir kilometre girdiniz; tahmin belirsizliği artabilir.")
    if power_hp < 75 and make in ["BMW", "Mercedes-Benz", "Porsche", "Land Rover", "Maserati"]:
        warnings.append("ℹ️ Not: Seçilen marka için olağandışı düşük bir beygir gücü girdiniz; eğitim verisinde bu kombinasyon nadir olabilir.")

    input_df = pd.DataFrame([{
        "make": make,
        "model": model,
        "first_registration_year": int(first_registration_year),
        "mileage_km": float(mileage_km),
        "power_hp": float(power_hp),
        "fuel_type": fuel_type,
        "transmission": transmission,
        "body_type": body_type,
        "seller_country": seller_country
    }])

    if loaded_model is None:
        return (
            "⚠️ `final_model.joblib` dosyası henüz bulunamadı. "
            "Lütfen önce `python train.py` komutunu çalıştırarak veya Colab üzerinden modeli indirerek aynı klasöre yerleştirin."
        )

    prediction = float(loaded_model.predict(input_df)[0])
    prediction = max(1000.0, prediction)
    low_band = prediction * 0.90
    high_band = prediction * 1.10

    msg = (
        f"🚗 Tahmini Piyasa Değeri: {prediction:,.2f} €\n"
        f"📊 Beklenen Piyasa Aralığı (±%10 MdAPE Bandı): {low_band:,.0f} € – {high_band:,.0f} €"
    )
    if warnings:
        msg += "\n\n" + "\n".join(warnings)
    return msg

with gr.Blocks(title="Avrupa İkinci El Araç Fiyat Tahmin Sistemi") as demo:
    gr.Markdown("# 🇪🇺 Avrupa İkinci El Araç Fiyat Tahmin Sistemi")
    gr.Markdown(
        "**BTK Akademi Uçtan Uca Makine Öğrenmesi Projesi** | "
        "91.146 benzersiz Avrupa araç ilanı üzerinde sızıntısız boru hattı (Pipeline) ile eğitilmiş **Optimize XGBoost** modeli."
    )

    with gr.Row():
        with gr.Column():
            default_make = "BMW" if "BMW" in UNIQUE_MAKES else UNIQUE_MAKES[0]
            make_input = gr.Dropdown(choices=UNIQUE_MAKES, label="Araç Markası", value=default_make)
            initial_models = MAKE_MODELS.get(default_make, [])
            default_model = "X3" if "X3" in initial_models else (initial_models[0] if initial_models else "")
            model_input = gr.Dropdown(choices=initial_models, label="Araç Modeli", value=default_model)
            year_input = gr.Slider(minimum=1995, maximum=2026, step=1, value=2020, label="İlk Tescil / Model Yılı")
            km_input = gr.Number(value=50000, minimum=0, maximum=500000, label="Toplam Kilometre (km)")
            hp_input = gr.Number(value=190, minimum=11, maximum=850, label="Motor Gücü (HP)")

        with gr.Column():
            fuel_input = gr.Dropdown(choices=UNIQUE_FUELS, label="Yakıt Türü", value="Diesel" if "Diesel" in UNIQUE_FUELS else UNIQUE_FUELS[0])
            trans_input = gr.Dropdown(choices=UNIQUE_TRANS, label="Şanzıman / Vites Türü", value="Automatik" if "Automatik" in UNIQUE_TRANS else UNIQUE_TRANS[0])
            body_input = gr.Dropdown(choices=UNIQUE_BODIES, label="Kasa Tipi", value="SUV" if "SUV" in UNIQUE_BODIES else UNIQUE_BODIES[0])
            country_input = gr.Dropdown(choices=UNIQUE_COUNTRIES, label="Satıcı Ülkesi", value="DE" if "DE" in UNIQUE_COUNTRIES else UNIQUE_COUNTRIES[0])

            predict_btn = gr.Button("🔮 Adil Piyasa Fiyatını Hesapla", variant="primary")
            output_text = gr.Textbox(label="Tahmin Sonucu ve Piyasa Bandı", lines=4)

    gr.Markdown(
        "---\n"
        "⚠️ **Kullanım Sınırları ve Yasal Uyarı (Etik İlke):** Bu tahmin sistemi halka açık ilan verileriyle eğitilmiş bir makine öğrenmesi "
        "modelidir (Test R²: %94,03 | Tipik Medyan Hata MdAPE: ~%9,3). İlan fiyatları nihai satış fiyatından farklı olabilir ve model aracın "
        "hasar/tramer geçmişini, boya durumunu veya özel donanım paketlerini içermez. Bağlayıcı ekspertiz veya finansal yatırım tavsiyesi niteliği taşımaz."
    )

    make_input.change(fn=update_models_dropdown, inputs=make_input, outputs=model_input)
    predict_btn.click(
        fn=predict_car_price,
        inputs=[make_input, model_input, year_input, km_input, hp_input, fuel_input, trans_input, body_input, country_input],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()
