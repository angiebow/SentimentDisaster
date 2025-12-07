# **SentimentDisaster: Spatio-Temporal Disaster Sentiment Analysis**

**SentimentDisaster** is a comprehensive research framework and toolset for scraping, analyzing, and visualizing disaster-related news data. Beyond simple data collection, it integrates **Large Language Models (LLMs)** for entity extraction and sentiment interpretation, using a **Knowledge Distillation** pipeline to train efficient, lightweight machine-learning models.

While this project focuses on disaster mitigation insights for the Bali region (Klungkung, Tabanan, Bangli), its methodology is fully scalable for national or global use.

---

## 🚀 **Features**

### **🔍 Multi-Source Web Scraping**
Robust, modular scrapers supporting major Indonesian news platforms:

- **Detik**
- **CNN Indonesia**
- **Mongabay**

Includes **smart chronological pagination** to retrieve historical data within a specified timeframe (e.g., Nov 2022 – Nov 2025).

---

### **🧠 LLM-Powered Entity Extraction**
Extracts structured disaster information using Google Gemini or DeepSeek:

- Disaster Type (*banjir, gempa, tanah longsor*, etc.)
- Village/District-level locations
- Casualties and infrastructure damage
- Government and community response details

---

### **💬 Sentiment Analysis via Knowledge Distillation**
A hybrid LLM + ML pipeline:

- **Teacher Model:**  
  LLM generates silver-standard sentiment labels (*Positive / Negative / Neutral*) focusing on government response and community mood.

- **Student Models:**  
  Trains efficient offline models using LLM-labeled data:  
  - **SVM (Linear Kernel)**
  - **Random Forest (Ensemble)**

This enables fast, reproducible sentiment classification without continuous LLM usage.

---

### **🗺️ Spatio-Temporal Visualization**
- Converts extracted locations into Geographic Coordinates (Lat/Lon)
- Exports **GeoJSON** for interactive mapping (e.g., **Kepler.gl**)

---

## 🛠️ **Installation**

This project uses **Poetry** for environment and dependency management.

### Setup ringkas
- `poetry install`
- `poetry run python -m spacy download xx_ent_wiki_sm`
- Salin `.env.example` → `.env`, isi `OPENROUTER_API_KEY` (LLM labeling) dan `GOOGLE_API_KEY` jika perlu.

### Random Forest (LLM guru → RF siswa)
1) Preprocess:  
   `poetry run python random_forest/preprocess_merge.py --data-dir ./unified --pattern "processed_*.csv" --output random_forest/output_rf/merged_clean_rf.csv`
2) Label dengan LLM:  
   `poetry run python random_forest/label_with_llm.py --input random_forest/output_rf/merged_clean_rf.csv --output random_forest/output_rf/labeled_sentiment_data_rf.csv --api-key <OPENROUTER_API_KEY>`
   - Jika sudah ada `labeled_sentiment_data_unified.csv`, bisa langsung pakai itu (lewati 1–2).
3) Train & evaluasi RF:  
   `poetry run python random_forest/train_and_evaluate_rf.py --data random_forest/output_rf/labeled_sentiment_data_rf.csv --bigrams --class-weight balanced --n-estimators 800 --max-features 15000 --oversample-minority --neutral-weight 0.7`
   - Output model/vectorizer/report/CM di `random_forest/output_rf/`. Bisa ganti `--data` ke `labeled_sentiment_data_unified.csv` untuk banding dengan SVM.

### SVM (LLM guru → SVM siswa)
- Label + train:  
  `poetry run python svm/sentiment.py`
  - Input: `./unified/processed_*.csv`
  - Output label: `labeled_sentiment_data_unified.csv`
  - Model: `svm/sentiment_models/svm_sentiment_model_unified.pkl` + `svm/sentiment_models/tfidf_vectorizer.pkl`
- Utilitas:  
  - Confusion matrix plot: `poetry run python svm/confusion_matrix_plot.py`
  - Tambah sentimen ke GeoJSON: `poetry run python svm/sentiment_map.py`

### Membandingkan SVM vs Random Forest (paralel, tidak berurutan)
- Keduanya memakai **label LLM yang sama** (guru) untuk fairness; siswa yang berbeda (SVM vs RF).
- Jika sudah ada `labeled_sentiment_data_unified.csv`, langsung pakai file itu untuk keduanya (tanpa LLM ulang):
  - SVM (model di `svm/sentiment_models/` atau latih ulang dengan `svm/sentiment.py`).
  - RF: `poetry run python random_forest/train_and_evaluate_rf.py --data labeled_sentiment_data_unified.csv`
- Jika belum ada label, jalankan langkah labeling LLM sekali saja, lalu gunakan file label yang sama untuk SVM/RF. Dengan begitu keduanya berjalan paralel, tidak sequential satu sama lain.

### Struktur folder (rapi)
- `svm/`: semua skrip & model khusus SVM (sentiment.py, sentiment_map.py, confusion_matrix*.py, sentiment_models/).
- `random_forest/`: semua skrip & artefak khusus Random Forest.
- Akar repo (root): data universal & utilitas bersama (mis. `labeled_sentiment_data_unified.csv`, `final_data/`, `geojson/`, `location_extractor.py`, dsb). Pakai dari kedua pipeline.
