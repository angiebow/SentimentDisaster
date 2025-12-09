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

### Pipeline Bersama (LLM guru → siswa)
1) Preprocess (opsional) untuk membuat gabungan baru:  
  `poetry run python pipelines/preprocess_merge.py --data-dir ./unified --pattern "processed_*.csv" --output artifacts/random_forest/merged_clean_rf.csv`
2) Label dengan LLM satu kali untuk seluruh siswa:  
  `poetry run python pipelines/label_dataset.py --input-csv artifacts/random_forest/merged_clean_rf.csv --output shared_data/labeled_sentiment_data_unified.csv --api-key <OPENROUTER_API_KEY>`
  - Bisa juga langsung memakai pola default (`--input-pattern "unified/processed_*.csv"`) tanpa membuat file gabungan manual.
3) Latih model siswa menggunakan dataset yang sama:
  - SVM: `poetry run python pipelines/train_model.py --model svm --data shared_data/labeled_sentiment_data_unified.csv`
  - RF: `poetry run python pipelines/train_model.py --model rf --data shared_data/labeled_sentiment_data_unified.csv --bigrams --class-weight balanced --n-estimators 800 --max-features 15000 --oversample-minority --neutral-weight 0.7`

Keduanya memakai **file label yang sama** (mis. `shared_data/labeled_sentiment_data_unified.csv`) agar gap akurasi hanya berasal dari model, bukan dari data. Jalankan `pipelines/label_dataset.py` sekali saja lalu reuse hasilnya untuk perbandingan SVM ↔ RF.

### Utilitas Analisis & Visualisasi
- Confusion matrix SVM (training cepat + plot): `poetry run python tools/confusion_matrix_plot.py`
- Contoh matriks statis: `poetry run python tools/confusion_matrix.py`
- Tambah sentimen siswa (SVM) ke GeoJSON: `poetry run python tools/sentiment_map.py`

Seluruh utilitas di atas mengakses artefak siswa melalui `artifacts/svm/` sehingga tidak lagi bergantung pada folder `svm/` lama.

### Struktur folder (rapi)
- `pipelines/`: seluruh pipeline generik (`preprocess_merge.py`, `label_dataset.py`, `train_model.py`).
- `shared_data/`: hasil labeling gabungan (`labeled_sentiment_data_unified.csv`) yang dipakai bersama siswa.
- `artifacts/`: output pelatihan siswa (model/vectorizer SVM dan laporan Random Forest).
- `tools/`: utilitas analisis & visualisasi (confusion matrix, sentiment map, dll).
- Akar repo (root): data universal & utilitas bersama (`final_data/`, `geojson/`, `location_extractor.py`, dsb).
