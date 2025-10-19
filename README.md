# 🎣 CatchAI – Intelligent Fishing Log & Predictor

**CatchAI** is a lightweight Python project that turns your fishing logs into insights.  
It learns from your catch data — time, temperature, location, lure, and conditions —  
to predict **when you’re most likely to catch fish** next time.

Built for curiosity, data exploration, and a love of fishing.

---

## 🧠 Features

- 🪣 **Data ingestion:** CSV-based fishing log with weather & conditions  
- 📊 **Model training:** Logistic Regression baseline with engineered time features  
- 🕓 **Catch predictions:** “Top hours” table for your next trip  
- 🖼️ **AI vision (optional):** Auto-detect species & captions from fish photos using Azure AI Vision  
- 🧩 **Modular design:** Works offline with dataset, or connected with Vision API for photo tagging

---

## 🧰 Project Structure
CatchAI/
│
├── catchai_baseline.py # Core model training & top-hour predictor
├── catchai_vision.py # Azure AI Vision wrapper for captions & species
├── species_keywords.json # Local fish-species keyword mapping
├── catchai_dataset_template.csv # Sample dataset (replace with your own log)
├── .env # Azure endpoint & key (DO NOT COMMIT)
├── requirements.txt
└── README.md


## 🔗 LinkedIn-Ready Bullets

- Built an AI-powered fishing log and predictor using Python, scikit-learn, and Azure Vision.  
- Integrated image captioning & species detection with Azure AI Vision API.  
- Delivered reproducible results and clean data pipelines for portfolio demonstration.  
- Applied time-series and logistic regression to forecast optimal catch windows.  
