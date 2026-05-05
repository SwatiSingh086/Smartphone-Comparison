# Smartphone-Comparison
# 📱 Smartphone Analysis Platform

A Streamlit-based web application for analysing, comparing, and getting personalised recommendations for smartphones using data scraped from **Flipkart**, **Amazon**, and **Cashify**. Sentiment analysis is powered by a DistilBERT model (HuggingFace Transformers), with TF-IDF aspect extraction and a content-based recommendation engine.

---

## ✨ Features

| Page | What it does |
|---|---|
| **Product Comparison** | Compare phones across platforms with brand/price filters |
| **Sentiment Analysis** | Aspect-level sentiment (camera, battery, performance, display, build, value) extracted from user reviews via DistilBERT |
| **Price Analysis** | Interactive charts showing price distributions and discounts |
| **Recommendations** | Personalised ranked recommendations based on your feature preferences |

---

## 🗂️ Project Structure

```
ParentPortfolioHub/
├── app.py                   # Main Streamlit application entry point
├── data_processor.py        # CSV loading, cleaning, and feature engineering
├── sentiment_analyzer.py    # DistilBERT sentiment pipeline + TF-IDF aspect extraction
├── recommendation_engine.py # Weighted scoring & recommendation logic
├── visualization.py         # Plotly chart helpers
├── pyproject.toml           # Python project metadata & dependencies
├── sentiment_cache.pkl      # Auto-generated cache after first run (do not delete)
├── .streamlit/
│   └── config.toml          # Streamlit server configuration (port 5000)
└── attached_assets/
    ├── flipkart_phones.csv
    ├── amazon_phones.csv
    └── cashify_phones.csv
```

---

## 🔧 Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.11 |
| pip / uv | Latest |
| Git | Any |
| ~2 GB free disk space | For PyTorch + DistilBERT model weights |

> **Note:** The DistilBERT model (~67 MB) is downloaded automatically from HuggingFace on the first run and cached locally. An internet connection is required for that first run only.

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd ParentPortfolioHub
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If there is no `requirements.txt`, install directly from `pyproject.toml` using `pip`:

```bash
pip install numpy pandas plotly streamlit trafilatura \
            transformers torch scikit-learn
```

Or, if you have **uv** installed (recommended — it is much faster):

```bash
uv sync
```

---

## ▶️ Running the App

```bash
streamlit run app.py --server.port 5000
```

Then open your browser and navigate to:

```
http://localhost:5000
```

> **First run warning:** The app downloads the DistilBERT model and runs sentiment analysis on all phones, then writes the results to `sentiment_cache.pkl`. This can take **2–5 minutes** depending on your hardware. All subsequent runs load from the cache and start in seconds.

---

## ⚙️ Configuration

Streamlit server settings live in `.streamlit/config.toml`:

```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

Change `port` here (and in the run command) if port 5000 is already in use on your machine.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit ≥ 1.44.1` | Web UI framework |
| `pandas ≥ 2.2.3` | Data loading and manipulation |
| `numpy ≥ 2.2.4` | Numerical operations |
| `plotly ≥ 6.0.1` | Interactive charts |
| `transformers ≥ 4.40.0` | HuggingFace DistilBERT pipeline |
| `torch ≥ 2.0.0` | PyTorch backend for Transformers |
| `scikit-learn ≥ 1.4.0` | TF-IDF vectoriser |
| `trafilatura ≥ 2.0.0` | Web content extraction utilities |

---

## 🗃️ Data

The app reads three CSV files from `attached_assets/`:

- `flipkart_phones.csv` — Flipkart listings
- `amazon_phones.csv` — Amazon listings
- `cashify_phones.csv` — Cashify (refurbished/resale) listings

Each file is expected to contain columns including: `Brand`, `Model`, `RAM`, `Storage`, `Color`, `Price`, `Original_Price`, `Discount_Percentage`, `Battery`, `Screen_Size`, `Main_Camera`, `Charging`, and a reviews/rating column used for sentiment analysis.

> If you want to refresh the data, replace the CSVs and **delete `sentiment_cache.pkl`** so that the sentiment pipeline re-runs on the new data.

---

## 🔄 Resetting the Sentiment Cache

The cache file (`sentiment_cache.pkl`) stores pre-computed sentiment scores so the app doesn't re-run the model on every restart. To force a full re-analysis (e.g., after updating the CSVs):

```bash
rm sentiment_cache.pkl       # macOS / Linux
del sentiment_cache.pkl      # Windows
```

Then restart the app.

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| Port 5000 already in use | Change `port` in `.streamlit/config.toml` and re-run |
| `ModuleNotFoundError` | Make sure your virtual environment is activated and all dependencies are installed |
| Slow first startup | Normal — DistilBERT is downloading and caching. Wait for the spinner to finish |
| `LookupError` on NLTK resources | Run `python -c "import nltk; nltk.download('all')"` in your venv |
| `InvalidComparison` errors | Ensure price columns in CSVs are numeric; check `data_processor.py` cleaning logic |
| App crashes on large CSV | Reduce batch size in `sentiment_analyzer.py` (`DISTILBERT_BATCH_SIZE = 32`) |

---

## 🚢 Deployment (Replit)

This project is pre-configured for Replit deployment. The `.replit` file defines:

```
run = ["streamlit", "run", "app.py", "--server.port", "5000"]
```

Simply import the project into Replit and hit **Run**. Replit maps internal port 5000 to external port 80 automatically.

---

## 📄 License

This project is for educational / portfolio purposes. See individual data source terms (Flipkart, Amazon, Cashify) regarding their data usage policies.
