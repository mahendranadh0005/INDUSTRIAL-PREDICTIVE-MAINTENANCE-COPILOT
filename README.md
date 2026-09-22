# Industrial Predictive Maintenance Copilot

> AI-powered failure prediction, anomaly detection, and maintenance knowledge assistant built on the AI4I 2020 Predictive Maintenance Dataset.

---

## Project Overview

This project is a **submission-ready academic/portfolio project** demonstrating a complete AI-powered predictive maintenance pipeline — from raw sensor data through machine learning, anomaly detection, Retrieval-Augmented Generation (RAG), and an interactive AI Maintenance Copilot.

> **Important:** All application source code is contained in a single file: `app.py`.

---

## Problem Statement

Industrial equipment failures cause unplanned downtime, safety hazards, and significant financial loss. Traditional maintenance approaches — reactive (fix when broken) or scheduled preventive maintenance — are either too late or wasteful. Predictive maintenance uses sensor data and machine learning to estimate failure probability **before** failure occurs, enabling targeted, timely maintenance actions.

---

## Objectives

1. Predict machine failure probability from sensor readings
2. Detect anomalous operating conditions without labelled failure data
3. Analyse failure mode distribution and sensor correlations
4. Provide model explainability via feature importance
5. Retrieve relevant maintenance documentation using RAG
6. Deliver an AI Maintenance Copilot that integrates all of the above

---

## Key Features

| Feature | Description |
|---------|-------------|
| Exploratory Data Analysis | Interactive sensor distributions, correlations, failure analysis |
| Binary Classification | Logistic Regression and Random Forest failure prediction |
| Class Imbalance Handling | `class_weight="balanced"` + stratified splitting |
| Model Explainability | Random Forest feature importance |
| Anomaly Detection | Isolation Forest for unsupervised anomaly detection |
| RAG Pipeline | Sentence Transformers + FAISS vector search on maintenance documents |
| AI Copilot | LLM-augmented (OpenAI, optional) maintenance Q&A |
| Interactive Dashboard | Multi-page Streamlit application with KPI cards and Plotly charts |

---

## Architecture

```
Dataset (AI4I 2020 CSV)
       ↓
Data Processing & Quality Checks
       ↓
Exploratory Data Analysis (Plotly)
       ↓
Machine Learning Pipeline (Logistic Regression + Random Forest)
       ↓
Failure Probability Estimation & Risk Scoring
       ↓
Anomaly Detection (Isolation Forest)
       ↓
RAG: Documents → Chunks → Embeddings → FAISS Index
       ↓
AI Maintenance Copilot (Retrieval + Optional LLM)
       ↓
Interactive Streamlit Dashboard
```

---

## Dataset Description

**AI4I 2020 Predictive Maintenance Dataset**
- Source: UCI Machine Learning Repository
- Records: 10,000 synthetic industrial machine observations
- Target: `Machine failure` (binary: 0 = Normal, 1 = Failure)

### Dataset Columns

| Column | Role | Type | Description | Unit |
|--------|------|------|-------------|------|
| UDI | ID | Integer | Unique row identifier | — |
| Product ID | ID | Categorical | Machine/product identifier | — |
| Type | Feature | Categorical | Machine type (L/M/H) | — |
| Air temperature [K] | Feature | Continuous | Ambient air temperature | K |
| Process temperature [K] | Feature | Continuous | Process temperature | K |
| Rotational speed [rpm] | Feature | Integer | Motor rotational speed | rpm |
| Torque [Nm] | Feature | Continuous | Torque applied | Nm |
| Tool wear [min] | Feature | Integer | Cumulative tool wear time | min |
| Machine failure | Target | Integer | Primary failure indicator (0/1) | — |
| TWF | Target | Integer | Tool Wear Failure indicator | — |
| HDF | Target | Integer | Heat Dissipation Failure indicator | — |
| PWF | Target | Integer | Power Failure indicator | — |
| OSF | Target | Integer | Overstrain Failure indicator | — |
| RNF | Target | Integer | Random Failure indicator | — |

---

## Data Preprocessing

- Column names normalised (handles spacing and unit bracket variations)
- Missing numeric values imputed with column median
- Duplicate rows removed
- `Type` encoded with One-Hot Encoding in the pipeline
- Numeric features standardised with `StandardScaler`
- No target leakage: `TWF`, `HDF`, `PWF`, `OSF`, `RNF`, `Machine failure` not used as input features

---

## Exploratory Data Analysis

The **Data Explorer** page provides:
- Dataset shape, dtypes, missing-value counts, duplicate counts
- Histogram distributions of all five numeric sensors, coloured by failure status
- Correlation heatmap
- Machine type distribution

The **Failure Analysis** page provides:
- Failure mode frequency bar chart
- Sensor box plots by failure status
- Torque vs Rotational speed scatter coloured by failure
- Tool wear histogram split by failure status

---

## Machine Learning Methodology

### Features
`Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`

### Target
`Machine failure` (binary)

### Models

| Model | Purpose |
|-------|---------|
| Logistic Regression | Baseline linear model |
| Random Forest (200 trees) | Main model — non-linear, feature importance |

### Pipeline
- `ColumnTransformer`: StandardScaler for numeric features, OneHotEncoder for `Type`
- `train_test_split`: 80/20, stratified, `random_state=42`
- `class_weight="balanced"`: adjusts loss weights inversely proportional to class frequency

---

## Model Comparison

Metrics are computed on the held-out test set (20% of data). Run the application to see actual values.

---

## Anomaly Detection

- **Algorithm:** Isolation Forest (200 estimators, `contamination=0.05`)
- **Input features:** all five numeric sensor columns
- **Output:** anomaly score and binary anomaly label per record
- **Note:** Anomalies represent unusual sensor value combinations, not guaranteed failures.

---

## RAG Architecture

```
Maintenance Documents (.txt / .md / .pdf)
       ↓
Text Extraction
       ↓
Word-based Chunking (400 words, 60-word overlap)
       ↓
Embeddings (sentence-transformers all-MiniLM-L6-v2)
       ↓
FAISS Vector Index (IndexFlatL2)
       ↓
Similarity Search (top-k=4)
       ↓
Retrieved Context
       ↓
LLM Prompt Construction / Fallback Answer
       ↓
AI Copilot Answer
```

---

## AI Copilot

The AI Maintenance Copilot combines:
1. **Dataset statistics** (failure rate, anomaly count)
2. **Machine-level context** (sensor values, failure probability, risk category)
3. **Retrieved maintenance documentation** (RAG)
4. **LLM generation** (OpenAI, optional)

If no LLM API key is configured, the copilot still returns retrieved documentation and structured model output.

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| UI / Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualisation | Plotly |
| Machine Learning | Scikit-learn |
| Anomaly Detection | Scikit-learn (Isolation Forest) |
| Embeddings | sentence-transformers |
| Vector Search | FAISS |
| PDF Parsing | PyPDF |
| LLM (optional) | OpenAI API |
| Config | python-dotenv |

---

## Project Structure

```
Industrial-Predictive-Maintenance-Copilot/
├── app.py                       ← All application source code
├── requirements.txt
├── README.md
├── FINAL_PROJECT_REPORT.docx
├── .env.example
├── dataset/
│   └── ai4i2020.csv
└── knowledge_base/
    ├── README.md
    ├── maintenance_manual.txt
    ├── failure_procedures.txt
    └── preventive_maintenance.txt
```

---

## Installation

```bash
# 1. Clone or download the project
cd Industrial-Predictive-Maintenance-Copilot

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Dataset Setup

Place the AI4I 2020 CSV at either:
- `ai4i2020.csv` (project root)  **or**
- `dataset/ai4i2020.csv`

The application will detect the file automatically.

Download the dataset from:
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

---

## Knowledge Base Setup

Place maintenance documents in `knowledge_base/`.

Supported formats: `.txt`, `.md`, `.pdf`

See `knowledge_base/README.md` for guidance.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your API key:

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=your_actual_api_key_here
```

The application works without an API key (retrieval-only mode).

---

## How to Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Example Copilot Questions

- "What does tool wear failure mean?"
- "What maintenance procedure should I follow for high tool wear?"
- "What are the common failure modes in industrial machines?"
- "What sensor conditions are associated with machine failure?"
- "What should I inspect when rotational speed becomes abnormal?"
- "How often should preventive maintenance be performed?"

---

## Limitations

- The dataset is **synthetic** (generated for research purposes), not real operational data.
- Risk thresholds (Low/Medium/High) are **application-defined** and not industrial safety standards.
- The model provides **estimated probabilities**, not guaranteed predictions.
- Anomaly detection uses a fixed contamination rate (5%) that may not suit all deployments.
- The LLM copilot depends on an external API and is optional.

---

## Future Enhancements

- Real-time sensor data ingestion via MQTT or REST API
- Time-series modelling (LSTM, temporal convolution)
- Remaining Useful Life (RUL) estimation
- Multi-class failure mode prediction
- SHAP explanations
- Scheduled retraining pipeline
- User authentication and audit logging

---

## Academic Disclaimer

This project is developed for academic and portfolio demonstration purposes.
It must not be used for real industrial safety decisions without professional engineering validation.
All model outputs are probabilistic estimates based on historical training data.
