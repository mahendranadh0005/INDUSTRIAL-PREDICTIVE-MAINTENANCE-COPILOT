"""
Industrial Predictive Maintenance Copilot
==========================================
Single-file Streamlit application.
All source code is contained in this file.

Run:
    streamlit run app.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import io
import re
import glob
import time
import warnings
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve,
)

from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Industrial Predictive Maintenance Copilot",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DATASET_PATHS = [
    "ai4i2020.csv",
    "dataset/ai4i2020.csv",
    "data/ai4i2020.csv",
]

KNOWLEDGE_BASE_DIR = Path("knowledge_base")

FEATURE_COLS = [
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]
NUMERIC_FEATURES = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]
TARGET_COL = "Machine failure"
FAILURE_MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

RISK_LOW_MAX    = 0.30
RISK_MEDIUM_MAX = 0.60

RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
COLUMN_ALIASES = {
    "udi": "UDI",
    "product id": "Product ID",
    "type": "Type",
    "air temperature": "Air temperature [K]",
    "air temperature [k]": "Air temperature [K]",
    "air temp": "Air temperature [K]",
    "process temperature": "Process temperature [K]",
    "process temperature [k]": "Process temperature [K]",
    "process temp": "Process temperature [K]",
    "rotational speed": "Rotational speed [rpm]",
    "rotational speed [rpm]": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "torque [nm]": "Torque [Nm]",
    "tool wear": "Tool wear [min]",
    "tool wear [min]": "Tool wear [min]",
    "machine failure": "Machine failure",
    "twf": "TWF",
    "hdf": "HDF",
    "pwf": "PWF",
    "osf": "OSF",
    "rnf": "RNF",
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names via lowercase alias lookup."""
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
    return df.rename(columns=rename_map)

REQUIRED_COLS = FEATURE_COLS + [TARGET_COL] + FAILURE_MODE_COLS

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file_source) -> pd.DataFrame:
    """Load and validate the AI4I dataset from a file path or uploaded bytes."""
    try:
        if isinstance(file_source, (str, Path)):
            raw = pd.read_csv(file_source)
        else:
            raw = pd.read_csv(file_source)
        df = normalize_columns(raw)
        missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing_required:
            st.error(f"Missing required columns: {missing_required}")
            return None
        return df
    except Exception as exc:
        st.error(f"Could not read dataset: {exc}")
        return None


def find_default_dataset() -> str | None:
    for p in DATASET_PATHS:
        if Path(p).exists():
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# DATA QUALITY
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def data_quality_report(df: pd.DataFrame) -> dict:
    report = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "failure_count": int(df[TARGET_COL].sum()),
        "failure_rate": float(df[TARGET_COL].mean()),
    }
    invalid = {}
    for col in NUMERIC_FEATURES:
        neg_count = int((df[col] < 0).sum())
        if neg_count:
            invalid[col] = neg_count
    report["invalid_negatives"] = invalid
    return report


@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing numerics with median; drop duplicate rows."""
    df = df.copy()
    for col in NUMERIC_FEATURES:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ML PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def build_preprocessor():
    num_pipeline = Pipeline([("scaler", StandardScaler())])
    cat_pipeline = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([
        ("num", num_pipeline, NUMERIC_FEATURES),
        ("cat", cat_pipeline, CATEGORICAL_FEATURES),
    ])


@st.cache_resource(show_spinner=False)
def train_models(df_clean: pd.DataFrame):
    """Train Logistic Regression and Random Forest; return all artefacts."""
    X = df_clean[FEATURE_COLS]
    y = df_clean[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()

    # ── Logistic Regression ──────────────────────────────────────────────────
    lr_pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        )),
    ])
    lr_pipe.fit(X_train, y_train)

    # ── Random Forest ────────────────────────────────────────────────────────
    rf_pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])
    rf_pipe.fit(X_train, y_train)

    # ── Feature importance ───────────────────────────────────────────────────
    ohe = rf_pipe.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + cat_names
    importances = rf_pipe.named_steps["clf"].feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    return dict(
        lr=lr_pipe,
        rf=rf_pipe,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names,
        feat_imp=feat_imp,
    )


def evaluate_model(pipe, X_test, y_test, label: str) -> dict:
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    return dict(
        label=label,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_test, y_prob),
        cm=confusion_matrix(y_test, y_pred),
        fpr=roc_curve(y_test, y_prob)[0],
        tpr=roc_curve(y_test, y_prob)[1],
        report=classification_report(y_test, y_pred, zero_division=0),
        y_pred=y_pred,
        y_prob=y_prob,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def detect_anomalies(df_clean: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    iso = IsolationForest(
        n_estimators=200, contamination=contamination,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    X_anom = df_clean[NUMERIC_FEATURES].values
    scores = iso.fit(X_anom).decision_function(X_anom)
    labels = iso.predict(X_anom)
    df_out = df_clean.copy()
    df_out["anomaly_score"] = scores
    df_out["is_anomaly"] = (labels == -1).astype(int)
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# RISK SCORING
# ─────────────────────────────────────────────────────────────────────────────
def risk_category(prob: float) -> tuple[str, str]:
    if prob < RISK_LOW_MAX:
        return "🟢 Low Risk", "green"
    elif prob < RISK_MEDIUM_MAX:
        return "🟡 Medium Risk", "orange"
    else:
        return "🔴 High Risk", "red"


# ─────────────────────────────────────────────────────────────────────────────
# RAG SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_rag_index():
    """
    Load documents from knowledge_base/, chunk, embed, build FAISS index.
    Returns (index, chunks, embedder) or (None, [], None) on failure.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError:
        return None, [], None, "sentence-transformers or faiss-cpu not installed."

    docs = []
    kb_path = KNOWLEDGE_BASE_DIR

    if not kb_path.exists():
        return None, [], None, "knowledge_base/ folder not found."

    # ── Load .txt and .md ────────────────────────────────────────────────────
    for ext in ("*.txt", "*.md"):
        for fpath in kb_path.glob(ext):
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
                docs.append((fpath.name, text))
            except Exception:
                pass

    # ── Load .pdf ────────────────────────────────────────────────────────────
    for fpath in kb_path.glob("*.pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(fpath))
            text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
            if text.strip():
                docs.append((fpath.name, text))
        except Exception:
            pass

    if not docs:
        return None, [], None, (
            "No documents found in knowledge_base/. "
            "Add .txt, .md, or .pdf files to enable RAG."
        )

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size = 400
    overlap    = 60
    chunks = []
    for fname, text in docs:
        words = text.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append({"source": fname, "text": chunk_text})
            if end == len(words):
                break
            start += chunk_size - overlap

    if not chunks:
        return None, [], None, "Documents found but no text could be extracted."

    # ── Embeddings ───────────────────────────────────────────────────────────
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        texts    = [c["text"] for c in chunks]
        vectors  = embedder.encode(texts, batch_size=32, show_progress_bar=False)
        vectors  = np.array(vectors, dtype="float32")

        import faiss
        dim   = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        return index, chunks, embedder, None
    except Exception as exc:
        return None, [], None, f"Embedding error: {exc}"


def rag_search(query: str, index, chunks, embedder, top_k: int = 4) -> list[dict]:
    """Return top-k relevant chunks for a query."""
    if index is None or not chunks or embedder is None:
        return []
    try:
        import faiss
        q_vec = embedder.encode([query], show_progress_bar=False).astype("float32")
        distances, indices = index.search(q_vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(chunks):
                results.append({
                    "source": chunks[idx]["source"],
                    "text":   chunks[idx]["text"],
                    "score":  float(dist),
                })
        return results
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# LLM GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def llm_generate(prompt: str) -> str | None:
    """
    Attempt to call OpenAI (or compatible) LLM.
    Returns None if API key not configured or call fails.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_api_key_here":
        return None
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"[LLM error: {exc}]"


def build_copilot_prompt(
    user_q: str,
    retrieved: list[dict],
    context_stats: dict,
) -> str:
    rag_section = ""
    if retrieved:
        rag_section = "\n\n--- RETRIEVED MAINTENANCE DOCUMENTATION ---\n"
        for i, r in enumerate(retrieved, 1):
            rag_section += f"\n[Doc {i} | Source: {r['source']}]\n{r['text'][:600]}\n"

    stats_section = "\n\n--- DATASET & MODEL CONTEXT ---\n" + "\n".join(
        f"{k}: {v}" for k, v in context_stats.items()
    )

    return (
        "You are an industrial maintenance AI assistant.\n"
        "Answer the user's question using the retrieved documentation and dataset context below.\n"
        "Clearly distinguish between MODEL OUTPUT and RETRIEVED DOCUMENTATION.\n"
        "Be concise, factual, and avoid fabrication.\n\n"
        f"User Question: {user_q}"
        + stats_section
        + rag_section
        + "\n\nAnswer:"
    )


def fallback_rag_answer(user_q: str, retrieved: list[dict], context_stats: dict) -> str:
    """Rule-based answer when no LLM is available."""
    q_lower = user_q.lower()

    answer_parts = ["**AI Maintenance Copilot (Retrieval Mode)**\n"]
    answer_parts.append("*(No LLM configured — showing retrieved information only)*\n")

    if context_stats:
        answer_parts.append("\n**📊 Dataset & Model Context**")
        for k, v in context_stats.items():
            answer_parts.append(f"- {k}: {v}")

    if retrieved:
        answer_parts.append("\n**📚 Retrieved Maintenance Documentation**")
        for i, r in enumerate(retrieved, 1):
            answer_parts.append(
                f"\n*Source: {r['source']}*\n> {r['text'][:500]}..."
            )
    else:
        answer_parts.append(
            "\nNo relevant documentation found. "
            "Add maintenance documents to `knowledge_base/` for richer answers."
        )

    return "\n".join(answer_parts)


# ─────────────────────────────────────────────────────────────────────────────
# STYLING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(label: str, value: str, color: str = "#3b82d4") -> str:
    return f"""
<div style="
    background:#f7f8fa;
    border:1px solid #e5e7eb;
    border-top:4px solid {color};
    border-radius:6px;
    padding:16px 20px;
    text-align:center;
    margin-bottom:8px;
">
    <div style="font-size:11px;color:#57606a;text-transform:uppercase;
                letter-spacing:1px;margin-bottom:6px;">{label}</div>
    <div style="font-size:26px;font-weight:700;color:#1f2328;">{value}</div>
</div>"""


def section_header(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"<p style='color:#57606a;margin-top:-10px;'>{subtitle}</p>",
                    unsafe_allow_html=True)
    st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EXECUTIVE OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
def page_overview(df: pd.DataFrame, df_anom: pd.DataFrame, models: dict):
    section_header(
        "⚙️ Industrial Predictive Maintenance Copilot",
        "AI-powered failure prediction, anomaly detection, and maintenance knowledge assistant."
    )

    qr = data_quality_report(df)
    failure_rate_pct = qr["failure_rate"] * 100
    total = df.shape[0]
    fail_count = qr["failure_count"]
    anom_count = int(df_anom["is_anomaly"].sum())

    # Predict high-risk count
    rf = models["rf"]
    probs = rf.predict_proba(df[FEATURE_COLS])[:, 1]
    high_risk = int((probs >= RISK_MEDIUM_MAX).sum())
    avg_tool_wear = df["Tool wear [min]"].mean()

    cols = st.columns(6)
    kpis = [
        ("Total Records",    f"{total:,}",           "#3b82d4"),
        ("Failure Count",    f"{fail_count:,}",       "#e74c3c"),
        ("Failure Rate",     f"{failure_rate_pct:.1f}%", "#e67e22"),
        ("High-Risk Count",  f"{high_risk:,}",        "#9b59b6"),
        ("Anomalies",        f"{anom_count:,}",       "#1abc9c"),
        ("Avg Tool Wear",    f"{avg_tool_wear:.0f} min", "#7c5cd8"),
    ]
    for col, (label, val, color) in zip(cols, kpis):
        col.markdown(kpi_card(label, val, color), unsafe_allow_html=True)

    st.markdown("")
    c1, c2 = st.columns(2)

    with c1:
        # Failure distribution
        counts = df[TARGET_COL].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        counts["Status"] = counts["Status"].map({0: "Normal", 1: "Failure"})
        fig = px.pie(
            counts, values="Count", names="Status",
            title="Machine Failure Distribution",
            color="Status",
            color_discrete_map={"Normal": "#3b82d4", "Failure": "#e74c3c"},
            hole=0.45,
        )
        fig.update_layout(height=320, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Failure rate by type
        rate_by_type = (
            df.groupby("Type")[TARGET_COL]
            .agg(["sum", "count"])
            .reset_index()
        )
        rate_by_type["rate"] = rate_by_type["sum"] / rate_by_type["count"] * 100
        fig2 = px.bar(
            rate_by_type, x="Type", y="rate",
            title="Failure Rate by Machine Type (%)",
            labels={"rate": "Failure Rate (%)", "Type": "Machine Type"},
            color="Type",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig2.update_layout(height=320, margin=dict(t=40, b=0, l=0, r=0),
                           showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Sensor overview
    st.subheader("Sensor Overview (All Records)")
    sensor_cols = [
        "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    ]
    desc = df[sensor_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    desc.columns = ["Mean", "Std Dev", "Min", "Median", "Max"]
    st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
def page_data_explorer(df: pd.DataFrame):
    section_header(
        "🔍 Data Explorer",
        "Dataset overview, quality report, and raw data inspection."
    )

    qr = data_quality_report(df)

    # Quality Report
    with st.expander("📋 Data Quality Report", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{qr['shape'][0]:,}")
        c2.metric("Columns", f"{qr['shape'][1]:,}")
        c3.metric("Duplicate Rows", f"{qr['duplicates']:,}")

        miss_df = pd.DataFrame.from_dict(
            qr["missing"], orient="index", columns=["Missing Values"]
        )
        miss_df = miss_df[miss_df["Missing Values"] > 0]
        if miss_df.empty:
            st.success("✅ No missing values detected.")
        else:
            st.warning("⚠️ Missing values found:")
            st.dataframe(miss_df, use_container_width=True)

        if qr["invalid_negatives"]:
            st.warning(f"⚠️ Negative values found: {qr['invalid_negatives']}")
        else:
            st.success("✅ No invalid negative sensor values.")

        st.markdown("**Column Data Types**")
        dtype_df = pd.DataFrame.from_dict(
            qr["dtypes"], orient="index", columns=["dtype"]
        )
        st.dataframe(dtype_df, use_container_width=True)

    # Distribution plots
    st.subheader("Feature Distributions")
    tab1, tab2 = st.tabs(["Numeric Features", "Categorical Features"])

    with tab1:
        nc = st.columns(3)
        for i, col in enumerate(NUMERIC_FEATURES):
            with nc[i % 3]:
                fig = px.histogram(
                    df, x=col, color=TARGET_COL,
                    barmode="overlay",
                    title=col,
                    labels={TARGET_COL: "Failure"},
                    color_discrete_map={0: "#3b82d4", 1: "#e74c3c"},
                    nbins=50,
                )
                fig.update_layout(height=280, margin=dict(t=40, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        vc = df["Type"].value_counts().reset_index()
        vc.columns = ["Type", "Count"]
        fig_t = px.bar(
            vc, x="Type", y="Count",
            title="Machine Type Distribution",
            color="Type",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig_t, use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr_cols = NUMERIC_FEATURES + [TARGET_COL]
    corr = df[corr_cols].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f",
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    )
    fig_corr.update_layout(height=450)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Raw data table
    st.subheader("Raw Data Sample")
    n_rows = st.slider("Rows to display", 10, 200, 50)
    st.dataframe(df.head(n_rows), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: FAILURE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def page_failure_analysis(df: pd.DataFrame):
    section_header(
        "🚨 Failure Analysis",
        "Descriptive analysis of failure modes and sensor conditions associated with machine failures."
    )

    st.info(
        "**Note:** TWF, HDF, PWF, OSF and RNF are failure-mode indicators. "
        "They are used here for *descriptive* analysis only and are **not** "
        "used as features in the primary failure prediction model (to avoid target leakage)."
    )

    qr = data_quality_report(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Failure Rate", f"{qr['failure_rate']*100:.2f}%")
    c2.metric("Total Failures", f"{qr['failure_count']:,}")
    c3.metric("Total Records",  f"{qr['shape'][0]:,}")

    # Failure modes
    st.subheader("Failure Mode Frequency")
    mode_counts = df[FAILURE_MODE_COLS].sum().reset_index()
    mode_counts.columns = ["Failure Mode", "Count"]
    mode_labels = {
        "TWF": "Tool Wear Failure",
        "HDF": "Heat Dissipation Failure",
        "PWF": "Power Failure",
        "OSF": "Overstrain Failure",
        "RNF": "Random Failure",
    }
    mode_counts["Label"] = mode_counts["Failure Mode"].map(mode_labels)
    fig_modes = px.bar(
        mode_counts, x="Label", y="Count",
        title="Failure Mode Frequency",
        labels={"Label": "Failure Mode", "Count": "Occurrences"},
        color="Label",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_modes.update_layout(showlegend=False)
    st.plotly_chart(fig_modes, use_container_width=True)

    # Sensor vs failure box plots
    st.subheader("Sensor Conditions vs Failure Status")
    sensor_choice = st.selectbox(
        "Select sensor", NUMERIC_FEATURES,
        key="fa_sensor_select"
    )
    df_plot = df.copy()
    df_plot["Failure Status"] = df_plot[TARGET_COL].map({0: "Normal", 1: "Failure"})
    fig_box = px.box(
        df_plot, x="Failure Status", y=sensor_choice,
        title=f"{sensor_choice} vs Failure Status",
        color="Failure Status",
        color_discrete_map={"Normal": "#3b82d4", "Failure": "#e74c3c"},
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Torque vs Rotational Speed scatter
    st.subheader("Torque vs Rotational Speed")
    fig_sc = px.scatter(
        df_plot, x="Rotational speed [rpm]", y="Torque [Nm]",
        color="Failure Status",
        color_discrete_map={"Normal": "#3b82d4", "Failure": "#e74c3c"},
        title="Torque vs Rotational Speed by Failure Status",
        opacity=0.6, size_max=5,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # Failure rate by type
    st.subheader("Failure Rate by Machine Type")
    rate_by_type = (
        df.groupby("Type")[TARGET_COL]
        .agg(total="count", failures="sum")
        .reset_index()
    )
    rate_by_type["Failure Rate (%)"] = rate_by_type["failures"] / rate_by_type["total"] * 100
    st.dataframe(rate_by_type, use_container_width=True)

    # Tool wear vs failure
    st.subheader("Tool Wear Distribution by Failure Status")
    fig_tw = px.histogram(
        df_plot, x="Tool wear [min]", color="Failure Status",
        barmode="overlay",
        title="Tool Wear Distribution — Normal vs Failure",
        color_discrete_map={"Normal": "#3b82d4", "Failure": "#e74c3c"},
        nbins=60,
    )
    st.plotly_chart(fig_tw, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICTIVE MAINTENANCE
# ─────────────────────────────────────────────────────────────────────────────
def page_predictive_maintenance(df: pd.DataFrame, models: dict):
    section_header(
        "🔮 Predictive Maintenance",
        "Estimate failure probability for individual machines using the trained Random Forest model."
    )

    st.info(
        f"**Risk thresholds (application-level, not industrial safety standards):**  "
        f"Low Risk < {RISK_LOW_MAX*100:.0f}% | "
        f"Medium Risk {RISK_LOW_MAX*100:.0f}–{RISK_MEDIUM_MAX*100:.0f}% | "
        f"High Risk ≥ {RISK_MEDIUM_MAX*100:.0f}%"
    )

    tab1, tab2 = st.tabs(["📋 Select from Dataset", "✏️ Manual Input"])

    with tab1:
        idx = st.number_input(
            "Record index (0-based)",
            min_value=0, max_value=len(df) - 1,
            value=0, step=1,
        )
        row = df.iloc[idx]
        st.markdown("**Selected Machine Parameters**")
        param_df = pd.DataFrame(
            {
                "Parameter": FEATURE_COLS,
                "Value": [row[c] for c in FEATURE_COLS],
            }
        )
        st.table(param_df)

        if st.button("Predict Failure Probability", key="pred_from_dataset"):
            X_in = pd.DataFrame([row[FEATURE_COLS]])
            prob = models["rf"].predict_proba(X_in)[0][1]
            cat, color_name = risk_category(prob)
            col_l, col_r = st.columns(2)
            col_l.metric("Estimated Failure Probability", f"{prob*100:.1f}%")
            col_r.markdown(
                kpi_card("Risk Category", cat,
                         "#e74c3c" if "High" in cat else
                         "#e67e22" if "Medium" in cat else "#27ae60"),
                unsafe_allow_html=True,
            )
            if df.loc[idx, TARGET_COL] == 1:
                st.warning("⚠️ Actual label in dataset: **Machine Failure**")
            else:
                st.success("✅ Actual label in dataset: **Normal**")

    with tab2:
        st.markdown("**Enter sensor values manually:**")
        c1, c2, c3 = st.columns(3)
        with c1:
            m_type = st.selectbox("Machine Type", sorted(df["Type"].unique()))
            air_temp = st.number_input(
                "Air Temperature [K]",
                min_value=280.0, max_value=330.0,
                value=float(df["Air temperature [K]"].median()),
                step=0.1
            )
        with c2:
            proc_temp = st.number_input(
                "Process Temperature [K]",
                min_value=290.0, max_value=340.0,
                value=float(df["Process temperature [K]"].median()),
                step=0.1
            )
            rot_speed = st.number_input(
                "Rotational Speed [rpm]",
                min_value=1000, max_value=3000,
                value=int(df["Rotational speed [rpm]"].median()),
                step=1
            )
        with c3:
            torque = st.number_input(
                "Torque [Nm]",
                min_value=0.0, max_value=100.0,
                value=float(df["Torque [Nm]"].median()),
                step=0.1
            )
            tool_wear = st.number_input(
                "Tool Wear [min]",
                min_value=0, max_value=300,
                value=int(df["Tool wear [min]"].median()),
                step=1
            )

        if st.button("Predict Failure Probability", key="pred_manual"):
            input_row = pd.DataFrame([{
                "Type": m_type,
                "Air temperature [K]": air_temp,
                "Process temperature [K]": proc_temp,
                "Rotational speed [rpm]": rot_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
            }])
            prob = models["rf"].predict_proba(input_row)[0][1]
            cat, _ = risk_category(prob)
            col_l, col_r = st.columns(2)
            col_l.metric("Estimated Failure Probability", f"{prob*100:.1f}%")
            col_r.markdown(
                kpi_card("Risk Category", cat,
                         "#e74c3c" if "High" in cat else
                         "#e67e22" if "Medium" in cat else "#27ae60"),
                unsafe_allow_html=True,
            )

    # Feature importance
    st.subheader("🔍 Feature Importance (Random Forest)")
    st.caption(
        "The chart below shows which features are most *associated with the model's predictions*. "
        "This does not imply causal relationships."
    )
    feat_imp = models["feat_imp"]
    fig_fi = px.bar(
        x=feat_imp.values, y=feat_imp.index,
        orientation="h",
        title="Feature Importance — Random Forest",
        labels={"x": "Importance", "y": "Feature"},
        color=feat_imp.values,
        color_continuous_scale="Blues",
    )
    fig_fi.update_layout(yaxis={"autorange": "reversed"}, coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def page_anomaly_detection(df_anom: pd.DataFrame):
    section_header(
        "🛸 Anomaly Detection",
        "Unsupervised detection of unusual operating conditions using Isolation Forest."
    )

    st.warning(
        "**Important:** An anomaly here indicates *unusual sensor readings* compared to "
        "the overall dataset distribution. An anomaly does **not** automatically indicate "
        "a machine failure. These are distinct concepts."
    )

    n_anom = int(df_anom["is_anomaly"].sum())
    n_total = len(df_anom)
    pct = n_anom / n_total * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records",    f"{n_total:,}")
    c2.metric("Anomalies Detected", f"{n_anom:,}")
    c3.metric("Anomaly Percentage", f"{pct:.1f}%")

    st.subheader("Anomaly Score Distribution")
    fig_score = px.histogram(
        df_anom, x="anomaly_score",
        color="is_anomaly",
        barmode="overlay",
        title="Isolation Forest Anomaly Scores",
        labels={"anomaly_score": "Anomaly Score", "is_anomaly": "Is Anomaly"},
        color_discrete_map={0: "#3b82d4", 1: "#e74c3c"},
        nbins=60,
    )
    st.plotly_chart(fig_score, use_container_width=True)

    st.subheader("Anomalies in Sensor Space")
    x_axis = st.selectbox("X axis", NUMERIC_FEATURES, index=2, key="anom_x")
    y_axis = st.selectbox("Y axis", NUMERIC_FEATURES, index=3, key="anom_y")

    df_plot = df_anom.copy()
    df_plot["Status"] = df_plot["is_anomaly"].map({0: "Normal", 1: "Anomaly"})
    fig_scatter = px.scatter(
        df_plot, x=x_axis, y=y_axis,
        color="Status",
        color_discrete_map={"Normal": "#3b82d4", "Anomaly": "#e74c3c"},
        title=f"Anomaly Detection: {x_axis} vs {y_axis}",
        opacity=0.6,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Anomaly vs actual failure overlap
    st.subheader("Anomaly Detection vs Actual Machine Failure")
    cross = pd.crosstab(
        df_anom["is_anomaly"], df_anom[TARGET_COL],
        rownames=["Is Anomaly"], colnames=["Machine Failure"],
    )
    st.dataframe(cross, use_container_width=True)
    st.caption(
        "Overlap between detected anomalies and actual machine failures reflects "
        "the complementary (not equivalent) nature of these two signals."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
def page_model_performance(models: dict):
    section_header(
        "📊 Model Performance",
        "Evaluation metrics computed on the held-out test set (20% of data, stratified split)."
    )

    st.info(
        "**Class imbalance handling:** Both models use `class_weight='balanced'` "
        "which adjusts sample weights inversely proportional to class frequency. "
        "Stratified splitting ensures the test set reflects the same failure rate as training data."
    )

    lr_eval = evaluate_model(
        models["lr"], models["X_test"], models["y_test"], "Logistic Regression"
    )
    rf_eval = evaluate_model(
        models["rf"], models["X_test"], models["y_test"], "Random Forest"
    )

    # Comparison table
    st.subheader("Model Comparison")
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    comp_df = pd.DataFrame({
        "Metric": metric_labels,
        "Logistic Regression": [f"{lr_eval[k]:.4f}" for k in metric_keys],
        "Random Forest":       [f"{rf_eval[k]:.4f}" for k in metric_keys],
    })
    st.table(comp_df)

    tab1, tab2 = st.tabs(["Logistic Regression", "Random Forest"])

    for tab, ev in [(tab1, lr_eval), (tab2, rf_eval)]:
        with tab:
            c1, c2, c3, c4, c5 = st.columns(5)
            for col, key, label in zip(
                [c1, c2, c3, c4, c5],
                metric_keys, metric_labels
            ):
                col.metric(label, f"{ev[key]:.4f}")

            col_cm, col_roc = st.columns(2)

            with col_cm:
                cm = ev["cm"]
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    title=f"Confusion Matrix — {ev['label']}",
                    labels=dict(x="Predicted", y="Actual"),
                    x=["Normal", "Failure"],
                    y=["Normal", "Failure"],
                    color_continuous_scale="Blues",
                )
                fig_cm.update_layout(height=350)
                st.plotly_chart(fig_cm, use_container_width=True)

            with col_roc:
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=ev["fpr"], y=ev["tpr"],
                    name=f"AUC = {ev['roc_auc']:.4f}",
                    line=dict(color="#3b82d4"),
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    line=dict(dash="dash", color="grey"),
                    showlegend=False,
                ))
                fig_roc.update_layout(
                    title=f"ROC Curve — {ev['label']}",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=350,
                )
                st.plotly_chart(fig_roc, use_container_width=True)

            with st.expander("Classification Report"):
                st.text(ev["report"])


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: AI MAINTENANCE COPILOT
# ─────────────────────────────────────────────────────────────────────────────
def page_copilot(df: pd.DataFrame, df_anom: pd.DataFrame, models: dict):
    section_header(
        "🤖 AI Maintenance Copilot",
        "Ask questions about machine health, failure risk, and maintenance procedures."
    )

    # Build RAG index once
    rag_index, rag_chunks, rag_embedder, rag_error = build_rag_index()

    if rag_error:
        st.warning(f"RAG status: {rag_error}")
    else:
        st.success(f"✅ RAG index ready — {len(rag_chunks):,} chunks from knowledge base.")

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_api_key_here":
        st.info(
            "💡 **LLM not configured.** Set `OPENAI_API_KEY` in your `.env` file for full "
            "AI-generated answers. Currently showing retrieved information only."
        )

    # Example questions
    with st.expander("💬 Example Questions"):
        examples = [
            "What does tool wear failure mean?",
            "What maintenance procedure should I follow for high tool wear?",
            "What are the common failure modes in industrial machines?",
            "What sensor conditions are associated with machine failure?",
            "What should I inspect when rotational speed becomes abnormal?",
            "What is Heat Dissipation Failure?",
            "How often should preventive maintenance be performed?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:30]}"):
                st.session_state["copilot_query"] = ex

    # Chat input
    user_query = st.text_input(
        "Ask the Maintenance Copilot:",
        value=st.session_state.get("copilot_query", ""),
        placeholder="e.g. What are the common failure modes?",
        key="copilot_input",
    )

    # Optional: associate with a specific machine record
    with st.expander("🔗 Associate with a specific machine record (optional)"):
        use_machine = st.checkbox("Include machine context in query")
        machine_idx = st.number_input(
            "Machine record index", 0, len(df) - 1, 0
        )

    if st.button("Ask Copilot", type="primary") and user_query.strip():
        st.session_state["copilot_query"] = ""
        with st.spinner("Searching knowledge base and building response..."):
            # Build context stats
            context_stats = {
                "Total records":      df.shape[0],
                "Failure rate":       f"{df[TARGET_COL].mean()*100:.2f}%",
                "Anomalies detected": int(df_anom["is_anomaly"].sum()),
            }

            if use_machine:
                row = df.iloc[machine_idx]
                prob = models["rf"].predict_proba(
                    pd.DataFrame([row[FEATURE_COLS]])
                )[0][1]
                cat, _ = risk_category(prob)
                is_anom = int(df_anom.iloc[machine_idx]["is_anomaly"])
                context_stats.update({
                    "Machine index":        machine_idx,
                    "Machine type":         row["Type"],
                    "Air temperature [K]":  f"{row['Air temperature [K]']:.1f}",
                    "Process temp [K]":     f"{row['Process temperature [K]']:.1f}",
                    "Rotational speed":     f"{row['Rotational speed [rpm]']} rpm",
                    "Torque":               f"{row['Torque [Nm]']:.1f} Nm",
                    "Tool wear":            f"{row['Tool wear [min]']} min",
                    "Failure probability":  f"{prob*100:.1f}%",
                    "Risk category":        cat,
                    "Anomaly status":       "Anomaly" if is_anom else "Normal",
                    "Actual failure label": str(int(row[TARGET_COL])),
                })

            # RAG retrieval
            retrieved = rag_search(
                user_query, rag_index, rag_chunks, rag_embedder, top_k=4
            )

            # Generate answer
            prompt = build_copilot_prompt(user_query, retrieved, context_stats)
            llm_response = llm_generate(prompt)

            # Display
            st.markdown("---")
            st.markdown(f"**Question:** {user_query}")
            st.markdown("---")

            if llm_response:
                st.markdown("**🤖 AI-Generated Answer (MODEL OUTPUT + LLM):**")
                st.markdown(llm_response)
            else:
                answer = fallback_rag_answer(user_query, retrieved, context_stats)
                st.markdown(answer)

            if retrieved:
                with st.expander("📚 Retrieved Documentation Chunks"):
                    for i, r in enumerate(retrieved, 1):
                        st.markdown(f"**Chunk {i} — Source: `{r['source']}`** "
                                    f"(distance: {r['score']:.3f})")
                        st.text(r["text"][:600])
                        st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
def page_about():
    section_header(
        "ℹ️ About This Project",
        "Project overview, architecture, and academic information."
    )

    st.markdown("""
## Industrial Predictive Maintenance Copilot

This is a submission-ready academic/portfolio project demonstrating the full pipeline of
an AI-powered industrial predictive maintenance system.

### What This System Does

| Component | Technology |
|-----------|-----------|
| Data Analytics & EDA | Pandas, Plotly |
| Machine Learning | Scikit-learn (Logistic Regression, Random Forest) |
| Anomaly Detection | Isolation Forest |
| Model Explainability | Feature Importance |
| RAG | Sentence Transformers + FAISS |
| AI Copilot | OpenAI API (optional) + Retrieval fallback |
| Dashboard | Streamlit |

### System Architecture

```
Dataset
   ↓
Data Processing & Quality Checks
   ↓
Exploratory Data Analysis
   ↓
Machine Learning (LR + RF)
   ↓
Failure Probability & Risk Scoring
   ↓
Anomaly Detection (Isolation Forest)
   ↓
RAG (Documents → Chunks → Embeddings → FAISS)
   ↓
AI Maintenance Copilot
   ↓
Interactive Streamlit Dashboard
```

### Dataset
**AI4I 2020 Predictive Maintenance Dataset** — 10,000 synthetic industrial machine records
with sensor readings and failure mode labels.

### Important Disclaimers

- **Estimated probabilities** are model predictions, not guaranteed outcomes.
- Risk thresholds (Low/Medium/High) are **application-level** and not industrial safety standards.
- Anomaly detection identifies unusual sensor patterns — not necessarily failures.
- This system is built for academic demonstration purposes only.
- Do not use this system for real industrial safety decisions without professional validation.

### Project Structure

```
app.py                          ← All application source code
requirements.txt
README.md
FINAL_PROJECT_REPORT.docx
.env.example
dataset/
    ai4i2020.csv
knowledge_base/
    README.md
    *.txt / *.md / *.pdf        ← Maintenance documents for RAG
```

### Running the Application

```bash
pip install -r requirements.txt
streamlit run app.py
```
""")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
            width=80,
        )
        st.title("Predictive Maintenance Copilot")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            [
                "⚙️ Executive Overview",
                "🔍 Data Explorer",
                "🚨 Failure Analysis",
                "🔮 Predictive Maintenance",
                "🛸 Anomaly Detection",
                "📊 Model Performance",
                "🤖 AI Maintenance Copilot",
                "ℹ️ About",
            ],
        )
        st.markdown("---")

        # Dataset loading
        st.subheader("Dataset")
        default_path = find_default_dataset()
        uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])

        if uploaded:
            file_source = uploaded
        elif default_path:
            file_source = default_path
            st.success(f"Using: `{default_path}`")
        else:
            file_source = None

        if file_source is None:
            st.error("Dataset not found. Upload the AI4I CSV to continue.")
            st.stop()

        # Training button
        st.subheader("Models")
        retrain = st.button("🔄 Train / Refresh Models")

        st.markdown("---")
        st.caption("All source code in `app.py`")

    # ── Data loading ──────────────────────────────────────────────────────────
    if retrain:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    df_raw = load_data(file_source)
    if df_raw is None:
        st.error("Could not load dataset. Check the file format.")
        st.stop()

    df = clean_data(df_raw)

    # ── Model training ────────────────────────────────────────────────────────
    with st.spinner("Training models (cached after first run)…"):
        models = train_models(df)

    # ── Anomaly detection ─────────────────────────────────────────────────────
    df_anom = detect_anomalies(df)

    # ── Page routing ──────────────────────────────────────────────────────────
    if "Executive Overview" in page:
        page_overview(df, df_anom, models)
    elif "Data Explorer" in page:
        page_data_explorer(df)
    elif "Failure Analysis" in page:
        page_failure_analysis(df)
    elif "Predictive Maintenance" in page:
        page_predictive_maintenance(df, models)
    elif "Anomaly Detection" in page:
        page_anomaly_detection(df_anom)
    elif "Model Performance" in page:
        page_model_performance(models)
    elif "AI Maintenance Copilot" in page:
        page_copilot(df, df_anom, models)
    elif "About" in page:
        page_about()


if __name__ == "__main__":
    main()
