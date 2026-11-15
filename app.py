import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import joblib
except Exception:
    joblib = None

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

st.set_page_config(page_title="Fake Review Detection Demo", page_icon="🕵️", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
MODELS_DIR = PROJECT_ROOT / "models"

def load_json(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def metric_file(name: str):
    """Saved metrics for one model, or None if that model has not been evaluated yet."""
    return load_json(RESULTS_DIR / name)


def pretty_name(key: str) -> str:
    return {
        "stylometric": "Stylometric Baseline",
        "bert": "BERT",
        "hybrid": "Hybrid Model",
    }.get(key, key.title())


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def try_predict_stylometric(text: str):
    model_path = MODELS_DIR / "stylometric" / "logreg.joblib"
    if not model_path.exists() or joblib is None:
        return None
    try:
        bundle = joblib.load(model_path)
        if hasattr(bundle, 'predict_proba'):
            # plain sklearn pipeline case
            prob_fake = float(bundle.predict_proba([text])[0][1])
            pred = int(prob_fake >= 0.5)
            return {"label": pred, "prob_fake": prob_fake}
        extractor = bundle.get("extractor")
        scaler = bundle.get("scaler")
        model = bundle.get("model")
        if extractor is None or scaler is None or model is None:
            return None
        X = extractor.transform([text])
        X_scaled = scaler.transform(X)
        prob_fake = float(model.predict_proba(X_scaled)[0][1])
        pred = int(prob_fake >= 0.5)
        return {"label": pred, "prob_fake": prob_fake}
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_bert():
    model_dir = MODELS_DIR / "bert"
    if not model_dir.exists() or AutoTokenizer is None or AutoModelForSequenceClassification is None:
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()
        return tokenizer, model
    except Exception:
        return None, None


def try_predict_bert(text: str):
    if torch is None:
        return None
    tokenizer, model = load_bert()
    if tokenizer is None or model is None:
        return None
    try:
        inputs = tokenizer([text], truncation=True, padding=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.detach().cpu().numpy()
        probs = softmax(logits)
        prob_fake = float(probs[0][1])
        pred = int(np.argmax(logits, axis=1)[0])
        return {"label": pred, "prob_fake": prob_fake}
    except Exception:
        return None


def render_card(title: str, result):
    st.subheader(title)
    if result is None:
        st.info("Prediction not available from saved local artifacts.")
        return
    label = "Fake / Deceptive Review" if result["label"] == 1 else "Truthful / Genuine Review"
    confidence = result["prob_fake"] if result["label"] == 1 else 1 - result["prob_fake"]
    st.metric("Prediction", label)
    st.progress(min(max(result["prob_fake"], 0.0), 1.0), text=f"Fake probability: {result['prob_fake']:.2%}")
    st.caption(f"Confidence in chosen class: {confidence:.2%}")


st.title("🕵️ Fake Review Detection System")
st.write(
    "This interface is designed for dissertation demonstration. It presents the project aim, "
    "a live classification demo, saved model metrics, and exported evidence plots."
)

k1, k2, k3 = st.columns(3)
k1.metric("Dataset", "1,600 reviews")
k2.metric("Classes", "Truthful vs Deceptive")
k3.metric("Best F1", "0.8974")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Live Demo", "Model Results", "Plots"])

with tab1:
    st.header("Project Aim")
    st.write(
        "Develop and evaluate a machine learning system for fake online review detection by comparing "
        "a stylometric model, a BERT-based model, and a hybrid model that combines both."
    )
    a, b, c = st.columns(3)
    a.info("""
Dataset:
- Contains fake and genuine reviews
- Used for classification task
- Source: OTT Deceptive Opinion Spam Dataset
- Size: 1600 reviews
""")
    b.info("""
Task:
- Binary classification
- Truthful vs deceptive
""")
    
    c.info("""
    Explainability:
    - SHAP-based interpretation
    """)
    
    st.subheader("Pipeline")
    st.markdown(
        """
1. Load and preprocess review text  
2. Create train, validation, and test splits  
3. Extract stylometric features  
4. Fine-tune BERT  
5. Build hybrid model using BERT + stylometric features  
6. Compare performance and significance  
"""
    )

with tab2:
    st.header("Live Review Classification")
    demo_text = (
        "The hotel was amazing. The staff were helpful, the room was clean, "
        "and the location was perfect for our weekend stay."
    )
    user_text = st.text_area("Enter a review", value=demo_text, height=180)
    if st.button("Run Prediction"):
        left, right = st.columns(2)
        with left:
            render_card("Stylometric Baseline", try_predict_stylometric(user_text))
        with right:
            render_card("BERT", try_predict_bert(user_text))
        st.caption("The hybrid model is shown in the saved evaluation results below.")

with tab3:
    st.header("Saved Model Performance")
    metrics = {
        "stylometric": metric_file("stylometric_metrics.json"),
        "bert": metric_file("bert_metrics.json"),
        "hybrid": metric_file("hybrid_metrics.json"),
    }
    rows = []
    for key, value in metrics.items():
        if value:
            rows.append({
                "Model": pretty_name(key),
                "Accuracy": round(value.get("accuracy", np.nan), 4),
                "Precision": round(value.get("precision", np.nan), 4),
                "Recall": round(value.get("recall", np.nan), 4),
                "F1": round(value.get("f1", np.nan), 4),
                "PR-AUC": round(value.get("pr_auc", np.nan), 4),
            })
    if not rows:
        st.warning("No saved metrics found in results/metrics. Run the training scripts first.")
    else:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        best = df.sort_values("F1", ascending=False).iloc[0]
        st.success(f"Highest F1 on the test split: {best['Model']} (F1 = {best['F1']:.4f}). See docs/RESULTS.md before reading this as a ranking.")

with tab4:
    st.header("Exported Evidence")
    plot_files = [
        ("Stylometric Confusion Matrix", PLOTS_DIR / "stylometric_confusion_matrix.png"),
        ("Stylometric PR Curve", PLOTS_DIR / "stylometric_pr_curve.png"),
        ("BERT Confusion Matrix", PLOTS_DIR / "bert_confusion_matrix.png"),
        ("BERT PR Curve", PLOTS_DIR / "bert_pr_curve.png"),
        ("Hybrid Confusion Matrix", PLOTS_DIR / "hybrid_confusion_matrix.png"),
        ("Hybrid PR Curve", PLOTS_DIR / "hybrid_pr_curve.png"),
    ]
    found = False
    for title, path in plot_files:
        if path.exists():
            found = True
            st.subheader(title)
            st.image(str(path), use_container_width=True)
    if not found:
        st.info("No plot files were found yet. Run the training scripts first to generate them.")

st.caption("Fake Review Detection Demo UI")
