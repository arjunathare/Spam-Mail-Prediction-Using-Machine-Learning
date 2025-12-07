import re
import html
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Optional language detection
try:
    from langdetect import detect  # type: ignore
except Exception:
    detect = None  # type: ignore


# ----------------------------
# Text preprocessing utilities
# ----------------------------

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
_PHONE_RE = re.compile(r"(\+?\d[\d\-\s\(\)]{8,}\d)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")

def clean_text(text: str) -> str:
    """Lowercase + strip HTML + replace URLs/emails/phones with tokens."""
    if text is None:
        return ""

    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)

    text = _URL_RE.sub(" <URL> ", text)
    text = _EMAIL_RE.sub(" <EMAIL> ", text)
    text = _PHONE_RE.sub(" <PHONE> ", text)

    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_subject_body(subject: str, body: str) -> str:
    subject = subject or ""
    body = body or ""
    combined = f"{subject}\n\n{body}".strip()
    return clean_text(combined)


def detect_lang(text: str) -> Optional[str]:
    """Returns ISO code like 'en', 'hi', 'mr' if langdetect is installed."""
    if detect is None:
        return None
    try:
        if len(text.split()) < 5:
            return None
        return detect(text)
    except Exception:
        return None


# ----------------------------
# Model loading + inference
# ----------------------------

ARTIFACT_PATH_DEFAULT = "spam_model.joblib"

@st.cache_resource
def load_artifacts(path: str) -> Dict[str, Any]:
    return joblib.load(path)

def predict_spam(
    artifacts: Dict[str, Any],
    subject: str,
    body: str,
    threshold: float,
) -> Dict[str, Any]:
    text = combine_subject_body(subject, body)
    lang = detect_lang(text)

    best_model = artifacts.get("best_model")
    fallback_model = artifacts.get("fallback_model")
    multilingual_model = artifacts.get("multilingual_model", None)
    label_map = artifacts.get("label_map", {"ham": 0, "spam": 1})

    # Route to multilingual pipeline for hi/mr if available
    use_multilingual = (lang in {"hi", "mr"}) and (multilingual_model is not None)
    model = multilingual_model if use_multilingual else best_model

    if model is None:
        raise RuntimeError("Model not found in artifacts. Train first with train.py")

    p_spam: Optional[float] = None
    used_proba = False

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        classes = getattr(model, "classes_", np.array([0, 1]))
        class_to_proba = {int(classes[i]): float(proba[i]) for i in range(len(classes))}
        p_spam = class_to_proba.get(label_map["spam"])
        used_proba = True
    elif hasattr(fallback_model, "predict_proba"):
        # fallback to LR probability if best model lacks probabilities
        proba = fallback_model.predict_proba([text])[0]
        classes = getattr(fallback_model, "classes_", np.array([0, 1]))
        class_to_proba = {int(classes[i]): float(proba[i]) for i in range(len(classes))}
        p_spam = class_to_proba.get(label_map["spam"])
        model = fallback_model
        used_proba = True

    if p_spam is None:
        pred = int(model.predict([text])[0])
        return {
            "text": text,
            "lang": lang,
            "used_multilingual": use_multilingual,
            "pred_label": pred,
            "p_spam": None,
            "threshold": threshold,
            "used_proba": False,
        }

    pred_label = 1 if p_spam >= threshold else 0
    return {
        "text": text,
        "lang": lang,
        "used_multilingual": use_multilingual,
        "pred_label": pred_label,
        "p_spam": float(p_spam),
        "threshold": threshold,
        "used_proba": used_proba,
        "used_model": type(model).__name__,
    }


# ----------------------------
# Explainability (LR-based)
# ----------------------------

def top_token_contributions(artifacts: Dict[str, Any], cleaned_text: str, k: int = 12) -> List[Dict[str, Any]]:
    """
    Explains using fallback LogisticRegression pipeline if available.
    Works when pipeline contains a 'features' step with a FeatureUnion that includes 'tfidf'.
    """
    fallback = artifacts.get("fallback_model")
    if fallback is None:
        return []

    try:
        feats = fallback.named_steps["features"]
        tfidf = feats.transformer_list[0][1]  # ("tfidf", TfidfVectorizer(...))
        clf = fallback.named_steps["clf"]

        if not hasattr(clf, "coef_"):
            return []

        X_tfidf = tfidf.transform([cleaned_text])
        vocab = tfidf.get_feature_names_out()
        coef = clf.coef_[0]  # positive class = spam (label 1)

        contrib = X_tfidf.multiply(coef).toarray()[0]
        idx = np.argsort(contrib)[::-1][:k]
        out = []
        for i in idx:
            if contrib[i] <= 0:
                continue
            out.append({"token": str(vocab[i]), "score": float(contrib[i])})
        return out
    except Exception:
        return []


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Spam Email Predictor", page_icon="📩", layout="centered")

st.title("📩 Spam Email Predictor (Upgraded)")
st.caption("Subject + body, smarter preprocessing, stronger models, thresholding, batch mode, and explainability.")

with st.sidebar:
    st.header("⚙️ Settings")
    artifact_path = st.text_input("Model file", value=ARTIFACT_PATH_DEFAULT)
    threshold = st.slider("Spam threshold", min_value=0.01, max_value=0.99, value=0.70, step=0.01)
    show_debug = st.toggle("Show debug info", value=False)

    st.divider()
    st.subheader("📦 Batch mode (CSV)")
    st.caption("CSV columns supported: subject/body OR Subject/Body OR Message")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

# Load model
artifacts = None
try:
    artifacts = load_artifacts(artifact_path)
except Exception as e:
    st.error(f"Could not load model file: {artifact_path}")
    st.code(str(e))

# History
if "history" not in st.session_state:
    st.session_state["history"] = []

tab1, tab2, tab3 = st.tabs(["Single Email", "Batch CSV", "History / Feedback"])

with tab1:
    colA, colB = st.columns(2)
    subject = colA.text_input("Subject", placeholder="e.g., Urgent: Verify your account")
    body = st.text_area("Body", placeholder="Paste email body here...", height=220)

    predict_btn = st.button("Predict", type="primary", use_container_width=True, disabled=(artifacts is None))
    if predict_btn and artifacts is not None:
        result = predict_spam(artifacts, subject, body, threshold)
        pred_label = result["pred_label"]
        p_spam = result["p_spam"]

        if p_spam is not None:
            st.progress(min(max(p_spam, 0.0), 1.0))
            st.caption(f"Spam probability: **{p_spam:.2%}** (threshold **{threshold:.2f}**)")

        if pred_label == 1:
            st.error("🚫 Prediction: **SPAM**")
        else:
            st.success("✅ Prediction: **NOT SPAM (HAM)**")

        st.subheader("Why did it say that?")
        contribs = top_token_contributions(artifacts, result["text"], k=15)
        if contribs:
            st.write("Top spam-indicating tokens (from Logistic Regression explainer):")
            st.dataframe(pd.DataFrame(contribs))
        else:
            st.info("No explainer available (train.py stores a Logistic Regression fallback for explanations).")

        st.session_state["history"].insert(0, {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "subject": subject,
            "p_spam": p_spam,
            "prediction": "spam" if pred_label == 1 else "ham",
            "lang": result.get("lang"),
            "used_multilingual": result.get("used_multilingual"),
        })
        st.session_state["history"] = st.session_state["history"][:10]

        if show_debug:
            st.json(result)

with tab2:
    if uploaded is None:
        st.info("Upload a CSV from the sidebar to run batch predictions.")
    elif artifacts is None:
        st.warning("Model not loaded.")
    else:
        df = pd.read_csv(uploaded)
        cols = {c.lower(): c for c in df.columns}

        if "message" in cols:
            subjects = [""] * len(df)
            bodies = df[cols["message"]].astype(str).fillna("").tolist()
        else:
            subj_col = cols.get("subject") or cols.get("subj") or cols.get("title")
            body_col = cols.get("body") or cols.get("text") or cols.get("content")

            if subj_col is None or body_col is None:
                st.error("CSV must contain either `Message` OR (`subject` and `body`) columns.")
                st.write("Columns found:", list(df.columns))
                st.stop()

            subjects = df[subj_col].astype(str).fillna("").tolist()
            bodies = df[body_col].astype(str).fillna("").tolist()

        results = []
        for s, b in zip(subjects, bodies):
            r = predict_spam(artifacts, s, b, threshold)
            results.append({
                "subject": s,
                "prediction": "spam" if r["pred_label"] == 1 else "ham",
                "p_spam": r["p_spam"],
                "lang": r.get("lang"),
                "used_multilingual": r.get("used_multilingual"),
            })

        out_df = pd.DataFrame(results)
        st.dataframe(out_df, use_container_width=True)

        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results CSV", data=csv_bytes, file_name="spam_predictions.csv", mime="text/csv")

with tab3:
    st.subheader("Recent predictions")
    hist = st.session_state.get("history", [])
    if not hist:
        st.info("No history yet.")
    else:
        st.dataframe(pd.DataFrame(hist), use_container_width=True)

    st.divider()
    st.subheader("Feedback loop")
    st.caption("If a prediction is wrong, record it for retraining. Saved locally as `feedback.csv`.")
    feedback_path = Path("feedback.csv")

    actual = st.selectbox("Actual label", options=["spam", "ham"], index=0)
    save_feedback = st.button("Save feedback for latest entry", disabled=(len(hist) == 0))
    if save_feedback and hist:
        latest = hist[0]
        row = {
            "time": datetime.now().isoformat(),
            "subject": latest.get("subject", ""),
            "actual_label": actual,
            "predicted": latest.get("prediction"),
            "p_spam": latest.get("p_spam"),
        }
        if feedback_path.exists():
            fb = pd.read_csv(feedback_path)
            fb = pd.concat([fb, pd.DataFrame([row])], ignore_index=True)
        else:
            fb = pd.DataFrame([row])
        fb.to_csv(feedback_path, index=False)
        st.success(f"Saved feedback to {feedback_path.resolve()}")
