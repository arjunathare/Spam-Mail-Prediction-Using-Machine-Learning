import argparse
import hashlib
import json
import re
import html
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
    accuracy_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, TransformerMixin

from scipy import sparse as sp  # type: ignore


# Optional language detection + embeddings
try:
    from langdetect import detect  # type: ignore
except Exception:
    detect = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore


_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
_PHONE_RE = re.compile(r"(\+?\d[\d\-\s\(\)]{8,}\d)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")

def clean_text(text: str) -> str:
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

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# -------------------------
# Heuristic phishing signals
# -------------------------

SUSPICIOUS_KEYWORDS = [
    "urgent", "verify", "password", "reset", "account", "login", "bank",
    "limited time", "act now", "click", "confirm", "win", "prize", "free",
    "payment", "invoice", "refund", "otp", "kyc",
]

def extract_heuristics(text: str) -> np.ndarray:
    t = text or ""
    url_count = len(_URL_RE.findall(t))
    email_count = len(_EMAIL_RE.findall(t))
    phone_count = len(_PHONE_RE.findall(t))

    letters = [c for c in t if c.isalpha()]
    upper = [c for c in letters if c.isupper()]
    caps_ratio = (len(upper) / max(1, len(letters)))

    exclam = t.count("!")
    question = t.count("?")
    currency = t.count("$") + t.count("₹")

    lower = t.lower()
    keyword_hits = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in lower)

    length = len(t)
    word_count = len(t.split())

    return np.array([
        url_count, email_count, phone_count,
        caps_ratio, exclam, question, currency,
        keyword_hits, length, word_count
    ], dtype=float)


class HeuristicFeaturesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = np.vstack([extract_heuristics(x) for x in X])
        return sp.csr_matrix(feats)


# -------------------------
# Data loading
# -------------------------

def load_dataset(csv_path: Path) -> Tuple[List[str], np.ndarray]:
    """
    Expects columns:
      - Category: 'spam'/'ham'
      - Message: email body text
    """
    df = pd.read_csv(csv_path)
    df = df.where(pd.notnull(df), "")

    if "Category" not in df.columns or "Message" not in df.columns:
        raise ValueError("CSV must contain columns: Category, Message")

    y = (df["Category"].astype(str).str.lower().str.strip() == "spam").astype(int).to_numpy()
    X = df["Message"].astype(str).map(clean_text).tolist()
    return X, y


@dataclass
class TrainMeta:
    trained_at: str
    dataset_path: str
    dataset_sha256: Optional[str]
    label_map: Dict[str, int]
    split_random_state: int
    test_size: float
    models_tried: List[str]
    best_model_name: str
    best_params: Dict[str, Any]
    metrics_test: Dict[str, Any]


def build_feature_union() -> FeatureUnion:
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        sublinear_tf=True,
    )
    heur = HeuristicFeaturesTransformer()

    return FeatureUnion([
        ("tfidf", tfidf),
        ("heur", heur),
    ])


def train_and_select(X: List[str], y: np.ndarray, seed: int = 42) -> Tuple[Any, Any, Dict[str, Any]]:
    features = build_feature_union()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Logistic Regression
    pipe_lr = Pipeline([
        ("features", features),
        ("clf", LogisticRegression(max_iter=300, class_weight="balanced")),
    ])
    grid_lr = {
        "features__tfidf__min_df": [1, 2, 3],
        "features__tfidf__max_df": [0.9, 0.95, 0.99],
        "clf__C": [0.5, 1.0, 2.0, 4.0],
    }
    gs_lr = GridSearchCV(pipe_lr, grid_lr, scoring="f1", cv=cv, n_jobs=-1)
    gs_lr.fit(X, y)
    best_lr = gs_lr.best_estimator_

    # Multinomial NB
    pipe_nb = Pipeline([
        ("features", features),
        ("clf", MultinomialNB()),
    ])
    grid_nb = {
        "features__tfidf__min_df": [1, 2, 3],
        "features__tfidf__max_df": [0.9, 0.95, 0.99],
        "clf__alpha": [0.1, 0.5, 1.0],
    }
    gs_nb = GridSearchCV(pipe_nb, grid_nb, scoring="f1", cv=cv, n_jobs=-1)
    gs_nb.fit(X, y)
    best_nb = gs_nb.best_estimator_

    # LinearSVC (calibrated)
    pipe_svm = Pipeline([
        ("features", features),
        ("clf", LinearSVC(class_weight="balanced")),
    ])
    grid_svm = {
        "features__tfidf__min_df": [1, 2, 3],
        "features__tfidf__max_df": [0.9, 0.95, 0.99],
        "clf__C": [0.5, 1.0, 2.0, 4.0],
    }
    gs_svm = GridSearchCV(pipe_svm, grid_svm, scoring="f1", cv=cv, n_jobs=-1)
    gs_svm.fit(X, y)
    best_svm = gs_svm.best_estimator_

    calibrated_svm = CalibratedClassifierCV(best_svm, method="sigmoid", cv=3)
    calibrated_svm.fit(X, y)

    scores = {
        "logreg": float(gs_lr.best_score_),
        "mnb": float(gs_nb.best_score_),
        "svm_calibrated": float(gs_svm.best_score_),
    }
    best_name = max(scores, key=scores.get)

    if best_name == "logreg":
        best_model = best_lr
        best_params = gs_lr.best_params_
    elif best_name == "mnb":
        best_model = best_nb
        best_params = gs_nb.best_params_
    else:
        best_model = calibrated_svm
        best_params = gs_svm.best_params_

    best_info = {"scores_cv_f1": scores, "selected": best_name, "best_params": best_params}
    return best_model, best_lr, best_info


def evaluate(model: Any, X_test: List[str], y_test: np.ndarray) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "classification_report": report,
    }


def train_multilingual_embeddings(
    X: List[str], y: np.ndarray, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
) -> Optional[Any]:
    if SentenceTransformer is None:
        return None

    encoder = SentenceTransformer(model_name)
    emb = encoder.encode(X, show_progress_bar=True, normalize_embeddings=True)

    clf = LogisticRegression(max_iter=400, class_weight="balanced")
    clf.fit(emb, y)

    class EmbeddingModel:
        def __init__(self, enc, clf_):
            self.enc = enc
            self.clf = clf_
            self.classes_ = clf_.classes_

        def predict_proba(self, X_text):
            E = self.enc.encode(list(X_text), show_progress_bar=False, normalize_embeddings=True)
            return self.clf.predict_proba(E)

        def predict(self, X_text):
            E = self.enc.encode(list(X_text), show_progress_bar=False, normalize_embeddings=True)
            return self.clf.predict(E)

    return EmbeddingModel(encoder, clf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="mail_data.csv")
    parser.add_argument("--out", type=str, default="spam_model.joblib")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--train_multilingual", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path.resolve()}")

    X, y = load_dataset(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    best_model, fallback_lr, best_info = train_and_select(X_train, y_train, seed=args.seed)
    metrics_test = evaluate(best_model, X_test, y_test)

    multilingual_model = None
    if args.train_multilingual:
        multilingual_model = train_multilingual_embeddings(X_train, y_train)

    meta = TrainMeta(
        trained_at=datetime.utcnow().isoformat() + "Z",
        dataset_path=str(data_path),
        dataset_sha256=sha256_file(data_path),
        label_map={"ham": 0, "spam": 1},
        split_random_state=args.seed,
        test_size=args.test_size,
        models_tried=["LogisticRegression", "MultinomialNB", "LinearSVC (calibrated)"],
        best_model_name=best_info["selected"],
        best_params=best_info["best_params"],
        metrics_test=metrics_test,
    )

    artifacts = {
        "best_model": best_model,
        "fallback_model": fallback_lr,
        "multilingual_model": multilingual_model,
        "label_map": meta.label_map,
        "meta": asdict(meta),
        "selection_info": best_info,
    }

    out_path = Path(args.out)
    joblib.dump(artifacts, out_path)

    print(f"Saved: {out_path.resolve()}")
    print("Test metrics (summary):")
    print(json.dumps({k: metrics_test[k] for k in ["accuracy", "precision", "recall", "f1", "confusion_matrix"]}, indent=2))


if __name__ == "__main__":
    main()
