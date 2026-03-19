from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import EntityStats, add_derived_features, compute_sender_stats, model_columns


ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def read_chunked_all(
    csv_path: str,
    *,
    chunksize: int = 200_000,
    neg_to_pos_ratio: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Reads the CSV in chunks but keeps **all** rows (no negative sampling).
    This still streams from disk in pieces so memory usage is reasonable,
    but the final dataframe contains every row from the file.
    """
    chunks = []

    usecols = [
        "Time",
        "Date",
        "Sender_account",
        "Receiver_account",
        "Amount",
        "Payment_currency",
        "Received_currency",
        "Sender_bank_location",
        "Receiver_bank_location",
        "Payment_type",
        "Is_laundering",
    ]

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=usecols):
        chunk["Is_laundering"] = pd.to_numeric(chunk["Is_laundering"], errors="coerce").fillna(0).astype("int8")
        chunks.append(chunk)

    if not chunks:
        raise RuntimeError("No data read from CSV. Check path/format.")

    df = pd.concat(chunks, ignore_index=True)
    return df


def build_pipeline() -> Pipeline:
    numeric, categorical = model_columns()

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical),
        ],
        remainder="drop",
    )

    clf = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_samples=50,
        objective="binary",
        # imbalance handling
        scale_pos_weight=20.0,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([("pre", pre), ("clf", clf)])


def choose_threshold(y_true: np.ndarray, y_proba: np.ndarray, target_precision: float = 0.90) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # thresholds has length n-1; align by dropping last p/r
    precision = precision[:-1]
    recall = recall[:-1]
    if thresholds.size == 0:
        return 0.5, 0.5

    mask = precision >= target_precision
    if mask.any():
        # among those, pick best recall
        idx = int(np.argmax(recall[mask]))
        thr = float(thresholds[mask][idx])
        achieved_precision = float(precision[mask][idx])
        return thr, achieved_precision

    # fallback: maximize F1
    f1 = 2 * (precision * recall) / np.clip(precision + recall, 1e-12, None)
    thr = float(thresholds[int(np.argmax(f1))])
    return thr, float(precision[int(np.argmax(f1))])


def main() -> None:
    csv_path = os.getenv("SAML_CSV_PATH", r"C:\Users\Asus\Downloads\SAML-D.csv\SAML-D.csv")
    df = read_chunked_all(csv_path)

    # Privacy-first: do NOT carry raw ids into artifacts/features beyond transient use
    df = add_derived_features(df)

    # Build behavioral baseline (sender stats) on training fold only
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Is_laundering"])

    sender_stats = compute_sender_stats(train_df)
    train_df = add_derived_features(train_df, sender_stats=sender_stats)
    test_df = add_derived_features(test_df, sender_stats=sender_stats)

    y_train = train_df["Is_laundering"].to_numpy(dtype=np.int8)
    y_test = test_df["Is_laundering"].to_numpy(dtype=np.int8)

    pipe = build_pipeline()
    pipe.fit(train_df, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    ap = float(average_precision_score(y_test, proba))
    auc = float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan")

    block_thr, achieved_precision = choose_threshold(y_test, proba, target_precision=0.90)
    flag_thr = float(max(0.05, min(block_thr * 0.5, block_thr - 1e-6)))

    # Save artifacts
    joblib.dump(pipe, ARTIFACT_DIR / "model.joblib")
    joblib.dump({k: asdict(v) for k, v in sender_stats.items()}, ARTIFACT_DIR / "sender_stats.joblib")
    joblib.dump(
        {
            "average_precision": ap,
            "roc_auc": auc,
            "flag_threshold": flag_thr,
            "block_threshold": block_thr,
            "block_threshold_precision_on_val": achieved_precision,
        },
        ARTIFACT_DIR / "metrics.joblib",
    )

    print("Saved artifacts to:", ARTIFACT_DIR)
    print({"average_precision": ap, "roc_auc": auc, "flag_threshold": flag_thr, "block_threshold": block_thr})


if __name__ == "__main__":
    main()

