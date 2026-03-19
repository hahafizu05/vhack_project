from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .features import EntityStats, add_derived_features


ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class TransactionIn(BaseModel):
    Time: str = Field(..., examples=["10:35:19"])
    Date: str = Field(..., examples=["2022-10-07"])
    Sender_account: str
    Receiver_account: str
    Amount: float
    Payment_currency: str
    Received_currency: str
    Sender_bank_location: str
    Receiver_bank_location: str
    Payment_type: str

    # Optional contextual signals (if you have them); not used unless you extend features
    ip_reputation: Optional[float] = None
    device_risk: Optional[float] = None


class ScoreOut(BaseModel):
    risk_score: float
    decision: Literal["Approve", "Flag", "Block"]
    thresholds: Dict[str, float]


def load_sender_stats() -> Dict[str, EntityStats]:
    raw: Dict[str, Dict[str, Any]] = joblib.load(ARTIFACT_DIR / "sender_stats.joblib")
    return {k: EntityStats(**v) for k, v in raw.items()}


def load_thresholds() -> Dict[str, float]:
    if (ARTIFACT_DIR / "metrics.joblib").exists():
        m = joblib.load(ARTIFACT_DIR / "metrics.joblib")
        base_flag = float(m.get("flag_threshold", 0.25))
        base_block = float(m.get("block_threshold", 0.6))
        # Demo-friendly: slightly lower thresholds so extreme scenarios
        # more easily show Flag / Block in the UI.
        flag = max(0.02, base_flag * 0.6)
        block = max(flag + 0.05, base_block * 0.8)
        return {"flag": flag, "block": block}
    return {"flag": 0.25, "block": 0.6}


app = FastAPI(title="Fraud Risk API Prototype", version="0.1.0")

if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

_MODEL = None
_SENDER_STATS: Optional[Dict[str, EntityStats]] = None
_THR: Optional[Dict[str, float]] = None


@app.on_event("startup")
def _startup() -> None:
    global _MODEL, _SENDER_STATS, _THR
    _MODEL = joblib.load(ARTIFACT_DIR / "model.joblib")
    _SENDER_STATS = load_sender_stats()
    _THR = load_thresholds()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """
    Serve the main UI shell. Kept as a simple single-page app for low latency.
    """
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Fraud Risk API</h1><p>Frontend not found.</p>", status_code=200)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/score", response_model=ScoreOut)
def score(tx: TransactionIn) -> ScoreOut:
    assert _MODEL is not None
    assert _SENDER_STATS is not None
    thr = _THR or load_thresholds()

    df = pd.DataFrame([tx.model_dump(exclude_none=True)])
    df = add_derived_features(df, sender_stats=_SENDER_STATS)

    proba = float(_MODEL.predict_proba(df)[:, 1][0])

    if proba >= thr["block"]:
        decision: Literal["Approve", "Flag", "Block"] = "Block"
    elif proba >= thr["flag"]:
        decision = "Flag"
    else:
        decision = "Approve"

    return ScoreOut(risk_score=proba, decision=decision, thresholds=thr)

