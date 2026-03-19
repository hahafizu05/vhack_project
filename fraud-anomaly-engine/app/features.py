from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def stable_hash(value: Any, *, salt: str = "saml-d") -> str:
    s = f"{salt}::{value}".encode("utf-8", errors="replace")
    return hashlib.sha256(s).hexdigest()[:16]


def parse_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    if not date_str or not time_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(f"{date_str} {time_str}", fmt)
        except ValueError:
            pass
    return None


@dataclass(frozen=True)
class EntityStats:
    count: int
    mean_amount: float
    std_amount: float
    max_amount: float
    unique_receivers: int
    cross_border_rate: float
    night_rate: float


def compute_sender_stats(df: pd.DataFrame) -> Dict[str, EntityStats]:
    # Expects hashed ids in Sender_id/Receiver_id
    if df.empty:
        return {}

    # basic helpers
    cross_border = (df["Sender_bank_location"] != df["Receiver_bank_location"]).astype(np.int8)
    night = df["Hour"].between(0, 5).astype(np.int8)

    grouped = df.assign(_cross=cross_border, _night=night).groupby("Sender_id", sort=False)

    stats: Dict[str, EntityStats] = {}
    for sender_id, g in grouped:
        amounts = g["Amount"].astype("float64")
        cnt = int(len(g))
        mean = float(amounts.mean()) if cnt else 0.0
        std = float(amounts.std(ddof=0)) if cnt else 0.0
        mx = float(amounts.max()) if cnt else 0.0
        uniq_recv = int(g["Receiver_id"].nunique())
        cbr = float(g["_cross"].mean()) if cnt else 0.0
        nr = float(g["_night"].mean()) if cnt else 0.0
        stats[str(sender_id)] = EntityStats(
            count=cnt,
            mean_amount=mean,
            std_amount=std if std > 1e-9 else 1.0,
            max_amount=mx,
            unique_receivers=uniq_recv,
            cross_border_rate=cbr,
            night_rate=nr,
        )
    return stats


def add_derived_features(
    df: pd.DataFrame,
    *,
    sender_stats: Optional[Dict[str, EntityStats]] = None,
    salt: str = "saml-d",
) -> pd.DataFrame:
    out = df.copy()

    out["Sender_id"] = out["Sender_account"].map(lambda v: stable_hash(v, salt=salt)).astype("string")
    out["Receiver_id"] = out["Receiver_account"].map(lambda v: stable_hash(v, salt=salt)).astype("string")

    # Parse time features
    def extract_datetime(r):
        date_str = r.get("Date", "") if "Date" in r.index else ""
        time_str = r.get("Time", "") if "Time" in r.index else ""
        return parse_datetime(str(date_str), str(time_str))
    
    dt = out.apply(extract_datetime, axis=1)
    out["Hour"] = dt.map(lambda x: x.hour if x else np.nan).astype("float64")
    out["DayOfWeek"] = dt.map(lambda x: x.weekday() if x else np.nan).astype("float64")

    out["IsCrossBorder"] = (out["Sender_bank_location"] != out["Receiver_bank_location"]).astype(np.int8)
    out["LogAmount"] = np.log1p(pd.to_numeric(out["Amount"], errors="coerce").astype("float64"))

    if sender_stats is None:
        # training-time: can fill later after stats computed
        out["SenderTxnCount"] = np.nan
        out["SenderMeanAmount"] = np.nan
        out["SenderStdAmount"] = np.nan
        out["SenderMaxAmount"] = np.nan
        out["SenderUniqueReceivers"] = np.nan
        out["SenderCrossBorderRate"] = np.nan
        out["SenderNightRate"] = np.nan
        out["SenderZAmount"] = np.nan
        return out

    def lookup(sender_id: str) -> Optional[EntityStats]:
        return sender_stats.get(str(sender_id))

    ss = out["Sender_id"].map(lookup)
    out["SenderTxnCount"] = ss.map(lambda s: s.count if s else 0).astype("int64")
    out["SenderMeanAmount"] = ss.map(lambda s: s.mean_amount if s else 0.0).astype("float64")
    out["SenderStdAmount"] = ss.map(lambda s: s.std_amount if s else 1.0).astype("float64")
    out["SenderMaxAmount"] = ss.map(lambda s: s.max_amount if s else 0.0).astype("float64")
    out["SenderUniqueReceivers"] = ss.map(lambda s: s.unique_receivers if s else 0).astype("int64")
    out["SenderCrossBorderRate"] = ss.map(lambda s: s.cross_border_rate if s else 0.0).astype("float64")
    out["SenderNightRate"] = ss.map(lambda s: s.night_rate if s else 0.0).astype("float64")

    amount = pd.to_numeric(out["Amount"], errors="coerce").astype("float64")
    out["SenderZAmount"] = (amount - out["SenderMeanAmount"]) / out["SenderStdAmount"]
    return out


def model_columns() -> Tuple[list[str], list[str]]:
    numeric = [
        "Amount",
        "LogAmount",
        "Hour",
        "DayOfWeek",
        "IsCrossBorder",
        "SenderTxnCount",
        "SenderMeanAmount",
        "SenderStdAmount",
        "SenderMaxAmount",
        "SenderUniqueReceivers",
        "SenderCrossBorderRate",
        "SenderNightRate",
        "SenderZAmount",
    ]
    categorical = [
        "Payment_currency",
        "Received_currency",
        "Sender_bank_location",
        "Receiver_bank_location",
        "Payment_type",
    ]
    return numeric, categorical

