from __future__ import annotations

from pathlib import Path

import pandas as pd


MODEL_COLUMNS = [
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "bytes_sent",
    "bytes_received",
    "user_agent",
    "url",
    "is_internal_traffic",
]


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the same model inputs used during training."""
    features = frame.drop(columns=["label", "attack_type"], errors="ignore").copy()

    for column in MODEL_COLUMNS:
        if column not in features:
            features[column] = pd.NA
    features = features[MODEL_COLUMNS]

    timestamp = pd.to_datetime(features.pop("timestamp"), errors="coerce")
    features["hour"] = timestamp.dt.hour
    features["dayofweek"] = timestamp.dt.dayofweek
    features["month"] = timestamp.dt.month

    for column in ["src_ip", "dst_ip"]:
        octets = features.pop(column).fillna("").astype(str).str.split(".", expand=True)
        for index in range(4):
            features[f"{column}_octet_{index + 1}"] = pd.to_numeric(
                octets[index], errors="coerce"
            )

    for column in ["url", "user_agent"]:
        text = features[column].fillna("").astype(str)
        features[f"{column}_length"] = text.str.len()
        features[f"{column}_has_query"] = text.str.contains(
            r"[?=&]", regex=True
        ).astype(int)

    return features.drop(columns="url")


def load_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
