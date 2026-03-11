from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"


def get_project_root() -> Path:
    return PROJECT_ROOT


def ensure_project_structure() -> None:
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        METRICS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    return value


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 120,
        }
    )


def min_max_scale(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    low = np.nanmin(array)
    high = np.nanmax(array)
    if np.isclose(low, high):
        return np.zeros_like(array, dtype=float)
    return (array - low) / (high - low)


def derive_risk_bucket(score: float) -> str:
    if score >= 0.8:
        return "Critical"
    if score >= 0.6:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def explain_risk_row(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if row.get("repeat_rate", 0) > 0.6 or row.get("top_song_share", 0) > 0.45:
        reasons.append("extreme same-song repetition")
    if row.get("top_artist_share", 0) > 0.65 or row.get("artist_concentration_score", 0) > 0.65:
        reasons.append("heavy concentration on a single artist")
    if row.get("night_activity_ratio", 0) > 0.35:
        reasons.append("unusually high overnight activity")
    if row.get("plays_per_hour", 0) > 8 or row.get("burstiness_score", 0) > 5:
        reasons.append("stream velocity inconsistent with typical listening")
    if row.get("shared_ip_count", 0) >= 5 or row.get("shared_device_count", 0) >= 4:
        reasons.append("shared infrastructure across many accounts")
    if row.get("completion_rate", 0) > 0.97 and row.get("skip_rate", 1) < 0.05:
        reasons.append("near-perfect completion behavior")
    return reasons or ["behavior within expected bounds"]


def run_full_pipeline(
    force_refresh: bool = False,
    n_users: int = 2000,
    suspicious_share: float = 0.12,
    random_state: int = 42,
) -> dict[str, Any]:
    ensure_project_structure()

    events_path = RAW_DATA_DIR / "stream_events.csv"
    users_path = PROCESSED_DATA_DIR / "user_level_features.csv"
    sessions_path = PROCESSED_DATA_DIR / "session_level_features.csv"
    scores_path = PROCESSED_DATA_DIR / "scored_user_predictions.csv"
    metrics_path = METRICS_DIR / "model_metrics.csv"
    business_path = METRICS_DIR / "business_metrics.csv"
    threshold_path = METRICS_DIR / "threshold_tradeoffs.csv"
    importance_path = TABLES_DIR / "feature_importance.csv"
    summary_path = METRICS_DIR / "simulation_summary.json"

    required_paths = (
        events_path,
        users_path,
        sessions_path,
        scores_path,
        metrics_path,
        business_path,
        threshold_path,
        importance_path,
        summary_path,
    )

    if not force_refresh and all(path.exists() for path in required_paths):
        summary = load_json(summary_path)
        requested_matches = (
            int(summary.get("n_users", -1)) == int(n_users)
            and abs(float(summary.get("suspicious_share", -1)) - float(suspicious_share)) < 1e-12
            and int(summary.get("random_state", -1)) == int(random_state)
        )
        if requested_matches:
            return {
                "events": pd.read_csv(events_path, parse_dates=["timestamp"]),
                "user_features": pd.read_csv(users_path),
                "session_features": pd.read_csv(sessions_path, parse_dates=["session_start", "session_end"]),
                "scored_users": pd.read_csv(scores_path),
                "metrics": pd.read_csv(metrics_path),
                "business_metrics": pd.read_csv(business_path),
                "threshold_tradeoffs": pd.read_csv(threshold_path),
                "feature_importance": pd.read_csv(importance_path),
            }

    from src.data_simulation import SimulationConfig, simulate_streaming_data
    from src.evaluation import compile_evaluation_outputs
    from src.feature_engineering import build_feature_tables
    from src.modeling import train_model_suite

    config = SimulationConfig(
        n_users=n_users,
        suspicious_share=suspicious_share,
        random_state=random_state,
    )
    events = simulate_streaming_data(config=config, save_outputs=True)
    user_features, session_features = build_feature_tables(events=events, save_outputs=True)
    model_outputs = train_model_suite(user_features=user_features, save_outputs=True, random_state=random_state)
    metrics_bundle = compile_evaluation_outputs(
        scored_users=model_outputs["scored_users"],
        save_outputs=True,
    )

    return {
        "events": events,
        "user_features": user_features,
        "session_features": session_features,
        "scored_users": model_outputs["scored_users"],
        "metrics": metrics_bundle["metrics"],
        "business_metrics": metrics_bundle["business_metrics"],
        "threshold_tradeoffs": metrics_bundle["threshold_tradeoffs"],
        "feature_importance": model_outputs["feature_importance"],
    }
