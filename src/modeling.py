from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import MODEL_DIR, PROCESSED_DATA_DIR, TABLES_DIR, min_max_scale, save_dataframe


DEFAULT_FEATURE_COLUMNS = [
    "account_age_days",
    "total_streams",
    "total_sessions",
    "active_days",
    "avg_streams_per_day",
    "plays_per_hour",
    "repeat_rate",
    "completion_rate",
    "skip_rate",
    "artist_diversity",
    "song_diversity",
    "playlist_diversity",
    "top_artist_share",
    "top_song_share",
    "avg_session_length",
    "median_session_length",
    "avg_streams_per_session",
    "sessions_per_day",
    "night_activity_ratio",
    "burstiness_score",
    "avg_inter_stream_gap_sec",
    "median_inter_stream_gap_sec",
    "same_song_repeat_ratio",
    "device_sharing_score",
    "ip_sharing_score",
    "shared_device_count",
    "shared_ip_count",
    "unique_songs_7d",
    "unique_artists_7d",
    "artist_concentration_score",
    "max_daily_streams",
    "max_hourly_streams",
    "daily_listening_count_mean",
    "hourly_listening_count_mean",
    "completion_gap",
    "session_length_std",
]


def train_model_suite(
    user_features: pd.DataFrame,
    feature_columns: list[str] | None = None,
    random_state: int = 42,
    save_outputs: bool = True,
) -> dict[str, Any]:
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    model_frame = user_features.copy()
    X = model_frame[feature_columns].fillna(model_frame[feature_columns].median())
    y = model_frame["label_suspicious"].astype(int)

    train_idx, test_idx = train_test_split(
        model_frame.index,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    contamination = float(max(y_train.mean(), 0.05))
    isolation_scaler = StandardScaler()
    X_train_scaled = isolation_scaler.fit_transform(X_train)
    X_test_scaled = isolation_scaler.transform(X_test)

    isolation_forest = IsolationForest(
        n_estimators=350,
        max_samples="auto",
        contamination=contamination,
        random_state=random_state,
    )
    isolation_forest.fit(X_train_scaled)
    iso_train_score = -isolation_forest.score_samples(X_train_scaled)
    iso_test_score = -isolation_forest.score_samples(X_test_scaled)
    iso_full_score = np.zeros(len(model_frame))
    iso_full_score[train_idx] = min_max_scale(iso_train_score)
    iso_full_score[test_idx] = min_max_scale(iso_test_score)
    iso_threshold = float(np.quantile(min_max_scale(iso_train_score), 1 - contamination))

    random_forest = RandomForestClassifier(
        n_estimators=450,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=random_state,
    )
    random_forest.fit(X_train, y_train)

    logistic_regression = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )
    logistic_regression.fit(X_train, y_train)

    scored_users = model_frame.copy()
    scored_users["split"] = np.where(scored_users.index.isin(test_idx), "test", "train")
    scored_users["isolation_forest_score"] = iso_full_score
    scored_users["isolation_forest_pred"] = (scored_users["isolation_forest_score"] >= iso_threshold).astype(int)
    scored_users["random_forest_score"] = random_forest.predict_proba(X)[:, 1]
    scored_users["random_forest_pred"] = (scored_users["random_forest_score"] >= 0.5).astype(int)
    scored_users["logistic_regression_score"] = logistic_regression.predict_proba(X)[:, 1]
    scored_users["logistic_regression_pred"] = (
        scored_users["logistic_regression_score"] >= 0.5
    ).astype(int)
    scored_users["risk_score"] = (
        0.55 * scored_users["random_forest_score"]
        + 0.25 * scored_users["isolation_forest_score"]
        + 0.20 * scored_users["logistic_regression_score"]
    )

    permutation = permutation_importance(
        random_forest,
        X_test,
        y_test,
        n_repeats=10,
        random_state=random_state,
        scoring="average_precision",
    )
    logistic_model = logistic_regression.named_steps["model"]
    feature_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "random_forest_importance": random_forest.feature_importances_,
            "permutation_importance": permutation.importances_mean,
            "logistic_coefficient": logistic_model.coef_[0],
            "abs_logistic_coefficient": np.abs(logistic_model.coef_[0]),
        }
    ).sort_values("permutation_importance", ascending=False)

    if save_outputs:
        save_dataframe(scored_users, PROCESSED_DATA_DIR / "scored_user_predictions.csv")
        save_dataframe(feature_importance, TABLES_DIR / "feature_importance.csv")
        joblib.dump(
            {
                "feature_columns": feature_columns,
                "isolation_scaler": isolation_scaler,
                "isolation_forest": isolation_forest,
                "random_forest": random_forest,
                "logistic_regression": logistic_regression,
                "isolation_threshold": iso_threshold,
            },
            MODEL_DIR / "model_suite.joblib",
        )

    return {
        "scored_users": scored_users,
        "feature_importance": feature_importance,
        "model_bundle_path": Path(MODEL_DIR / "model_suite.joblib"),
        "feature_columns": feature_columns,
        "test_indices": list(test_idx),
    }

