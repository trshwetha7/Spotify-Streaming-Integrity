from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import METRICS_DIR, save_dataframe


MODEL_SPECS = {
    "isolation_forest": {"score_col": "isolation_forest_score", "pred_col": "isolation_forest_pred"},
    "random_forest": {"score_col": "random_forest_score", "pred_col": "random_forest_pred"},
    "logistic_regression": {
        "score_col": "logistic_regression_score",
        "pred_col": "logistic_regression_pred",
    },
}


def _safe_metric(metric_fn, y_true: pd.Series, y_score: pd.Series) -> float:
    try:
        return float(metric_fn(y_true, y_score))
    except ValueError:
        return float("nan")


def model_metrics_table(scored_users: pd.DataFrame) -> pd.DataFrame:
    test_frame = scored_users.loc[scored_users["split"] == "test"].copy()
    rows: list[dict[str, float | str]] = []

    for model_name, columns in MODEL_SPECS.items():
        y_true = test_frame["label_suspicious"]
        y_pred = test_frame[columns["pred_col"]]
        y_score = test_frame[columns["score_col"]]
        rows.append(
            {
                "model": model_name,
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "roc_auc": _safe_metric(roc_auc_score, y_true, y_score),
                "pr_auc": _safe_metric(average_precision_score, y_true, y_score),
            }
        )

    return pd.DataFrame(rows).sort_values("f1", ascending=False)


def business_metrics_table(scored_users: pd.DataFrame) -> pd.DataFrame:
    test_frame = scored_users.loc[scored_users["split"] == "test"].copy()
    suspicious_streams_total = test_frame.loc[
        test_frame["label_suspicious"] == 1, "total_streams"
    ].sum()
    total_streams = test_frame["total_streams"].sum()
    rows: list[dict[str, float | str]] = []

    for model_name, columns in MODEL_SPECS.items():
        pred_col = columns["pred_col"]
        flagged = test_frame[pred_col] == 1
        suspicious_flagged = test_frame["label_suspicious"].eq(1) & flagged
        legitimate_flagged = test_frame["label_suspicious"].eq(0) & flagged

        suspicious_stream_capture = (
            test_frame.loc[suspicious_flagged, "total_streams"].sum() / suspicious_streams_total
            if suspicious_streams_total
            else np.nan
        )
        legitimate_user_base = max(int(test_frame["label_suspicious"].eq(0).sum()), 1)
        false_positive_rate = legitimate_flagged.sum() / legitimate_user_base

        rows.append(
            {
                "model": model_name,
                "review_load_pct_accounts": float(flagged.mean()),
                "review_load_pct_streams": float(test_frame.loc[flagged, "total_streams"].sum() / total_streams),
                "suspicious_stream_capture": float(suspicious_stream_capture),
                "false_positive_rate_legit_users": float(false_positive_rate),
            }
        )

    return pd.DataFrame(rows).sort_values("suspicious_stream_capture", ascending=False)


def confusion_matrix_table(scored_users: pd.DataFrame, model_name: str) -> pd.DataFrame:
    test_frame = scored_users.loc[scored_users["split"] == "test"].copy()
    spec = MODEL_SPECS[model_name]
    matrix = confusion_matrix(
        test_frame["label_suspicious"],
        test_frame[spec["pred_col"]],
        labels=[0, 1],
    )
    return pd.DataFrame(
        matrix,
        index=["actual_normal", "actual_suspicious"],
        columns=["predicted_normal", "predicted_suspicious"],
    )


def threshold_tradeoff_table(
    scored_users: pd.DataFrame,
    score_col: str = "random_forest_score",
    label_col: str = "label_suspicious",
) -> pd.DataFrame:
    test_frame = scored_users.loc[scored_users["split"] == "test"].copy()
    thresholds = np.linspace(0.1, 0.9, 17)
    rows: list[dict[str, float]] = []

    for threshold in thresholds:
        y_true = test_frame[label_col]
        y_pred = (test_frame[score_col] >= threshold).astype(int)
        suspicious_streams_total = test_frame.loc[y_true == 1, "total_streams"].sum()
        flagged = y_pred == 1
        suspicious_flagged = y_true.eq(1) & flagged
        legitimate_flagged = y_true.eq(0) & flagged

        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "review_load_pct_accounts": float(flagged.mean()),
                "suspicious_stream_capture": float(
                    test_frame.loc[suspicious_flagged, "total_streams"].sum() / suspicious_streams_total
                ),
                "false_positive_rate_legit_users": float(
                    legitimate_flagged.sum() / max(int(y_true.eq(0).sum()), 1)
                ),
            }
        )

    return pd.DataFrame(rows)


def compile_evaluation_outputs(
    scored_users: pd.DataFrame,
    save_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    metrics = model_metrics_table(scored_users=scored_users)
    business_metrics = business_metrics_table(scored_users=scored_users)
    threshold_tradeoffs = threshold_tradeoff_table(scored_users=scored_users)

    if save_outputs:
        save_dataframe(metrics, METRICS_DIR / "model_metrics.csv")
        save_dataframe(business_metrics, METRICS_DIR / "business_metrics.csv")
        save_dataframe(threshold_tradeoffs, METRICS_DIR / "threshold_tradeoffs.csv")
        for model_name in MODEL_SPECS:
            save_dataframe(
                confusion_matrix_table(scored_users=scored_users, model_name=model_name),
                METRICS_DIR / f"{model_name}_confusion_matrix.csv",
                index=True,
            )

    return {
        "metrics": metrics,
        "business_metrics": business_metrics,
        "threshold_tradeoffs": threshold_tradeoffs,
    }
