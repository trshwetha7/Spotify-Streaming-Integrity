from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils import derive_risk_bucket, explain_risk_row, run_full_pipeline


st.set_page_config(
    page_title="Artificial Streaming Detection",
    page_icon=":notes:",
    layout="wide",
)


@st.cache_data(show_spinner=True)
def load_data(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    return run_full_pipeline(force_refresh=force_refresh)


def format_pct(value: float) -> str:
    return f"{value:.1%}"


def main() -> None:
    st.title("Artificial Streaming Detection for Music Platforms")
    st.caption(
        "Behavioral anomaly detection demo for suspicious streaming patterns such as looping, "
        "coordinated artist boosting, and abnormal session activity."
    )

    with st.sidebar:
        st.header("Controls")
        refresh = st.checkbox("Rebuild synthetic data and models", value=False)
        selected_model = st.selectbox(
            "Primary scoring lens",
            options=["random_forest", "isolation_forest", "logistic_regression"],
            index=0,
        )

    bundle = load_data(force_refresh=refresh)
    events = bundle["events"]
    user_features = bundle["user_features"]
    scored_users = bundle["scored_users"].copy()
    metrics = bundle["metrics"]
    business_metrics = bundle["business_metrics"]
    thresholds = bundle["threshold_tradeoffs"]
    feature_importance = bundle["feature_importance"]

    score_column = f"{selected_model}_score"
    pred_column = f"{selected_model}_pred"
    scored_users["risk_bucket"] = scored_users["risk_score"].apply(derive_risk_bucket)
    scored_users["flagged_primary_model"] = scored_users[pred_column].map({0: "Not flagged", 1: "Flagged"})

    st.subheader("Overview Dashboard")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Users", f"{user_features['user_id'].nunique():,}")
    metric_cols[1].metric("Streams", f"{len(events):,}")
    metric_cols[2].metric(
        "Suspicious label share",
        format_pct(user_features["label_suspicious"].mean()),
    )
    metric_cols[3].metric(
        "Flagged by selected model",
        format_pct(scored_users[pred_column].mean()),
    )
    metric_cols[4].metric(
        "Mean risk score",
        f"{scored_users['risk_score'].mean():.2f}",
    )

    overview_left, overview_right = st.columns(2)

    distribution_frame = user_features.copy()
    distribution_frame["segment"] = distribution_frame["label_suspicious"].map({0: "Normal", 1: "Suspicious"})
    repeat_fig = px.box(
        distribution_frame,
        x="segment",
        y="repeat_rate",
        color="segment",
        title="Repeat Rate by Behavioral Segment",
        points="outliers",
    )
    repeat_fig.update_layout(showlegend=False)
    overview_left.plotly_chart(repeat_fig, use_container_width=True)

    concentration_fig = px.scatter(
        scored_users,
        x="top_artist_share",
        y="repeat_rate",
        color="flagged_primary_model",
        size="total_streams",
        hover_data=["user_id", "label_suspicious", "night_activity_ratio", "shared_ip_count"],
        title="Suspicious Pattern Map",
    )
    overview_right.plotly_chart(concentration_fig, use_container_width=True)

    st.subheader("Account Inspector")
    inspector_left, inspector_right = st.columns([1, 2])
    user_options = scored_users.sort_values("risk_score", ascending=False)["user_id"].tolist()
    selected_user = inspector_left.selectbox("Select account", options=user_options)
    user_row = scored_users.loc[scored_users["user_id"] == selected_user].iloc[0]

    inspector_left.metric("Composite risk score", f"{user_row['risk_score']:.2f}")
    inspector_left.metric("Risk bucket", derive_risk_bucket(float(user_row["risk_score"])))
    inspector_left.metric("Selected model score", f"{user_row[score_column]:.2f}")
    inspector_left.metric("Selected model flag", "Flagged" if int(user_row[pred_column]) else "Not flagged")

    reasons = explain_risk_row(user_row)
    inspector_left.markdown("**Flagging rationale**")
    for reason in reasons:
        inspector_left.write(f"- {reason}")

    profile_fields = [
        "total_streams",
        "total_sessions",
        "repeat_rate",
        "completion_rate",
        "top_artist_share",
        "top_song_share",
        "night_activity_ratio",
        "burstiness_score",
        "shared_device_count",
        "shared_ip_count",
    ]
    profile_table = pd.DataFrame(
        {
            "feature": profile_fields,
            "value": [user_row[field] for field in profile_fields],
        }
    )
    inspector_right.dataframe(profile_table, hide_index=True, use_container_width=True)

    st.subheader("Suspicious Pattern Explorer")
    explorer_left, explorer_right = st.columns(2)

    burst_fig = px.scatter(
        scored_users,
        x="night_activity_ratio",
        y="burstiness_score",
        color="risk_bucket",
        hover_data=["user_id", "repeat_rate", "top_song_share", "shared_ip_count"],
        title="Night Activity vs Burstiness",
    )
    explorer_left.plotly_chart(burst_fig, use_container_width=True)

    top_repeaters = scored_users.sort_values(["repeat_rate", "top_song_share"], ascending=False).head(20)
    explorer_right.dataframe(
        top_repeaters[
            [
                "user_id",
                "risk_score",
                "repeat_rate",
                "top_song_share",
                "top_artist_share",
                "night_activity_ratio",
                "shared_ip_count",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Model Summary")
    summary_left, summary_right = st.columns(2)
    summary_left.dataframe(metrics.round(3), hide_index=True, use_container_width=True)
    summary_right.dataframe(business_metrics.round(3), hide_index=True, use_container_width=True)

    threshold_fig = go.Figure()
    threshold_fig.add_trace(
        go.Scatter(
            x=thresholds["threshold"],
            y=thresholds["precision"],
            mode="lines+markers",
            name="Precision",
        )
    )
    threshold_fig.add_trace(
        go.Scatter(
            x=thresholds["threshold"],
            y=thresholds["recall"],
            mode="lines+markers",
            name="Recall",
        )
    )
    threshold_fig.add_trace(
        go.Scatter(
            x=thresholds["threshold"],
            y=thresholds["review_load_pct_accounts"],
            mode="lines+markers",
            name="Review load",
        )
    )
    threshold_fig.update_layout(title="Threshold Tradeoff for Random Forest")
    st.plotly_chart(threshold_fig, use_container_width=True)

    importance_fig = px.bar(
        feature_importance.head(12).sort_values("permutation_importance"),
        x="permutation_importance",
        y="feature",
        orientation="h",
        title="Top Predictive Features",
    )
    st.plotly_chart(importance_fig, use_container_width=True)


if __name__ == "__main__":
    main()

