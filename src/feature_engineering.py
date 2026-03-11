from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import PROCESSED_DATA_DIR, save_dataframe


def build_session_features(events: pd.DataFrame) -> pd.DataFrame:
    ordered = events.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"])

    session_features = (
        ordered.groupby(["session_id", "user_id"])
        .agg(
            session_start=("timestamp", "min"),
            session_end=("timestamp", "max"),
            stream_count=("song_id", "size"),
            unique_songs_session=("song_id", "nunique"),
            unique_artists_session=("artist_id", "nunique"),
            session_repeat_rate=("repeat_stream_flag", "mean"),
            session_completion_rate=("completed_stream_flag", "mean"),
            session_skip_rate=("skip_flag", "mean"),
            shared_device_count=("shared_device_count", "max"),
            shared_ip_count=("shared_ip_count", "max"),
            label_suspicious=("label_suspicious", "max"),
        )
        .reset_index()
    )
    session_features["session_length_min"] = (
        (session_features["session_end"] - session_features["session_start"]).dt.total_seconds() / 60
    ).clip(lower=1)
    session_features["streams_per_minute"] = (
        session_features["stream_count"] / session_features["session_length_min"].clip(lower=1)
    )
    return session_features


def build_user_features(events: pd.DataFrame, session_features: pd.DataFrame) -> pd.DataFrame:
    ordered = events.copy().sort_values(["user_id", "timestamp"])
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"])
    ordered["event_date"] = ordered["timestamp"].dt.date
    ordered["event_hour"] = ordered["timestamp"].dt.floor("h")
    ordered["hour_of_day"] = ordered["timestamp"].dt.hour

    ordered["previous_timestamp"] = ordered.groupby("user_id")["timestamp"].shift()
    ordered["inter_stream_gap_sec"] = (
        ordered["timestamp"] - ordered["previous_timestamp"]
    ).dt.total_seconds().fillna(np.nan)

    user_features = (
        ordered.groupby("user_id")
        .agg(
            total_streams=("song_id", "size"),
            total_sessions=("session_id", "nunique"),
            active_days=("event_date", "nunique"),
            unique_songs=("song_id", "nunique"),
            unique_artists=("artist_id", "nunique"),
            unique_playlists=("playlist_id", "nunique"),
            repeat_rate=("repeat_stream_flag", "mean"),
            completion_rate=("completed_stream_flag", "mean"),
            skip_rate=("skip_flag", "mean"),
            night_activity_ratio=("hour_of_day", lambda values: values.isin([0, 1, 2, 3, 4, 5]).mean()),
            avg_stream_duration_sec=("stream_duration_sec", "mean"),
            avg_track_length_sec=("track_length_sec", "mean"),
            avg_inter_stream_gap_sec=("inter_stream_gap_sec", "mean"),
            median_inter_stream_gap_sec=("inter_stream_gap_sec", "median"),
            account_age_days=("account_age_days", "max"),
            shared_device_count=("shared_device_count", "max"),
            shared_ip_count=("shared_ip_count", "max"),
            country=("country", "first"),
            device_type=("device_type", "first"),
            platform=("platform", "first"),
            subscription_type=("subscription_type", "first"),
            label_suspicious=("label_suspicious", "max"),
            unique_songs_7d=("unique_songs_7d", "max"),
            unique_artists_7d=("unique_artists_7d", "max"),
            artist_concentration_score=("artist_concentration_score", "max"),
            daily_listening_count_mean=("daily_listening_count", "mean"),
            hourly_listening_count_mean=("hourly_listening_count", "mean"),
        )
        .reset_index()
    )

    top_artist_share = (
        ordered.groupby(["user_id", "artist_id"]).size().rename("plays").reset_index()
        .sort_values(["user_id", "plays"], ascending=[True, False])
        .drop_duplicates("user_id")
        .rename(columns={"plays": "top_artist_plays", "artist_id": "top_artist_id"})
    )
    top_song_share = (
        ordered.groupby(["user_id", "song_id"]).size().rename("plays").reset_index()
        .sort_values(["user_id", "plays"], ascending=[True, False])
        .drop_duplicates("user_id")
        .rename(columns={"plays": "top_song_plays", "song_id": "top_song_id"})
    )

    max_daily_streams = (
        ordered.groupby(["user_id", "event_date"]).size().rename("daily_streams").reset_index()
        .groupby("user_id")["daily_streams"].max().rename("max_daily_streams").reset_index()
    )
    max_hourly_streams = (
        ordered.groupby(["user_id", "event_hour"]).size().rename("hourly_streams").reset_index()
        .groupby("user_id")["hourly_streams"].max().rename("max_hourly_streams").reset_index()
    )
    session_summary = (
        session_features.groupby("user_id")
        .agg(
            avg_session_length=("session_length_min", "mean"),
            median_session_length=("session_length_min", "median"),
            avg_streams_per_session=("stream_count", "mean"),
            session_length_std=("session_length_min", "std"),
        )
        .reset_index()
    )

    user_features = (
        user_features.merge(top_artist_share[["user_id", "top_artist_id", "top_artist_plays"]], on="user_id", how="left")
        .merge(top_song_share[["user_id", "top_song_id", "top_song_plays"]], on="user_id", how="left")
        .merge(max_daily_streams, on="user_id", how="left")
        .merge(max_hourly_streams, on="user_id", how="left")
        .merge(session_summary, on="user_id", how="left")
    )

    user_features["avg_streams_per_day"] = (
        user_features["total_streams"] / user_features["active_days"].clip(lower=1)
    )
    user_features["plays_per_hour"] = user_features["total_streams"] / (
        user_features["active_days"].clip(lower=1) * 24
    )
    user_features["sessions_per_day"] = (
        user_features["total_sessions"] / user_features["active_days"].clip(lower=1)
    )
    user_features["artist_diversity"] = user_features["unique_artists"] / user_features["total_streams"].clip(lower=1)
    user_features["song_diversity"] = user_features["unique_songs"] / user_features["total_streams"].clip(lower=1)
    user_features["playlist_diversity"] = user_features["unique_playlists"] / user_features["total_sessions"].clip(lower=1)
    user_features["top_artist_share"] = (
        user_features["top_artist_plays"] / user_features["total_streams"].clip(lower=1)
    )
    user_features["top_song_share"] = (
        user_features["top_song_plays"] / user_features["total_streams"].clip(lower=1)
    )
    user_features["same_song_repeat_ratio"] = user_features["repeat_rate"]
    user_features["device_sharing_score"] = np.clip(
        np.log1p(user_features["shared_device_count"]) / np.log(8), 0, 1
    )
    user_features["ip_sharing_score"] = np.clip(
        np.log1p(user_features["shared_ip_count"]) / np.log(12), 0, 1
    )
    user_features["burstiness_score"] = user_features["max_hourly_streams"] / (
        user_features["hourly_listening_count_mean"].clip(lower=1)
    )
    user_features["completion_gap"] = (
        user_features["avg_track_length_sec"] - user_features["avg_stream_duration_sec"]
    )

    numeric_fill_columns = [
        "avg_inter_stream_gap_sec",
        "median_inter_stream_gap_sec",
        "session_length_std",
    ]
    user_features[numeric_fill_columns] = user_features[numeric_fill_columns].fillna(0)

    ordered_columns = [
        "user_id",
        "country",
        "device_type",
        "platform",
        "subscription_type",
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
        "top_artist_id",
        "top_song_id",
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
        "label_suspicious",
    ]
    return user_features[ordered_columns]


def build_feature_tables(
    events: pd.DataFrame,
    save_outputs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    session_features = build_session_features(events=events)
    user_features = build_user_features(events=events, session_features=session_features)

    if save_outputs:
        save_dataframe(session_features, PROCESSED_DATA_DIR / "session_level_features.csv")
        save_dataframe(user_features, PROCESSED_DATA_DIR / "user_level_features.csv")

    return user_features, session_features

