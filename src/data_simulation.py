from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils import RAW_DATA_DIR, METRICS_DIR, save_dataframe, save_json


DEVICE_TYPES = ["mobile", "desktop", "smart_speaker", "tablet", "tv"]
PLATFORMS = ["ios", "android", "web", "windows", "macos"]
COUNTRIES = ["US", "GB", "DE", "BR", "IN", "CA", "MX", "FR", "JP", "AU"]
SUBSCRIPTION_TYPES = ["free", "premium", "family", "student"]


@dataclass
class SimulationConfig:
    n_users: int = 2000
    suspicious_share: float = 0.12
    random_state: int = 42
    start_date: str = "2025-01-01"
    n_days: int = 30
    n_artists: int = 650
    min_songs_per_artist: int = 6
    max_songs_per_artist: int = 18


def _build_song_catalog(config: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    song_rows: list[dict[str, object]] = []
    song_counter = 1
    for artist_number in range(1, config.n_artists + 1):
        artist_id = f"artist_{artist_number:04d}"
        song_count = int(rng.integers(config.min_songs_per_artist, config.max_songs_per_artist + 1))
        for _ in range(song_count):
            song_rows.append(
                {
                    "artist_id": artist_id,
                    "song_id": f"song_{song_counter:05d}",
                    "track_length_sec": int(rng.integers(140, 320)),
                }
            )
            song_counter += 1
    return pd.DataFrame(song_rows)


def _weighted_hour(rng: np.random.Generator, suspicious: bool) -> int:
    if suspicious:
        weights = np.array([6, 6, 5, 5, 5, 4, 3, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6])
    else:
        weights = np.array([1, 1, 1, 1, 1, 2, 4, 5, 4, 3, 3, 3, 3, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2])
    probabilities = weights / weights.sum()
    return int(rng.choice(np.arange(24), p=probabilities))


def _generate_user_profile(
    user_id: str,
    suspicious: bool,
    normal_households: np.ndarray,
    suspicious_farms: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, object]:
    if suspicious:
        farm_id = int(rng.choice(suspicious_farms))
        if rng.random() < 0.65:
            ip_address = f"10.0.{farm_id // 255}.{farm_id % 255}"
            device_fingerprint = f"farm_device_{farm_id}_{int(rng.integers(1, 5))}"
        else:
            ip_address = f"100.64.{int(rng.integers(0, 255))}.{int(rng.integers(1, 255))}"
            device_fingerprint = f"device_{user_id}"
        account_age_days = int(rng.integers(3, 900))
        subscription = rng.choice(["free", "premium", "family"], p=[0.55, 0.30, 0.15])
    else:
        household_id = int(rng.choice(normal_households))
        if rng.random() < 0.25:
            ip_address = f"192.168.{household_id // 255}.{household_id % 255}"
        else:
            ip_address = f"172.20.{int(rng.integers(0, 255))}.{int(rng.integers(1, 255))}"
        device_fingerprint = (
            f"household_device_{household_id}_{int(rng.integers(1, 3))}"
            if rng.random() < 0.12
            else f"device_{user_id}"
        )
        account_age_days = int(rng.integers(20, 3200))
        subscription = rng.choice(SUBSCRIPTION_TYPES, p=[0.18, 0.56, 0.16, 0.10])

    return {
        "user_id": user_id,
        "country": rng.choice(COUNTRIES),
        "device_type": rng.choice(DEVICE_TYPES, p=[0.42, 0.22, 0.12, 0.14, 0.10]),
        "platform": rng.choice(PLATFORMS, p=[0.26, 0.28, 0.20, 0.14, 0.12]),
        "subscription_type": subscription,
        "account_age_days": account_age_days,
        "device_fingerprint": device_fingerprint,
        "ip_address": ip_address,
        "behavior_suspicious": int(suspicious),
        "label_suspicious": int(suspicious),
    }


def simulate_streaming_data(
    config: SimulationConfig | None = None,
    save_outputs: bool = True,
) -> pd.DataFrame:
    config = config or SimulationConfig()
    rng = np.random.default_rng(config.random_state)
    date_anchor = pd.Timestamp(config.start_date)

    catalog = _build_song_catalog(config=config, rng=rng)
    artist_to_songs = {
        artist_id: group["song_id"].to_numpy()
        for artist_id, group in catalog.groupby("artist_id", sort=False)
    }
    song_to_length = catalog.set_index("song_id")["track_length_sec"].to_dict()
    song_to_artist_map = catalog.set_index("song_id")["artist_id"].to_dict()
    all_song_ids = catalog["song_id"].to_numpy()
    artist_ids = catalog["artist_id"].drop_duplicates().to_numpy()
    campaign_artists = rng.choice(artist_ids, size=25, replace=False)

    n_suspicious = int(config.n_users * config.suspicious_share)
    user_ids = [f"user_{idx:05d}" for idx in range(1, config.n_users + 1)]
    suspicious_users = set(rng.choice(user_ids, size=n_suspicious, replace=False))
    normal_households = np.arange(1, max(config.n_users // 3, 10))
    suspicious_farms = np.arange(1, max(n_suspicious // 3, 6))

    user_profiles = pd.DataFrame(
        [
            _generate_user_profile(
                user_id=user_id,
                suspicious=user_id in suspicious_users,
                normal_households=normal_households,
                suspicious_farms=suspicious_farms,
                rng=rng,
            )
            for user_id in user_ids
        ]
    )

    noisy_labels = rng.choice(
        user_profiles["user_id"],
        size=max(1, int(config.n_users * 0.06)),
        replace=False,
    )
    noisy_mask = user_profiles["user_id"].isin(noisy_labels)
    user_profiles.loc[noisy_mask, "label_suspicious"] = 1 - user_profiles.loc[noisy_mask, "label_suspicious"]

    shared_device_lookup = user_profiles.groupby("device_fingerprint")["user_id"].transform("count")
    shared_ip_lookup = user_profiles.groupby("ip_address")["user_id"].transform("count")
    user_profiles["shared_device_count"] = shared_device_lookup
    user_profiles["shared_ip_count"] = shared_ip_lookup

    event_rows: list[dict[str, object]] = []
    session_counter = 1

    for profile in user_profiles.itertuples(index=False):
        suspicious = bool(profile.behavior_suspicious)
        stealth_mode = suspicious and (rng.random() < 0.40)
        power_user = (not suspicious) and (rng.random() < 0.25)

        if suspicious:
            if stealth_mode:
                active_days = int(rng.integers(max(config.n_days - 20, 7), config.n_days - 1))
                sessions_per_active_day = rng.integers(2, 5, size=active_days)
                companion_artist_count = int(rng.integers(3, 10))
            else:
                active_days = int(rng.integers(max(config.n_days - 16, 8), config.n_days + 1))
                sessions_per_active_day = rng.integers(2, 7, size=active_days)
                companion_artist_count = int(rng.integers(1, 6))
            focus_artist = str(rng.choice(campaign_artists))
            companion_artists = rng.choice(
                artist_ids[artist_ids != focus_artist],
                size=companion_artist_count,
                replace=False,
            )
            user_artists = np.concatenate(([focus_artist], companion_artists))
            focus_song_pool = artist_to_songs[focus_artist]
            focus_songs = rng.choice(
                focus_song_pool,
                size=min(len(focus_song_pool), int(rng.integers(2, 6 if stealth_mode else 5))),
                replace=False,
            )
        else:
            if power_user:
                active_days = int(rng.integers(max(config.n_days - 12, 10), config.n_days - 1))
                sessions_per_active_day = rng.integers(2, 5, size=active_days)
                artist_pool_size = int(rng.integers(10, 24))
            else:
                active_days = int(rng.integers(max(config.n_days - 22, 5), config.n_days - 4))
                sessions_per_active_day = rng.integers(1, 4, size=active_days)
                artist_pool_size = int(rng.integers(18, 55))
            user_artists = rng.choice(artist_ids, size=artist_pool_size, replace=False)
            focus_artist = str(rng.choice(user_artists))
            focus_songs = rng.choice(
                artist_to_songs[focus_artist],
                size=min(len(artist_to_songs[focus_artist]), int(rng.integers(2, 7))),
                replace=False,
            )

        active_dates = np.sort(
            rng.choice(
                pd.date_range(date_anchor, periods=config.n_days, freq="D"),
                size=active_days,
                replace=False,
            )
        )

        for date_index, active_date in enumerate(active_dates):
            for _ in range(int(sessions_per_active_day[date_index])):
                session_id = f"session_{session_counter:07d}"
                session_counter += 1

                start_hour = _weighted_hour(rng=rng, suspicious=suspicious)
                start_minute = int(rng.integers(0, 60))
                session_start = active_date + pd.Timedelta(hours=start_hour, minutes=start_minute)

                if suspicious:
                    stream_count = int(rng.integers(8, 22)) if stealth_mode else int(rng.integers(10, 30))
                elif power_user:
                    stream_count = int(rng.integers(6, 18))
                else:
                    stream_count = int(rng.integers(3, 14))
                playlist_id = f"playlist_{int(rng.integers(1, 600)):04d}" if rng.random() < 0.82 else None
                last_song_id: str | None = None
                current_ts = session_start

                for _ in range(stream_count):
                    if suspicious:
                        repeat_probability = 0.25 if stealth_mode else 0.46
                        focus_probability = 0.50 if stealth_mode else 0.68
                        if last_song_id and rng.random() < repeat_probability:
                            song_id = last_song_id
                        elif rng.random() < focus_probability:
                            song_id = str(rng.choice(focus_songs))
                        else:
                            chosen_artist = str(rng.choice(user_artists))
                            song_id = str(rng.choice(artist_to_songs[chosen_artist]))
                    else:
                        repeat_probability = 0.28 if power_user else 0.12
                        home_catalog_probability = 0.86 if power_user else 0.68
                        if last_song_id and rng.random() < repeat_probability:
                            song_id = last_song_id
                        elif rng.random() < home_catalog_probability:
                            chosen_artist = str(rng.choice(user_artists))
                            song_id = str(rng.choice(artist_to_songs[chosen_artist]))
                        else:
                            song_id = str(rng.choice(all_song_ids))

                    artist_id = str(song_to_artist_map[song_id])

                    track_length_sec = int(song_to_length[song_id])
                    if suspicious:
                        mean_completion = 0.88 if stealth_mode else 0.94
                        completion_rate = float(np.clip(rng.normal(mean_completion, 0.09), 0.45, 1.0))
                        stream_duration_sec = int(max(20, round(track_length_sec * completion_rate)))
                        skip_flag = int(stream_duration_sec < 0.4 * track_length_sec)
                        gap_after_stream = int(rng.integers(5, 120 if stealth_mode else 45))
                    else:
                        base_completion = 0.86 if power_user else 0.78
                        completion_rate = float(np.clip(rng.normal(base_completion, 0.16), 0.08, 1.0))
                        stream_duration_sec = int(max(8, round(track_length_sec * completion_rate)))
                        skip_flag = int(stream_duration_sec < 0.45 * track_length_sec)
                        gap_after_stream = int(rng.integers(5, 180 if power_user else 240))

                    completed_stream_flag = int(stream_duration_sec >= 0.92 * track_length_sec)
                    repeat_stream_flag = int(song_id == last_song_id)

                    event_rows.append(
                        {
                            "user_id": profile.user_id,
                            "session_id": session_id,
                            "song_id": song_id,
                            "artist_id": artist_id,
                            "playlist_id": playlist_id,
                            "timestamp": current_ts,
                            "device_type": profile.device_type,
                            "platform": profile.platform,
                            "country": profile.country,
                            "account_age_days": profile.account_age_days,
                            "subscription_type": profile.subscription_type,
                            "stream_duration_sec": stream_duration_sec,
                            "track_length_sec": track_length_sec,
                            "completed_stream_flag": completed_stream_flag,
                            "repeat_stream_flag": repeat_stream_flag,
                            "skip_flag": skip_flag,
                            "device_fingerprint": profile.device_fingerprint,
                            "ip_address": profile.ip_address,
                            "shared_device_count": profile.shared_device_count,
                            "shared_ip_count": profile.shared_ip_count,
                            "label_suspicious": profile.label_suspicious,
                        }
                    )

                    last_song_id = song_id
                    current_ts += pd.Timedelta(seconds=stream_duration_sec + gap_after_stream)

    events = pd.DataFrame(event_rows).sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    events["event_date"] = events["timestamp"].dt.date
    events["event_hour"] = events["timestamp"].dt.floor("h")
    events["night_flag"] = events["timestamp"].dt.hour.isin([0, 1, 2, 3, 4, 5]).astype(int)

    daily_counts = (
        events.groupby(["user_id", "event_date"])
        .size()
        .rename("daily_listening_count")
        .reset_index()
    )
    hourly_counts = (
        events.groupby(["user_id", "event_hour"])
        .size()
        .rename("hourly_listening_count")
        .reset_index()
    )
    events = events.merge(daily_counts, on=["user_id", "event_date"], how="left")
    events = events.merge(hourly_counts, on=["user_id", "event_hour"], how="left")

    user_summary = (
        events.groupby("user_id")
        .agg(
            total_streams=("song_id", "size"),
            active_days=("event_date", "nunique"),
            unique_songs_total=("song_id", "nunique"),
            unique_artists_total=("artist_id", "nunique"),
            night_activity_ratio=("night_flag", "mean"),
            same_song_repeat_count=("repeat_stream_flag", "sum"),
        )
        .reset_index()
    )

    artist_concentration = (
        events.groupby(["user_id", "artist_id"])
        .size()
        .groupby(level=0)
        .max()
        .rename("top_artist_plays")
        .reset_index()
    )
    user_summary = user_summary.merge(artist_concentration, on="user_id", how="left")
    user_summary["artist_concentration_score"] = (
        user_summary["top_artist_plays"] / user_summary["total_streams"]
    )
    user_summary["unique_songs_7d"] = (
        user_summary["unique_songs_total"] / user_summary["active_days"].clip(lower=1) * 7
    ).round(2)
    user_summary["unique_artists_7d"] = (
        user_summary["unique_artists_total"] / user_summary["active_days"].clip(lower=1) * 7
    ).round(2)

    session_lengths = (
        events.groupby(["user_id", "session_id"])
        .agg(session_start=("timestamp", "min"), session_end=("timestamp", "max"))
        .reset_index()
    )
    session_lengths["session_length_min"] = (
        (session_lengths["session_end"] - session_lengths["session_start"]).dt.total_seconds() / 60
    ).clip(lower=1)
    avg_session_length = (
        session_lengths.groupby("user_id")["session_length_min"].mean().rename("session_length_min").reset_index()
    )
    sessions_per_day = (
        session_lengths.groupby("user_id")["session_id"].nunique().rename("total_sessions").reset_index()
        .merge(user_summary[["user_id", "active_days"]], on="user_id", how="left")
    )
    sessions_per_day["sessions_per_day"] = (
        sessions_per_day["total_sessions"] / sessions_per_day["active_days"].clip(lower=1)
    )

    user_merge = (
        user_summary.merge(avg_session_length, on="user_id", how="left")
        .merge(sessions_per_day[["user_id", "sessions_per_day"]], on="user_id", how="left")
    )

    events = events.merge(
        user_merge[
            [
                "user_id",
                "unique_songs_7d",
                "unique_artists_7d",
                "artist_concentration_score",
                "same_song_repeat_count",
                "night_activity_ratio",
                "session_length_min",
                "sessions_per_day",
            ]
        ],
        on="user_id",
        how="left",
    )

    final_columns = [
        "user_id",
        "session_id",
        "song_id",
        "artist_id",
        "playlist_id",
        "timestamp",
        "device_type",
        "platform",
        "country",
        "account_age_days",
        "subscription_type",
        "stream_duration_sec",
        "track_length_sec",
        "completed_stream_flag",
        "repeat_stream_flag",
        "skip_flag",
        "hourly_listening_count",
        "daily_listening_count",
        "unique_songs_7d",
        "unique_artists_7d",
        "artist_concentration_score",
        "same_song_repeat_count",
        "night_activity_ratio",
        "session_length_min",
        "sessions_per_day",
        "shared_device_count",
        "shared_ip_count",
        "label_suspicious",
        "device_fingerprint",
        "ip_address",
    ]
    events = events[final_columns]

    if save_outputs:
        save_dataframe(events, RAW_DATA_DIR / "stream_events.csv")
        save_dataframe(user_profiles.drop(columns=["behavior_suspicious"]), RAW_DATA_DIR / "user_profiles.csv")
        save_dataframe(catalog, RAW_DATA_DIR / "song_catalog.csv")
        save_json(
            {
                "n_users": config.n_users,
                "n_suspicious_users": n_suspicious,
                "n_events": len(events),
                "n_sessions": int(events["session_id"].nunique()),
                "n_songs": int(catalog["song_id"].nunique()),
                "suspicious_share": config.suspicious_share,
                "random_state": config.random_state,
            },
            METRICS_DIR / "simulation_summary.json",
        )

    return events
