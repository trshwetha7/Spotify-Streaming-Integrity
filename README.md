# Artificial Streaming Detection for Music Platforms

An end-to-end behavioral anomaly detection workflow for a music streaming platform to identify suspicious listening behavior such as repeated song loops, coordinated artist boosting, and abnormal session activity. The repository combines simulation, exploratory analysis, anomaly detection, supervised modeling, and an interactive Streamlit dashboard.

## Project Overview

This repository frames artificial streaming as a platform integrity problem. The goal is to distinguish natural music engagement from suspicious behavior that can distort artist rankings, royalty allocation, recommendation quality, and trust in the product ecosystem.

The repository includes two main deliverables:

1. `notebooks/streaming_integrity_analysis.ipynb` as the primary end-to-end analysis artifact
2. `app/streamlit_app.py` as a lightweight interactive demo for exploring accounts and model outputs

## Business Motivation

Artificial streaming creates several product and trust risks for a music platform:

- It can inflate artist and track performance, distorting discovery surfaces and editorial decisions.
- It can bias growth analytics and content performance reporting.
- It weakens creator trust by rewarding coordinated boosting over genuine listener demand.
- It introduces moderation and operations costs when suspicious activity must be reviewed manually.

The analysis approaches the problem through behavioral anomaly detection and account-level risk scoring, using realistic synthetic event logs when public labeled data is unavailable.

## Dataset

The dataset is fully synthetic but intentionally designed to feel operationally realistic:

- Roughly a few thousand user accounts across a 30-day window
- Stream-level event data with session, device, platform, and geography metadata
- Normal listeners, heavy legitimate listeners, stealthy suspicious accounts, and overt suspicious accounts
- Shared device and IP behavior to simulate coordinated farms and household overlap
- Mild label noise to mimic imperfect moderation labels and avoid toy-perfect separation

Core raw fields include:

- `user_id`, `session_id`, `song_id`, `artist_id`, `playlist_id`, `timestamp`
- `device_type`, `platform`, `country`, `account_age_days`, `subscription_type`
- `stream_duration_sec`, `track_length_sec`, `completed_stream_flag`, `repeat_stream_flag`, `skip_flag`
- `hourly_listening_count`, `daily_listening_count`, `unique_songs_7d`, `unique_artists_7d`
- `artist_concentration_score`, `same_song_repeat_count`, `night_activity_ratio`
- `session_length_min`, `sessions_per_day`, `shared_device_count`, `shared_ip_count`
- `label_suspicious`

## Methodology

### 1. Simulation

`src/data_simulation.py` creates stream-level behavior for:

- Normal listeners with varied devices, session timing, and artist diversity
- Heavy but legitimate listeners who can look suspicious on individual metrics
- Suspicious accounts that loop songs, over-index on one artist, stream overnight, and share infrastructure
- Stealthy suspicious accounts that mimic more natural listening behavior

### 2. Feature Engineering

`src/feature_engineering.py` aggregates event logs into account-level features such as:

- `plays_per_hour`, `repeat_rate`, `completion_rate`, `skip_rate`
- `artist_diversity`, `song_diversity`, `top_artist_share`, `top_song_share`
- `avg_session_length`, `sessions_per_day`, `night_activity_ratio`
- `burstiness_score`, `device_sharing_score`, `ip_sharing_score`
- high-water-mark activity features like `max_daily_streams` and `max_hourly_streams`

### 3. Modeling

`src/modeling.py` trains:

- Isolation Forest for unsupervised anomaly detection
- Random Forest as the primary supervised model
- Logistic Regression as an interpretable baseline

### 4. Evaluation

`src/evaluation.py` produces:

- Precision, recall, F1, ROC-AUC, and PR-AUC
- Confusion matrices
- Threshold tradeoff views
- Business-facing measures such as suspicious-stream capture, review load, and legitimate-user false positive rate

## Notebook and App

### Notebook

The notebook is the main analysis artifact and walks through:

- problem framing
- synthetic data generation
- exploratory analysis
- anomaly hypotheses
- feature engineering
- model training and evaluation
- interpretation and error analysis
- business recommendations and limitations

### Streamlit App

The app provides a compact demo layer with:

- an overview dashboard
- an account inspector with risk drivers
- a suspicious pattern explorer
- a model summary section with metrics, thresholds, and feature importance

## How to Run Locally

```bash
cd spotify_streaming_integrity
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook notebooks/streaming_integrity_analysis.ipynb
```

Run the app:

```bash
streamlit run app/streamlit_app.py
```

The app will load existing processed artifacts if they are present, or regenerate the synthetic data and models on first run.

## Repo Structure

```text
spotify_streaming_integrity/
├── README.md
├── requirements.txt
├── notebooks/
│   └── streaming_integrity_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_simulation.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── utils.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── tables/
└── app/
    └── streamlit_app.py
```

## Key Findings

- Same-song repetition, concentration on a single artist, and overnight activity are among the strongest suspicious signals.
- Shared infrastructure features improve account triage but should not be used in isolation because households can create benign overlap.
- Threshold selection materially changes moderation workload; a tighter threshold can reduce false positives while still capturing most suspicious stream volume.
- Heavy legitimate listeners create important edge cases, which makes feature engineering and review policy design more important than raw model accuracy alone.

## Author
***Shwetha Tinnium Raju***

