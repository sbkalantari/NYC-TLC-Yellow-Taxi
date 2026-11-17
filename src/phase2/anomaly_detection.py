
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest

# --------------------------------------------------
# Phase 2 - Anomaly Detection
# NYC TLC Yellow Taxi - March 2026
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "phase2"
    / "yellow_taxi_phase2_features.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase2"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 150_000
RANDOM_STATE = 42
CONTAMINATION = 0.02

print("Loading engineered dataset...")

columns = [
    "duration_min",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "tip_amount",
    "PULocationID",
    "DOLocationID",
    "pickup_hour",
    "is_weekend",
    "eligible_march_pickup",
    "eligible_positive_duration",
]

df = pd.read_parquet(INPUT_FILE, columns=columns)

print(f"Loaded rows: {len(df):,}")

# --------------------------------------------------
# 1. Define analytical population
# --------------------------------------------------

df = df[
    (df["eligible_march_pickup"])
    & (df["eligible_positive_duration"])
    & (df["trip_distance"] > 0)
].copy()

# Keep plausible nonnegative financial values
df = df[
    (df["fare_amount"] >= 0)
    & (df["total_amount"] >= 0)
    & (df["tip_amount"] >= 0)
    & (df["duration_min"] <= 720)
    & (df["trip_distance"] <= 100)
].copy()

print(f"Eligible rows: {len(df):,}")

# --------------------------------------------------
# 2. Random sample
# --------------------------------------------------

sample = df.sample(
    n=min(SAMPLE_SIZE, len(df)),
    random_state=RANDOM_STATE,
).copy()

print(f"Sample size: {len(sample):,}")

# --------------------------------------------------
# 3. Prepare model features
# --------------------------------------------------

continuous_features = [
    "duration_min",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "tip_amount",
]

for col in continuous_features:
    sample[f"log_{col}"] = np.log1p(sample[col])

features = [
    "log_duration_min",
    "log_trip_distance",
    "log_fare_amount",
    "log_total_amount",
    "log_tip_amount",
]

X = sample[features].replace(
    [np.inf, -np.inf],
    np.nan,
)

valid_mask = X.notna().all(axis=1)

sample = sample.loc[valid_mask].copy()
X = X.loc[valid_mask].astype("float32")

print(f"Valid model observations: {len(sample):,}")

# --------------------------------------------------
# 4. Fit Isolation Forest
# --------------------------------------------------

print("Fitting Isolation Forest...")

model = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

sample["anomaly_prediction"] = model.fit_predict(X)

# -1 = anomaly
#  1 = normal

sample["is_anomaly"] = (
    sample["anomaly_prediction"] == -1
)

sample["anomaly_score"] = model.decision_function(X)

# --------------------------------------------------
# 5. Anomaly summary
# --------------------------------------------------

anomaly_count = int(sample["is_anomaly"].sum())
normal_count = int((~sample["is_anomaly"]).sum())

anomaly_share = (
    100 * anomaly_count / len(sample)
)

print("\nANOMALY SUMMARY")
print(f"Total observations: {len(sample):,}")
print(f"Normal observations: {normal_count:,}")
print(f"Anomalies: {anomaly_count:,}")
print(f"Anomaly percentage: {anomaly_share:.2f}%")

# --------------------------------------------------
# 6. Compare normal trips and anomalies
# --------------------------------------------------

profile = sample.groupby("is_anomaly").agg(
    trips=("is_anomaly", "size"),
    avg_duration=("duration_min", "mean"),
    median_duration=("duration_min", "median"),
    avg_distance=("trip_distance", "mean"),
    median_distance=("trip_distance", "median"),
    avg_fare=("fare_amount", "mean"),
    avg_total=("total_amount", "mean"),
    avg_tip=("tip_amount", "mean"),
).reset_index()

profile["group"] = profile["is_anomaly"].map({
    False: "Normal",
    True: "Anomaly",
})

profile.to_csv(
    OUTPUT_DIR / "anomaly_profile_summary.csv",
    index=False,
)

print("\nNORMAL VS ANOMALY PROFILE")
print(profile.to_string(index=False))

# --------------------------------------------------
# 7. Save anomaly records
# --------------------------------------------------

anomalies = sample[
    sample["is_anomaly"]
].sort_values("anomaly_score")

anomalies.to_parquet(
    OUTPUT_DIR / "anomaly_trip_sample.parquet",
    index=False,
)

anomalies.head(1000).to_csv(
    OUTPUT_DIR / "top_1000_anomalies.csv",
    index=False,
)

# --------------------------------------------------
# 8. Anomaly score distribution
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.hist(
    sample["anomaly_score"],
    bins=60,
)

plt.xlabel("Isolation Forest Anomaly Score")
plt.ylabel("Number of Trips")
plt.title("Distribution of Anomaly Scores")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "phase2_anomaly_score_distribution.png",
    dpi=150,
)

plt.close()

# --------------------------------------------------
# 9. Duration vs Distance
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    sample.loc[~sample["is_anomaly"], "trip_distance"],
    sample.loc[~sample["is_anomaly"], "duration_min"],
    s=3,
    alpha=0.15,
    label="Normal",
)

plt.scatter(
    anomalies["trip_distance"],
    anomalies["duration_min"],
    s=5,
    alpha=0.5,
    label="Anomaly",
)

plt.xlabel("Trip Distance (miles)")
plt.ylabel("Trip Duration (minutes)")
plt.title("Anomaly Detection: Duration vs Distance")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "phase2_anomaly_duration_distance.png",
    dpi=150,
)

plt.close()

# --------------------------------------------------
# 10. Final report
# --------------------------------------------------

print("\nSaved outputs:")
print(OUTPUT_DIR)
print(FIGURES_DIR)

print("\nANOMALY DETECTION COMPLETED.")


# --------------------------------------------------
# 11. Geographic Analysis of Anomalies
# --------------------------------------------------

print("\nRunning geographic anomaly analysis...")

# Load detected anomaly records
ANOMALY_FILE = OUTPUT_DIR / "anomaly_trip_sample.parquet"

anomaly_data = pd.read_parquet(ANOMALY_FILE)

# Load taxi zone lookup
LOOKUP_FILE = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "taxi_zone_lookup.csv"
)

zones = pd.read_csv(LOOKUP_FILE)

print(f"Anomaly records loaded: {len(anomaly_data):,}")

# --------------------------------------------------
# 1. Pickup zone analysis
# --------------------------------------------------

pickup_summary = (
    anomaly_data
    .groupby("PULocationID")
    .size()
    .reset_index(name="anomaly_trips")
)

pickup_summary = pickup_summary.merge(
    zones,
    left_on="PULocationID",
    right_on="LocationID",
    how="left",
)

pickup_summary["share_pct"] = (
    100 * pickup_summary["anomaly_trips"]
    / len(anomaly_data)
)

pickup_summary = pickup_summary.sort_values(
    "anomaly_trips",
    ascending=False,
)

pickup_summary.to_csv(
    OUTPUT_DIR / "anomaly_pickup_zone_summary.csv",
    index=False,
)

print("\nTOP 10 ANOMALY PICKUP ZONES")

print(
    pickup_summary[
        [
            "PULocationID",
            "Zone",
            "Borough",
            "anomaly_trips",
            "share_pct",
        ]
    ].head(10).to_string(index=False)
)

# --------------------------------------------------
# 2. Dropoff zone analysis
# --------------------------------------------------

dropoff_summary = (
    anomaly_data
    .groupby("DOLocationID")
    .size()
    .reset_index(name="anomaly_trips")
)

dropoff_summary = dropoff_summary.merge(
    zones,
    left_on="DOLocationID",
    right_on="LocationID",
    how="left",
)

dropoff_summary["share_pct"] = (
    100 * dropoff_summary["anomaly_trips"]
    / len(anomaly_data)
)

dropoff_summary = dropoff_summary.sort_values(
    "anomaly_trips",
    ascending=False,
)

dropoff_summary.to_csv(
    OUTPUT_DIR / "anomaly_dropoff_zone_summary.csv",
    index=False,
)

print("\nTOP 10 ANOMALY DROPOFF ZONES")

print(
    dropoff_summary[
        [
            "DOLocationID",
            "Zone",
            "Borough",
            "anomaly_trips",
            "share_pct",
        ]
    ].head(10).to_string(index=False)
)

# --------------------------------------------------
# 3. Airport-related anomaly analysis
# --------------------------------------------------

# NYC TLC airport zone IDs:
# 132 = JFK Airport
# 138 = LaGuardia Airport

airport_ids = [132, 138]

jfk_pickup = (
    anomaly_data["PULocationID"] == 132
).sum()

laguardia_pickup = (
    anomaly_data["PULocationID"] == 138
).sum()

jfk_dropoff = (
    anomaly_data["DOLocationID"] == 132
).sum()

laguardia_dropoff = (
    anomaly_data["DOLocationID"] == 138
).sum()

airport_related = (
    anomaly_data["PULocationID"].isin(airport_ids)
    | anomaly_data["DOLocationID"].isin(airport_ids)
)

airport_related_count = int(airport_related.sum())

airport_summary = pd.DataFrame([
    {
        "airport": "JFK Airport",
        "pickup_trips": int(jfk_pickup),
        "dropoff_trips": int(jfk_dropoff),
    },
    {
        "airport": "LaGuardia Airport",
        "pickup_trips": int(laguardia_pickup),
        "dropoff_trips": int(laguardia_dropoff),
    },
])

airport_summary.to_csv(
    OUTPUT_DIR / "anomaly_airport_summary.csv",
    index=False,
)

print("\nAIRPORT ANOMALY SUMMARY")
print(airport_summary.to_string(index=False))

print(
    "\nAnomalies with pickup OR dropoff at JFK/LaGuardia:",
    f"{airport_related_count:,}",
)

print(
    "Airport-related anomaly share:",
    f"{100 * airport_related_count / len(anomaly_data):.2f}%"
)

# --------------------------------------------------
# 4. Plot top pickup and dropoff zones
# --------------------------------------------------

for zone_data, zone_column, filename, title in [
    (
        pickup_summary,
        "Zone",
        "phase2_anomaly_top_pickup_zones.png",
        "Top 10 Anomaly Pickup Zones",
    ),
    (
        dropoff_summary,
        "Zone",
        "phase2_anomaly_top_dropoff_zones.png",
        "Top 10 Anomaly Dropoff Zones",
    ),
]:

    plot_data = (
        zone_data
        .head(10)
        .sort_values("anomaly_trips")
        .copy()
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        plot_data[zone_column].fillna("Unknown"),
        plot_data["anomaly_trips"],
    )

    plt.xlabel("Number of Anomalous Trips")
    plt.ylabel("Taxi Zone")
    plt.title(title)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / filename,
        dpi=150,
    )

    plt.close()

print("\nGeographic anomaly outputs saved.")
print("\nGEOGRAPHIC ANOMALY ANALYSIS COMPLETED.")
