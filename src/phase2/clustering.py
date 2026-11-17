
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)
# --------------------------------------------------
# Phase 2 - PCA and Clustering
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

print("Loading engineered dataset...")

columns = [
    "tpep_pickup_datetime",
    "duration_min",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "tip_amount",
    "PULocationID",
    "DOLocationID",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "missing_profile",
    "payment_type",
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

print(f"Eligible rows: {len(df):,}")

# --------------------------------------------------
# 2. Exclude invalid values for model inputs
# --------------------------------------------------

continuous_features = [
    "duration_min",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "tip_amount",
]

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(
    subset=continuous_features + [
        "pickup_hour",
        "pickup_dayofweek",
        "is_weekend",
    ]
)

df = df[
    (df["fare_amount"] >= 0)
    & (df["total_amount"] >= 0)
    & (df["tip_amount"] >= 0)
]

df = df[
    (df["duration_min"] <= 720)
    & (df["trip_distance"] <= 100)
]

print(f"Rows after model eligibility rules: {len(df):,}")

# --------------------------------------------------
# 3. Random sample
# --------------------------------------------------

sample_n = min(SAMPLE_SIZE, len(df))

sample = df.sample(
    n=sample_n,
    random_state=RANDOM_STATE,
).copy()

print(f"Sample size: {len(sample):,}")

# --------------------------------------------------
# 4. Feature transformations
# --------------------------------------------------

for col in continuous_features:
    sample[f"log_{col}"] = np.log1p(sample[col])

# Cyclical encoding of pickup hour
sample["pickup_hour_sin"] = np.sin(
    2 * np.pi * sample["pickup_hour"] / 24
)

sample["pickup_hour_cos"] = np.cos(
    2 * np.pi * sample["pickup_hour"] / 24
)

# Cyclical encoding of day of week
sample["pickup_day_sin"] = np.sin(
    2 * np.pi * sample["pickup_dayofweek"] / 7
)

sample["pickup_day_cos"] = np.cos(
    2 * np.pi * sample["pickup_dayofweek"] / 7
)

# is_weekend is intentionally excluded from model inputs.
# It will only be used later to interpret the clusters.

features = [
    "log_duration_min",
    "log_trip_distance",
    "log_fare_amount",
    "log_total_amount",
    "log_tip_amount",
    "pickup_hour_sin",
    "pickup_hour_cos",
    "pickup_day_sin",
    "pickup_day_cos",
]

X = sample[features].astype("float32")

# --------------------------------------------------
# 5. Standardization
# --------------------------------------------------

print("Scaling features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# 6. PCA
# --------------------------------------------------

print("Running PCA...")

pca_full = PCA(
    n_components=0.95,
    svd_solver="full",
    random_state=RANDOM_STATE,
)

X_pca = pca_full.fit_transform(X_scaled)

print(
    "PCA components for 95% variance:",
    pca_full.n_components_,
)

print(
    "Explained variance:",
    pca_full.explained_variance_ratio_.sum(),
)

plt.figure(figsize=(8, 5))

cumulative_variance = np.cumsum(
    pca_full.explained_variance_ratio_
)

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o",
)

plt.axhline(
    0.95,
    linestyle="--",
    label="95% variance",
)

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Cumulative Explained Variance")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "phase2_pca_variance.png",
    dpi=150,
)

plt.close()

# --------------------------------------------------
# 7. KMeans model selection
# --------------------------------------------------

print("Evaluating KMeans clusters...")

results = []
best_k = None
best_score = -1

for k in range(2, 9):

    print(f"Fitting KMeans with k={k}...")

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=300,
    )

    labels = model.fit_predict(X_pca)

    silhouette = silhouette_score(
        X_pca,
        labels,
        sample_size=min(10_000, len(X_pca)),
        random_state=RANDOM_STATE,
    )

    db_score = davies_bouldin_score(
        X_pca,
        labels,
    )

    ch_score = calinski_harabasz_score(
        X_pca,
        labels,
    )

    results.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette": silhouette,
        "davies_bouldin": db_score,
        "calinski_harabasz": ch_score,
    })

    if silhouette > best_score:
        best_score = silhouette
        best_k = k

metrics = pd.DataFrame(results)

metrics.to_csv(
    OUTPUT_DIR / "clustering_model_selection.csv",
    index=False,
)

print("\nMODEL SELECTION RESULTS")
print(metrics.to_string(index=False))

print(f"\nBest k by silhouette: {best_k}")

# --------------------------------------------------
# 8. Fit final KMeans
# --------------------------------------------------

final_model = KMeans(
    n_clusters=best_k,
    random_state=RANDOM_STATE,
    n_init=10,
)

sample["cluster"] = final_model.fit_predict(X_pca)

# --------------------------------------------------
# 8.1. Two-dimensional PCA visualization
# --------------------------------------------------

print("Creating 2D PCA cluster visualization...")

# Use a reproducible subset for a clearer and faster plot
plot_sample = sample.sample(
    n=min(20_000, len(sample)),
    random_state=RANDOM_STATE,
)

plot_positions = sample.index.get_indexer(plot_sample.index)

plt.figure(figsize=(10, 7))

for cluster_id in sorted(sample["cluster"].unique()):
    mask = plot_sample["cluster"].to_numpy() == cluster_id

    plt.scatter(
        X_pca[plot_positions[mask], 0],
        X_pca[plot_positions[mask], 1],
        s=8,
        alpha=0.35,
        label=f"Cluster {cluster_id}",
    )

plt.xlabel(
    f"PC1 ({pca_full.explained_variance_ratio_[0] * 100:.1f}% variance)"
)
plt.ylabel(
    f"PC2 ({pca_full.explained_variance_ratio_[1] * 100:.1f}% variance)"
)
plt.title("KMeans Clusters in 2D PCA Space")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "phase2_pca_clusters_2d.png",
    dpi=150,
)

plt.close()

print("2D PCA visualization saved.")


# --------------------------------------------------
# 9. Cluster interpretation
# --------------------------------------------------

summary = sample.groupby("cluster").agg(
    trips=("cluster", "size"),
    avg_duration=("duration_min", "mean"),
    median_duration=("duration_min", "median"),
    avg_distance=("trip_distance", "mean"),
    median_distance=("trip_distance", "median"),
    avg_fare=("fare_amount", "mean"),
    avg_total=("total_amount", "mean"),
    avg_tip=("tip_amount", "mean"),
    avg_pickup_hour=("pickup_hour", "mean"),
    weekend_share=("is_weekend", "mean"),
).reset_index()

summary.to_csv(
    OUTPUT_DIR / "cluster_profile_summary.csv",
    index=False,
)

# --------------------------------------------------
# 10. Save clustered sample
# --------------------------------------------------

sample.to_parquet(
    OUTPUT_DIR / "clustered_trip_sample.parquet",
    index=False,
)

# --------------------------------------------------
# 11. Cluster size plot
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sample["cluster"].value_counts().sort_index().plot(
    kind="bar"
)

plt.xlabel("Cluster")
plt.ylabel("Number of Trips")
plt.title("KMeans Cluster Sizes")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "phase2_cluster_sizes.png",
    dpi=150,
)

plt.close()

# --------------------------------------------------
# 12. Final report
# --------------------------------------------------

print("\nCLUSTER PROFILES")
print(summary.to_string(index=False))

print("\nPCA COMPONENTS:", pca_full.n_components_)
print("BEST K:", best_k)
print("SILHOUETTE:", best_score)

print("\nSaved outputs:")
print(OUTPUT_DIR)
print(FIGURES_DIR)

print("\nPHASE 2 CLUSTERING COMPLETED.")

# --------------------------------------------------
# 13. Spatial Analysis by Cluster
# --------------------------------------------------

print("\nRunning spatial analysis...")

# Load taxi zone lookup table
LOOKUP_FILE = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "taxi_zone_lookup.csv"
)

zones = pd.read_csv(LOOKUP_FILE)

# Count pickup trips by cluster and pickup zone
pickup_zone_summary = (
    sample
    .groupby(["cluster", "PULocationID"])
    .size()
    .reset_index(name="trip_count")
)

# Add zone names and boroughs
pickup_zone_summary = pickup_zone_summary.merge(
    zones,
    left_on="PULocationID",
    right_on="LocationID",
    how="left",
)

# Calculate each zone's share within its cluster
pickup_zone_summary["cluster_total"] = (
    pickup_zone_summary
    .groupby("cluster")["trip_count"]
    .transform("sum")
)

pickup_zone_summary["zone_share_pct"] = (
    100
    * pickup_zone_summary["trip_count"]
    / pickup_zone_summary["cluster_total"]
)

# Sort results
pickup_zone_summary = pickup_zone_summary.sort_values(
    ["cluster", "trip_count"],
    ascending=[True, False],
)

# Save complete spatial summary
pickup_zone_summary.to_csv(
    OUTPUT_DIR / "cluster_pickup_zone_summary.csv",
    index=False,
)

# Save top 10 pickup zones for each cluster
top_zones = (
    pickup_zone_summary
    .groupby("cluster", group_keys=False)
    .head(10)
)

top_zones.to_csv(
    OUTPUT_DIR / "cluster_top10_pickup_zones.csv",
    index=False,
)

print("\nTOP 10 PICKUP ZONES BY CLUSTER")
print(
    top_zones[
        [
            "cluster",
            "PULocationID",
            "Zone",
            "Borough",
            "trip_count",
            "zone_share_pct",
        ]
    ].to_string(index=False)
)

# Plot top 10 pickup zones for each cluster
for cluster_id in sorted(sample["cluster"].unique()):

    plot_data = (
        pickup_zone_summary[
            pickup_zone_summary["cluster"] == cluster_id
        ]
        .head(10)
        .sort_values("trip_count")
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        plot_data["Zone"].fillna(
            plot_data["PULocationID"].astype(str)
        ),
        plot_data["trip_count"],
    )

    plt.xlabel("Number of Trips")
    plt.ylabel("Pickup Zone")
    plt.title(
        f"Top 10 Pickup Zones - Cluster {cluster_id}"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / f"phase2_cluster_{cluster_id}_top_pickup_zones.png",
        dpi=150,
    )

    plt.close()

print("\nSpatial analysis outputs saved.")



# --------------------------------------------------
# 14. Destination Zone Analysis by Cluster
# --------------------------------------------------

print("\nRunning destination zone analysis...")

# Count dropoff trips by cluster and destination zone
dropoff_zone_summary = (
    sample
    .groupby(["cluster", "DOLocationID"])
    .size()
    .reset_index(name="trip_count")
)

# Add zone names and boroughs
dropoff_zone_summary = dropoff_zone_summary.merge(
    zones,
    left_on="DOLocationID",
    right_on="LocationID",
    how="left",
)

# Calculate each zone's share within its cluster
dropoff_zone_summary["cluster_total"] = (
    dropoff_zone_summary
    .groupby("cluster")["trip_count"]
    .transform("sum")
)

dropoff_zone_summary["zone_share_pct"] = (
    100
    * dropoff_zone_summary["trip_count"]
    / dropoff_zone_summary["cluster_total"]
)

# Sort results
dropoff_zone_summary = dropoff_zone_summary.sort_values(
    ["cluster", "trip_count"],
    ascending=[True, False],
)

# Save complete destination summary
dropoff_zone_summary.to_csv(
    OUTPUT_DIR / "cluster_dropoff_zone_summary.csv",
    index=False,
)

# Save top 10 destinations for each cluster
top_destinations = (
    dropoff_zone_summary
    .groupby("cluster", group_keys=False)
    .head(10)
)

top_destinations.to_csv(
    OUTPUT_DIR / "cluster_top10_dropoff_zones.csv",
    index=False,
)

print("\nTOP 10 DROPOFF ZONES BY CLUSTER")
print(
    top_destinations[
        [
            "cluster",
            "DOLocationID",
            "Zone",
            "Borough",
            "trip_count",
            "zone_share_pct",
        ]
    ].to_string(index=False)
)

# Plot top 10 destinations for each cluster
for cluster_id in sorted(sample["cluster"].unique()):

    plot_data = (
        dropoff_zone_summary[
            dropoff_zone_summary["cluster"] == cluster_id
        ]
        .head(10)
        .sort_values("trip_count")
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        plot_data["Zone"].fillna(
            plot_data["DOLocationID"].astype(str)
        ),
        plot_data["trip_count"],
    )

    plt.xlabel("Number of Trips")
    plt.ylabel("Dropoff Zone")
    plt.title(
        f"Top 10 Dropoff Zones - Cluster {cluster_id}"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / f"phase2_cluster_{cluster_id}_top_dropoff_zones.png",
        dpi=150,
    )

    plt.close()

print("\nDestination zone analysis outputs saved.")


# --------------------------------------------------
# 15. Top Origin-Destination Routes by Cluster
# --------------------------------------------------

print("\nRunning origin-destination route analysis...")

# Count trips for each origin-destination pair
route_summary = (
    sample
    .groupby(["cluster", "PULocationID", "DOLocationID"])
    .size()
    .reset_index(name="trip_count")
)

# Add origin zone information
route_summary = route_summary.merge(
    zones[["LocationID", "Zone", "Borough"]].rename(
        columns={
            "LocationID": "PULocationID",
            "Zone": "pickup_zone",
            "Borough": "pickup_borough",
        }
    ),
    on="PULocationID",
    how="left",
)

# Add destination zone information
route_summary = route_summary.merge(
    zones[["LocationID", "Zone", "Borough"]].rename(
        columns={
            "LocationID": "DOLocationID",
            "Zone": "dropoff_zone",
            "Borough": "dropoff_borough",
        }
    ),
    on="DOLocationID",
    how="left",
)

# Calculate each route's share within its cluster
route_summary["cluster_total"] = (
    route_summary
    .groupby("cluster")["trip_count"]
    .transform("sum")
)

route_summary["route_share_pct"] = (
    100
    * route_summary["trip_count"]
    / route_summary["cluster_total"]
)

# Sort routes by frequency
route_summary = route_summary.sort_values(
    ["cluster", "trip_count"],
    ascending=[True, False],
)

# Save all routes
route_summary.to_csv(
    OUTPUT_DIR / "cluster_route_summary.csv",
    index=False,
)

# Select top 20 routes per cluster
top_routes = (
    route_summary
    .groupby("cluster", group_keys=False)
    .head(20)
)

top_routes.to_csv(
    OUTPUT_DIR / "cluster_top20_routes.csv",
    index=False,
)

print("\nTOP 20 ROUTES BY CLUSTER")

print(
    top_routes[
        [
            "cluster",
            "PULocationID",
            "pickup_zone",
            "pickup_borough",
            "DOLocationID",
            "dropoff_zone",
            "dropoff_borough",
            "trip_count",
            "route_share_pct",
        ]
    ].to_string(index=False)
)

# Plot top 10 routes for each cluster
for cluster_id in sorted(sample["cluster"].unique()):

    plot_data = (
        route_summary[
            route_summary["cluster"] == cluster_id
        ]
        .head(10)
        .copy()
    )

    plot_data["route_label"] = (
        plot_data["pickup_zone"].fillna(
            plot_data["PULocationID"].astype(str)
        )
        + " -> "
        + plot_data["dropoff_zone"].fillna(
            plot_data["DOLocationID"].astype(str)
        )
    )

    plot_data = plot_data.sort_values("trip_count")

    plt.figure(figsize=(12, 7))

    plt.barh(
        plot_data["route_label"],
        plot_data["trip_count"],
    )

    plt.xlabel("Number of Trips")
    plt.ylabel("Origin -> Destination")
    plt.title(
        f"Top 10 Routes - Cluster {cluster_id}"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / f"phase2_cluster_{cluster_id}_top_routes.png",
        dpi=150,
    )

    plt.close()

print("\nOrigin-destination route analysis outputs saved.")


# --------------------------------------------------
# 16. Clustering Stability Analysis
# --------------------------------------------------

print("\nRunning clustering stability analysis...")

stability_results = []

# Use different random samples to test cluster stability
stability_seeds = [21, 42, 84]

for seed in stability_seeds:

    print(f"Testing stability with random seed {seed}...")

    stability_sample = df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=seed,
    ).copy()

    # Apply the same feature transformations
    for col in continuous_features:
        stability_sample[f"log_{col}"] = np.log1p(
            stability_sample[col]
        )

    stability_sample["pickup_hour_sin"] = np.sin(
        2 * np.pi * stability_sample["pickup_hour"] / 24
    )

    stability_sample["pickup_hour_cos"] = np.cos(
        2 * np.pi * stability_sample["pickup_hour"] / 24
    )

    stability_sample["pickup_day_sin"] = np.sin(
        2 * np.pi * stability_sample["pickup_dayofweek"] / 7
    )

    stability_sample["pickup_day_cos"] = np.cos(
        2 * np.pi * stability_sample["pickup_dayofweek"] / 7
    )

    X_stability = stability_sample[features].astype("float32")

    X_stability_scaled = scaler.transform(X_stability)

    X_stability_pca = pca_full.transform(X_stability_scaled)

    # Fit a two-cluster KMeans model
    stability_model = KMeans(
        n_clusters=2,
        random_state=seed,
        n_init=10,
        max_iter=300,
    )

    stability_labels = stability_model.fit_predict(
        X_stability_pca
    )

    stability_silhouette = silhouette_score(
        X_stability_pca,
        stability_labels,
        sample_size=min(10_000, len(X_stability_pca)),
        random_state=RANDOM_STATE,
    )

    cluster_counts = pd.Series(
        stability_labels
    ).value_counts(normalize=True).sort_index()

    stability_results.append({
        "random_seed": seed,
        "sample_size": len(stability_sample),
        "silhouette": stability_silhouette,
        "cluster_0_share": cluster_counts.get(0, 0),
        "cluster_1_share": cluster_counts.get(1, 0),
    })

stability_df = pd.DataFrame(stability_results)

stability_df.to_csv(
    OUTPUT_DIR / "clustering_stability_summary.csv",
    index=False,
)

print("\nCLUSTERING STABILITY RESULTS")
print(stability_df.to_string(index=False))

print("\nStability results saved.")


# --------------------------------------------------
# 17. Adjusted Rand Index (ARI) Stability Analysis
# --------------------------------------------------

print("\nRunning ARI stability analysis...")

# Use the original sample as the reference
reference_sample = sample.copy()

reference_features = reference_sample[features].astype("float32")

reference_scaled = scaler.transform(reference_features)
reference_pca = pca_full.transform(reference_scaled)

reference_model = KMeans(
    n_clusters=2,
    random_state=RANDOM_STATE,
    n_init=10,
    max_iter=300,
)

reference_labels = reference_model.fit_predict(reference_pca)

ari_results = []

for seed in [21, 84]:

    print(f"Comparing reference sample with seed {seed}...")

    comparison_sample = df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=seed,
    ).copy()

    for col in continuous_features:
        comparison_sample[f"log_{col}"] = np.log1p(
            comparison_sample[col]
        )

    comparison_sample["pickup_hour_sin"] = np.sin(
        2 * np.pi * comparison_sample["pickup_hour"] / 24
    )

    comparison_sample["pickup_hour_cos"] = np.cos(
        2 * np.pi * comparison_sample["pickup_hour"] / 24
    )

    comparison_sample["pickup_day_sin"] = np.sin(
        2 * np.pi * comparison_sample["pickup_dayofweek"] / 7
    )

    comparison_sample["pickup_day_cos"] = np.cos(
        2 * np.pi * comparison_sample["pickup_dayofweek"] / 7
    )

    comparison_features = comparison_sample[
        features
    ].astype("float32")

    comparison_scaled = scaler.transform(
        comparison_features
    )

    comparison_pca = pca_full.transform(
        comparison_scaled
    )

    comparison_model = KMeans(
        n_clusters=2,
        random_state=seed,
        n_init=10,
        max_iter=300,
    )

    comparison_labels = comparison_model.fit_predict(
        comparison_pca
    )

    # Match observations using their original row index
    common_indices = reference_sample.index.intersection(
        comparison_sample.index
    )

    reference_positions = reference_sample.index.get_indexer(
        common_indices
    )

    comparison_positions = comparison_sample.index.get_indexer(
        common_indices
    )

    ari = adjusted_rand_score(
        reference_labels[reference_positions],
        comparison_labels[comparison_positions],
    )

    ari_results.append({
        "comparison_seed": seed,
        "common_trips": len(common_indices),
        "adjusted_rand_index": ari,
    })

ari_df = pd.DataFrame(ari_results)

ari_df.to_csv(
    OUTPUT_DIR / "clustering_ari_stability.csv",
    index=False,
)

print("\nARI STABILITY RESULTS")
print(ari_df.to_string(index=False))

print("\nARI stability results saved.")
