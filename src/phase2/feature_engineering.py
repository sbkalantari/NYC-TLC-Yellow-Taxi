
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

# --------------------------------------------------
# Phase 2 - Feature Engineering
# NYC TLC Yellow Taxi - March 2026
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "yellow_tripdata_2026-03.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "yellow_taxi_phase2_features.parquet"

# Columns required for Phase 2
columns = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "Airport_fee",
    "cbd_congestion_fee",
]

print("Reading raw Parquet file...")

df = pd.read_parquet(RAW_FILE, columns=columns)

print(f"Raw rows: {len(df):,}")

# --------------------------------------------------
# 1. Datetime features
# --------------------------------------------------

pickup = df["tpep_pickup_datetime"]
dropoff = df["tpep_dropoff_datetime"]

df["duration_min"] = (
    dropoff - pickup
).dt.total_seconds() / 60

df["pickup_hour"] = pickup.dt.hour
df["pickup_day"] = pickup.dt.day
df["pickup_dayofweek"] = pickup.dt.day_of_week
df["pickup_month"] = pickup.dt.month
df["pickup_date"] = pickup.dt.date

df["is_weekend"] = (
    df["pickup_dayofweek"] >= 5
).astype("int8")

# Cyclical time features
df["pickup_hour_sin"] = __import__("numpy").sin(
    2 * __import__("numpy").pi * df["pickup_hour"] / 24
)

df["pickup_hour_cos"] = __import__("numpy").cos(
    2 * __import__("numpy").pi * df["pickup_hour"] / 24
)


# --------------------------------------------------
# 2. Missingness profile
# --------------------------------------------------

# These five columns were found to be missing together
# in records where payment_type == 0.
#
# We preserve this missingness pattern instead of applying
# blanket mean/median imputation, because the missing values
# may represent a distinct record type rather than random gaps.

missing_columns = [
    "passenger_count",
    "RatecodeID",
    "store_and_fwd_flag",
    "congestion_surcharge",
    "Airport_fee",
]

df["missing_profile"] = (
    df[missing_columns]
    .isna()
    .all(axis=1)
    .astype("int8")
)

# Keep an explicit indicator for the associated payment type.
df["payment_type_zero"] = (
    df["payment_type"] == 0
).astype("int8")

# --------------------------------------------------
# 2.1 Missingness Profile Validation
# --------------------------------------------------

missing_profile_count = int(df["missing_profile"].sum())

payment_type_zero_count = int(df["payment_type_zero"].sum())

mismatch_count = int(
    (df["missing_profile"] != df["payment_type_zero"]).sum()
)

validation_passed = mismatch_count == 0

validation_summary = pd.DataFrame([{
    "total_records": len(df),
    "missing_profile_count": missing_profile_count,
    "payment_type_zero_count": payment_type_zero_count,
    "mismatch_count": mismatch_count,
    "validation_passed": validation_passed,
}])

VALIDATION_FILE = OUTPUT_DIR / "missingness_validation.csv"

validation_summary.to_csv(
    VALIDATION_FILE,
    index=False,
)

print("\nMISSINGNESS VALIDATION")
print(validation_summary.to_string(index=False))

print(f"Validation report saved to: {VALIDATION_FILE}")

if not validation_passed:
    raise ValueError(
        "Missingness validation failed! "
        f"Found {mismatch_count:,} mismatched records."
    )

print("Missingness validation passed.")


# --------------------------------------------------
# 3. Data quality flags
# --------------------------------------------------

df["flag_pickup_outside_march"] = (
    (pickup < pd.Timestamp("2026-03-01"))
    | (pickup >= pd.Timestamp("2026-04-01"))
)

df["flag_dropoff_outside_analysis_window"] = (
    (dropoff < pd.Timestamp("2026-03-01"))
    | (dropoff >= pd.Timestamp("2026-04-02"))
)

df["flag_negative_duration"] = df["duration_min"] < 0
df["flag_zero_duration"] = df["duration_min"] == 0
df["flag_long_duration_12h"] = df["duration_min"] > 720

df["flag_zero_distance"] = df["trip_distance"] == 0
df["flag_extreme_distance_100mi"] = df["trip_distance"] > 100

df["flag_negative_fare"] = df["fare_amount"] < 0
df["flag_zero_fare"] = df["fare_amount"] == 0
df["flag_negative_total"] = df["total_amount"] < 0
df["flag_zero_total"] = df["total_amount"] == 0

# --------------------------------------------------
# 4. Analysis eligibility flags
# --------------------------------------------------

df["eligible_march_pickup"] = (
    (pickup >= pd.Timestamp("2026-03-01"))
    & (pickup < pd.Timestamp("2026-04-01"))
)

df["eligible_positive_duration"] = (
    df["duration_min"] > 0
)

df["eligible_basic_trip_analysis"] = (
    df["eligible_march_pickup"]
    & df["eligible_positive_duration"]
)

# --------------------------------------------------
# 5. Save engineered dataset
# --------------------------------------------------

print("Saving engineered dataset...")

df.to_parquet(
    OUTPUT_FILE,
    index=False,
    compression="snappy"
)

# --------------------------------------------------
# 6. Summary
# --------------------------------------------------

print("\nPHASE 2 FEATURE ENGINEERING SUMMARY")
print("-----------------------------------")

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")

print(
    "March pickup eligible:",
    f"{df['eligible_march_pickup'].sum():,}"
)

print(
    "Negative duration:",
    f"{df['flag_negative_duration'].sum():,}"
)

print(
    "Zero duration:",
    f"{df['flag_zero_duration'].sum():,}"
)

print(
    "Long duration over 12h:",
    f"{df['flag_long_duration_12h'].sum():,}"
)

print(
    "Missing profile:",
    f"{df['missing_profile'].sum():,}"
)

print("\nSaved to:")
print(OUTPUT_FILE)
