#!/usr/bin/env python
"""Reproducible structural and data-quality audit for NYC TLC March 2026.

Run from the project root:
    python src/audit_data.py

The script does not alter the source parquet and does not delete records.
"""
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "yellow_tripdata_2026-03.parquet"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
START_DATE = pd.Timestamp("2026-03-01")
END_DATE = pd.Timestamp("2026-04-01")


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print(f"ERROR: input dataset not found: {DATA_PATH}", file=sys.stderr)
        print("Download the NYC TLC March 2026 Yellow Taxi parquet and place it at the path above.", file=sys.stderr)
        return 2

    df = pd.read_parquet(DATA_PATH)
    required = {"tpep_pickup_datetime", "tpep_dropoff_datetime"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        print(f"ERROR: required columns missing: {sorted(missing_columns)}", file=sys.stderr)
        return 3

    # Preserve original data and filter using the same half-open date interval as the notebook.
    pickup = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    mask = pickup.ge(START_DATE) & pickup.lt(END_DATE)
    df_mar = df.loc[mask].copy()
    df_mar["tpep_pickup_datetime"] = pickup.loc[mask]

    structure = pd.DataFrame({
        "column": df.columns,
        "data_type": df.dtypes.astype(str).values,
        "missing_count": df.isna().sum().values,
        "missing_percent": (df.isna().mean() * 100).round(2).values,
        "unique_values": df.nunique(dropna=True).values,
    })
    structure.to_csv(TABLES_DIR / "raw_dataset_structure.csv", index=False, encoding="utf-8-sig")

    overview = pd.DataFrame([
        ("source_file", str(DATA_PATH.relative_to(PROJECT_ROOT))),
        ("analysis_period", "2026-03-01 through 2026-03-31"),
        ("filtering_variable", "tpep_pickup_datetime"),
        ("original_rows", len(df)),
        ("filtered_rows", len(df_mar)),
        ("columns", df_mar.shape[1]),
        ("rows_outside_period_after_filter", int((~mask).sum())),
        ("exact_duplicate_rows_in_filtered_data", int(df_mar.duplicated().sum())),
    ], columns=["item", "value"])
    overview.to_csv(TABLES_DIR / "dataset_overview_march_2026.csv", index=False, encoding="utf-8-sig")

    missing = pd.DataFrame({
        "missing_count": df_mar.isna().sum(),
        "missing_percent": (df_mar.isna().mean() * 100).round(2),
    }).sort_values("missing_percent", ascending=False)
    missing.to_csv(TABLES_DIR / "missing_values_march_2026.csv", encoding="utf-8-sig")

    quality = {}
    if "tpep_dropoff_datetime" in df_mar.columns:
        dropoff = pd.to_datetime(df_mar["tpep_dropoff_datetime"], errors="coerce")
        duration_min = (dropoff - df_mar["tpep_pickup_datetime"]).dt.total_seconds() / 60
        quality["missing_pickup_time"] = int(df_mar["tpep_pickup_datetime"].isna().sum())
        quality["missing_dropoff_time"] = int(dropoff.isna().sum())
        quality["negative_duration"] = int((duration_min < 0).sum())
        quality["zero_duration"] = int((duration_min == 0).sum())
        quality["duration_over_240_minutes"] = int((duration_min > 240).sum())

    for col in ["trip_distance", "fare_amount", "tip_amount", "tolls_amount", "total_amount"]:
        if col in df_mar.columns:
            quality[f"negative_{col}"] = int((df_mar[col] < 0).sum())
            quality[f"zero_{col}"] = int((df_mar[col] == 0).sum())

    pd.DataFrame([{"check": k, "count": v, "percent_of_filtered_rows": round(v / len(df_mar) * 100, 4) if len(df_mar) else 0}
                  for k, v in quality.items()]).to_csv(
        TABLES_DIR / "data_quality_summary_march_2026.csv", index=False, encoding="utf-8-sig"
    )

    print(f"Original shape: {df.shape}")
    print(f"March 2026 shape: {df_mar.shape}")
    print(f"Exact duplicate rows in March data: {int(df_mar.duplicated().sum()):,}")
    print(f"Audit tables written to: {TABLES_DIR}")
    print("No rows were deleted from the source file; anomalies require documented decisions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
