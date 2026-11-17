# NYC TLC Yellow Taxi — Phase 1 & Phase 2

تحلیل داده‌های سفر تاکسی زرد نیویورک (NYC TLC Yellow Taxi) برای ماه مارس ۲۰۲۶، شامل بررسی کیفیت داده، تحلیل اکتشافی، مهندسی ویژگی، خوشه‌بندی سفرها و شناسایی سفرهای غیرعادی.

This project analyzes NYC TLC Yellow Taxi trip records for March 2026 through data auditing, exploratory analysis, feature engineering, clustering, and anomaly detection.

## Project Structure

```text
NYC-TLC-Yellow-Taxi/
│
├── README.md
├── PROJECT_CARD.md
├── AI_USAGE_FA.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── external/
│
├── notebooks/
│
├── src/
│   ├── audit_data.py
│   └── phase2/
│       ├── feature_engineering.py
│       ├── clustering.py
│       └── anomaly_detection.py
│
└── outputs/
    ├── figures/
    ├── tables/
    ├── reports/
    └── phase2/
```

## Dataset

- **Source:** NYC Taxi & Limousine Commission (TLC)
- **Dataset:** Yellow Taxi Trip Records
- **Period:** March 2026
- **External data:** Taxi zone shapefile and taxi zone lookup table

The raw and external datasets are not included in the Git repository.

## Environment Setup

Recommended Python version: Python 3.11 or newer.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Phase 1 — Data Audit and Exploratory Analysis

Phase 1 focuses on understanding the dataset and assessing its quality.

Main activities:

- Data loading and initial inspection
- Schema and data type checks
- Missing-value analysis
- Duplicate and invalid-record checks
- Descriptive statistics
- Temporal demand analysis
- Geographic and route analysis
- Exploratory visualizations
- Demand forecasting and forecast evaluation

Main script:

```bash
python src/audit_data.py
```

The Phase 1 notebook and report provide additional analysis and interpretation.

## Phase 2 — Advanced Analysis

Phase 2 extends the initial analysis with feature engineering, clustering, and anomaly detection.

### 1. Feature Engineering

Script:

```bash
python src/phase2/feature_engineering.py
```

The feature engineering pipeline creates additional variables, including:

- Trip duration
- Pickup hour, day, weekday, and month
- Weekend indicator
- Cyclical hour and weekday features
- Data-quality indicators
- Missingness profile
- `payment_type_zero` indicator

The `missing_profile` feature identifies records where the selected fields are all missing. The `payment_type_zero` feature identifies records with `payment_type == 0`.

The two indicators were checked against each other in the generated dataset: both identify 945,748 records, with zero mismatches.

Missing values are not automatically treated as errors or indiscriminately imputed. Their handling depends on the analytical purpose.

### 2. Trip Clustering

Script:

```bash
python src/phase2/clustering.py
```

The clustering workflow includes:

- Eligibility filtering
- Feature preprocessing and scaling
- Principal Component Analysis (PCA)
- KMeans clustering
- Model evaluation using Silhouette Score, Davies–Bouldin Index, and Calinski–Harabasz Index
- Cluster profiling
- Pickup and drop-off zone analysis
- Route analysis
- Clustering stability analysis

PCA retained 6 components to explain approximately 95.79% of the variance.

The selected KMeans solution uses 2 clusters, with a Silhouette Score of approximately 0.250. This indicates limited or moderate cluster separation rather than perfectly distinct groups.

The two clusters show different trip patterns:

- **Cluster 0:** Longer trips, greater average distance, and higher average fare and total amount.
- **Cluster 1:** Shorter trips, lower average distance, and lower average fare and total amount.

The 2D PCA visualization displays the first two principal components, which together explain approximately 53.9% of the variance.

Clustering stability was also evaluated across multiple random seeds. Adjusted Rand Index values on the common comparison subsets were approximately 0.98.

### 3. Anomaly Detection

Script:

```bash
python src/phase2/anomaly_detection.py
```

An Isolation Forest model is used to identify statistically unusual trips.

The analysis uses a sample of 150,000 observations and the following features:

- Trip duration
- Trip distance
- Fare amount
- Total amount
- Tip amount

The configured contamination rate is 0.02.

Results:

- Sample size: 150,000 trips
- Normal observations: 147,000
- Anomalies: 3,000
- Anomaly share: 2%

The analysis also examines anomaly profiles and geographic patterns, including airport-related pickup and drop-off zones.

**Important:** An anomaly is a statistically unusual observation, not necessarily a fraudulent, incorrect, or invalid trip. Airport trips and other legitimate travel patterns may naturally have unusual distances, durations, or fares.

## Outputs

Generated outputs are organized into the following directories:

- `outputs/figures/` — Phase 1 and Phase 2 visualizations
- `outputs/tables/` — analytical tables and summaries
- `outputs/reports/` — project reports
- `outputs/phase2/` — Phase 2 feature data, clustering results, anomaly summaries, and related outputs

The Phase 2 report is available at:

```text
outputs/reports/phase_2_report.md
```

Important Phase 2 visualizations include:

- `phase2_cluster_sizes.png`
- `phase2_pca_variance.png`
- `phase2_pca_clusters_2d.png`
- `phase2_anomaly_duration_distance.png`
- `phase2_anomaly_score_distribution.png`
- `phase2_anomaly_top_pickup_zones.png`
- `phase2_anomaly_top_dropoff_zones.png`

## Reproducibility

Run the Phase 2 scripts in this order:

```bash
python src/phase2/feature_engineering.py
python src/phase2/clustering.py
python src/phase2/anomaly_detection.py
```

The feature engineering script must be executed before clustering and anomaly detection.

Make sure the required TLC data files and external geographic reference files are available in the expected directories before running the analysis.

## Limitations

- The analysis covers March 2026 only.
- Results depend on the quality and completeness of the source data.
- Clustering results depend on feature selection, preprocessing, and sampling.
- A Silhouette Score of approximately 0.25 indicates limited separation between clusters.
- Anomaly detection identifies unusual patterns, not confirmed fraud or data errors.
- Geographic patterns should be interpreted in the context of NYC taxi demand and airport travel.

## GitHub Releases

The repository includes separate project milestones:

- `phase-1` — initial data audit and exploratory analysis
- `phase-2` — advanced analysis, clustering, and anomaly detection

The project documentation includes the project card and AI usage statement.

## License and Data Attribution

The source data is provided by the NYC Taxi & Limousine Commission (TLC). Users should follow the applicable TLC data-use terms and attribution requirements.