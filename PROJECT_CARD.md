\# PROJECT CARD — NYC TLC Yellow Taxi (Phase 1 \& Phase 2)



\## Project Overview



This project analyzes NYC Yellow Taxi trip records for March 2026. It combines data quality assessment, exploratory data analysis, demand forecasting, feature engineering, clustering, and anomaly detection.



The project is organized into two phases.



\## Phase 1 — Data Understanding and Exploratory Analysis



\- Data auditing and quality assessment

\- Missing-value and invalid-record analysis

\- Descriptive statistics and distribution analysis

\- Temporal demand analysis by hour, weekday, and date

\- Geographic pickup and dropoff analysis

\- Origin-destination analysis

\- Naive and Seasonal Naive demand forecasting

\- Forecast evaluation and error analysis



\## Phase 2 — Feature Engineering and Machine Learning



\- Feature engineering and data quality flags

\- Temporal and cyclical feature extraction

\- K-Means clustering

\- PCA dimensionality reduction

\- Cluster profiling and spatial analysis

\- Pickup, dropoff, and popular route analysis

\- Clustering stability evaluation

\- Isolation Forest anomaly detection

\- Geographic analysis of anomalous trips



\## Data Sources



\- NYC TLC Yellow Taxi Trip Records, March 2026

\- NYC TLC Taxi Zone Shapefile

\- NYC TLC Taxi Zone Lookup Table



Expected local data paths:



\- `data/raw/yellow\\\_tripdata\\\_2026-03.parquet`

\- `data/external/taxi\\\_zones.shp`

\- `data/external/taxi\\\_zone\\\_lookup.csv`



Raw datasets and external geographic files are not included in Git tracking. They must be obtained separately from the official NYC TLC sources.



\## Analysis Scope



\- Unit of analysis: Individual taxi trip

\- Main observation period: March 1–31, 2026

\- Pickup datetime is used to define the March observation period.

\- Demand forecasting is based on daily trip counts.



\## Methods and Tools



\- Python

\- pandas and NumPy

\- Matplotlib and Seaborn

\- GeoPandas

\- scikit-learn

\- PCA

\- K-Means

\- Isolation Forest

\- JupyterLab



\## Key Deliverables



\- Phase 1 report: `outputs/reports/phase\\\_1\\\_report.md`

\- Phase 1 PDF report: `outputs/reports/phase\\\_1\\\_report.pdf`

\- Phase 2 report: `outputs/reports/phase\\\_2\\\_report.md`

\- Source code: `src/`

\- Notebooks: `notebooks/`

\- Figures: `outputs/figures/`

\- Phase 2 tables: `outputs/phase2/`



\## Quality, Reproducibility, and Limitations



\- Data quality checks are performed before analysis.

\- Invalid and unusual observations are identified using explicit criteria.

\- Phase 2 clustering and anomaly detection use a random sample of 150,000 eligible trips.

\- Clustering results are evaluated using internal validation metrics and sampling stability checks.

\- The selected clustering solution has moderate separation; clusters should not be interpreted as perfectly distinct groups.

\- Isolation Forest anomalies are statistically unusual observations, not confirmed errors or fraud.

\- Forecasting results are based on a limited observation period and should be interpreted accordingly.

\- Full reproduction requires the original datasets, geographic files, and dependencies listed in `requirements.txt`.



\## Versioning



\- Phase 1 release: `phase-1`

\- Phase 2 release: `phase-2`



\## Project Status



Both project phases have been completed and published as GitHub releases.
