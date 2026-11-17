# NYC TLC Yellow Taxi — Phase 1 & Phase 2

پروژه تحلیل داده‌های سفر تاکسی زرد نیویورک (NYC Yellow Taxi) برای مارس ۲۰۲۶. این پروژه شامل ممیزی و تحلیل اکتشافی داده‌ها در فاز اول و مهندسی ویژگی، خوشه‌بندی سفرها و تشخیص ناهنجاری‌ها در فاز دوم است.

## ساختار پروژه

```text
.
├── PROJECT_CARD.md
├── README.md
├── AI_USAGE_FA.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── yellow_tripdata_2026-03.parquet
│   └── external/
│       ├── taxi_zones.shp
│       ├── taxi_zone_lookup.csv
│       └── ...
├── notebooks/
├── src/
│   ├── audit_data.py
│   └── phase2/
│       ├── feature_engineering.py
│       ├── clustering.py
│       └── anomaly_detection.py
└── outputs/
    ├── figures/
    ├── tables/
    ├── reports/
    │   ├── phase_1_report.md
    │   └── phase_2_report.md
    └── phase2/
```

## ۱. تهیه داده‌ها

داده اصلی شامل سفرهای تاکسی زرد نیویورک در مارس ۲۰۲۶ است.

نام فایل:

`yellow_tripdata_2026-03.parquet`

مسیر مورد انتظار:

`data/raw/yellow_tripdata_2026-03.parquet`

برای تحلیل جغرافیایی، فایل shapefile مناطق تاکسی نیویورک و فایل lookup نیز استفاده می‌شوند:

- `data/external/taxi_zones.shp`
- فایل‌های همراه shapefile مانند `.shx`، `.dbf` و `.prj`
- `data/external/taxi_zone_lookup.csv`

اگر فایل‌های داده در نسخه دریافتی پروژه موجود نباشند، باید آن‌ها را جداگانه از منابع رسمی NYC TLC تهیه کرد.

## ۲. نصب وابستگی‌ها

Python 3.11 یا جدیدتر توصیه می‌شود.

```bash
python -m venv .venv
```

فعال‌سازی محیط مجازی در ویندوز:

```cmd
.venv\Scripts\activate
```

نصب کتابخانه‌ها:

```bash
python -m pip install -r requirements.txt
```

## ۳. فاز اول: ممیزی و تحلیل اکتشافی داده‌ها

در فاز اول، ساختار داده، مقادیر گمشده، رکوردهای تکراری و کیفیت داده بررسی شده‌اند. همچنین تحلیل‌های زمانی، جغرافیایی و پیش‌بینی تقاضا انجام شده‌اند.

اجرای ممیزی داده:

```bash
python src/audit_data.py
```

اجرای نوت‌بوک:

```bash
jupyter lab
```

سپس نوت‌بوک تحلیل داده را از پوشه `notebooks/` باز کنید.

گزارش فاز اول:

`outputs/reports/phase_1_report.md`

## ۴. فاز دوم: مهندسی ویژگی‌ها

در این مرحله، ویژگی‌های جدید از داده خام استخراج شده‌اند تا برای مدل‌سازی و تحلیل‌های پیشرفته آماده شوند.

اجرای اسکریپت:

```bash
python src/phase2/feature_engineering.py
```

مهم‌ترین ویژگی‌های ایجادشده عبارت‌اند از:

- مدت زمان سفر بر حسب دقیقه
- ساعت، روز، روز هفته و ماه سفر
- تشخیص روزهای آخر هفته
- ویژگی‌های چرخه‌ای ساعت و روز هفته با توابع سینوس و کسینوس
- پرچم‌های کنترل کیفیت برای مقادیر نامعتبر یا غیرعادی
- شناسایی الگوهای مقادیر گمشده

خروجی اصلی:

`outputs/phase2/yellow_taxi_phase2_features.parquet`

## ۵. خوشه‌بندی سفرها (Clustering)

در این مرحله، سفرها با استفاده از الگوریتم K-Means گروه‌بندی شده‌اند.

روش اجرا:

```bash
python src/phase2/clustering.py
```

مراحل اصلی:

- انتخاب رکوردهای واجد شرایط
- تبدیل لگاریتمی برخی ویژگی‌های عددی
- استانداردسازی ویژگی‌ها
- کاهش ابعاد با PCA
- ارزیابی تعداد خوشه‌ها با معیارهای Silhouette، Davies-Bouldin و Calinski-Harabasz
- تحلیل ویژگی‌های هر خوشه
- بررسی مناطق مبدا، مقصد و مسیرهای پرتکرار
- بررسی پایداری نتایج با نمونه‌گیری‌های مختلف

در اجرای انجام‌شده، تعداد دو خوشه انتخاب شده است. این خوشه‌ها به‌طور کلی سفرهای کوتاه‌تر و سفرهای طولانی‌تر را از هم متمایز می‌کنند. مقدار Silhouette نشان می‌دهد جدایی خوشه‌ها متوسط تا محدود است و نباید آن‌ها را گروه‌های کاملاً مجزا در نظر گرفت.

## ۶. تشخیص ناهنجاری‌ها (Anomaly Detection)

برای شناسایی سفرهای دارای الگوهای غیرمعمول، از الگوریتم Isolation Forest استفاده شده است.

روش اجرا:

```bash
python src/phase2/anomaly_detection.py
```

مراحل اصلی:

- انتخاب سفرهای واجد شرایط
- نمونه‌گیری تصادفی از ۱۵۰ هزار سفر
- استفاده از ویژگی‌های مدت زمان، مسافت، کرایه، مبلغ کل و انعام
- اجرای Isolation Forest با contamination برابر ۰٫۰۲
- مقایسه سفرهای عادی و ناهنجار
- بررسی مناطق مبدا و مقصد سفرهای ناهنجار
- بررسی ارتباط سفرهای ناهنجار با فرودگاه‌های JFK و LaGuardia

در اجرای انجام‌شده، ۳۰۰۰ سفر از نمونه ۱۵۰ هزار رکوردی به‌عنوان ناهنجار شناسایی شده‌اند.

نکته: ناهنجار بودن یک سفر به معنی تقلب یا خطای قطعی نیست؛ این الگوریتم صرفاً الگوهای آماری غیرمعمول را شناسایی می‌کند.

## ۷. خروجی‌ها

### گزارش‌ها

- `outputs/reports/phase_1_report.md`
- `outputs/reports/phase_2_report.md`

### خروجی‌های فاز دوم

پوشه `outputs/phase2/` شامل جداول و داده‌های خروجی مهندسی ویژگی، خوشه‌بندی، تحلیل پایداری و تشخیص ناهنجاری است.

از جمله:

- `yellow_taxi_phase2_features.parquet`
- `clustering_model_selection.csv`
- `cluster_profile_summary.csv`
- `cluster_pickup_zone_summary.csv`
- `cluster_dropoff_zone_summary.csv`
- `cluster_route_summary.csv`
- `clustering_stability_summary.csv`
- `clustering_ari_stability.csv`
- `anomaly_profile_summary.csv`
- `anomaly_pickup_zone_summary.csv`
- `anomaly_dropoff_zone_summary.csv`
- `anomaly_airport_summary.csv`
- `top_1000_anomalies.csv`

### نمودارها

نمودارهای فاز اول و دوم در مسیر زیر ذخیره می‌شوند:

`outputs/figures/`

این نمودارها شامل توزیع ویژگی‌ها، تقاضای زمانی، تحلیل جغرافیایی، نتایج خوشه‌بندی و تحلیل ناهنجاری‌ها هستند.

## ۸. محدودیت‌ها

- داده مورد استفاده مربوط به یک ماه است و نتایج لزوماً به ماه‌های دیگر قابل تعمیم نیستند.
- بخشی از تحلیل‌های فاز دوم بر اساس نمونه تصادفی ۱۵۰ هزار رکوردی انجام شده‌اند؛ بنابراین نتایج نمونه‌گیری باید با این محدودیت تفسیر شوند.
- مقدار Silhouette در خوشه‌بندی نشان‌دهنده جدایی محدود خوشه‌ها است.
- بررسی پایداری خوشه‌بندی بر اساس رکوردهای مشترک نمونه‌ها انجام شده و به‌تنهایی تضمین‌کننده پایداری در کل جمعیت داده نیست.
- مقدار contamination در Isolation Forest از پیش روی ۰٫۰۲ تنظیم شده است؛ بنابراین درصد ناهنجاری‌های شناسایی‌شده به این تنظیم وابسته است.
- ناهنجاری آماری الزاماً به معنی خطای ثبت داده یا رفتار غیرعادی واقعی نیست.
- مقادیر غیرعادی بدون قاعده مستند و قابل دفاع حذف نمی‌شوند.

## ۹. Git و نسخه‌بندی

پس از بررسی فایل‌ها و خروجی‌ها، تغییرات را بررسی کنید:

```bash
git status
```

برای ثبت تغییرات فاز دوم، پس از اطمینان از وجود مخزن Git:

```bash
git add README.md PROJECT_CARD.md AI_USAGE_FA.md requirements.txt src/phase2 outputs/reports/phase_2_report.md outputs/phase2 outputs/figures
```

سپس:

```bash
git commit -m "Complete phase 2 feature engineering clustering and anomaly detection"
```

برای ساخت تگ فاز دوم، ابتدا تگ‌های موجود را بررسی کنید:

```bash
git tag --list
```

اگر تگ `phase-2` از قبل وجود ندارد:

```bash
git tag -a phase-2 -m "NYC TLC Phase 2 deliverables"
```

بررسی تگ:

```bash
git show phase-2 --stat
```

اگر پوشه `.git` در پروژه وجود نداشته باشد، ابتدا باید پروژه به‌صورت یک مخزن Git دریافت یا راه‌اندازی شود.
## Project Deliverables

This project is organized into two main phases.

### Phase 1 – Exploratory Data Analysis and Forecasting

**Report:**
- [Phase 1 Report (Markdown)](outputs/reports/phase_1_report.md)
- [Phase 1 Report (PDF)](outputs/reports/phase_1_report.pdf)

**Visualizations:**

The Phase 1 figures are available in `outputs/figures/`, including:
- Daily and hourly demand analysis
- Weekday and hourly demand patterns
- Passenger count distribution
- Numerical feature distributions
- Correlation heatmaps
- Pickup demand maps
- Origin-destination analysis
- Forecast comparison and forecast errors

### Phase 2 – Feature Engineering, Clustering and Anomaly Detection

**Report:**
- [Phase 2 Report (Markdown)](outputs/reports/phase_2_report.md)

**Source code:**
- `src/phase2/feature_engineering.py`
- `src/phase2/clustering.py`
- `src/phase2/anomaly_detection.py`

**Visualizations:**

Available in `outputs/figures/`:
- PCA explained variance
- Cluster sizes
- Cluster-specific pickup and dropoff zones
- Cluster-specific popular routes
- Anomaly score distribution
- Anomaly duration and distance analysis
- Geographic distribution of anomalies

**Analysis tables:**

Available in `outputs/phase2/`:
- Cluster profiles and model selection metrics
- Clustering stability and ARI results
- Pickup and dropoff zone summaries
- Popular origin-destination routes
- Anomaly profiles and geographic summaries
- Top 1,000 anomalous trips

### Reproducibility and Data Availability

Generated CSV summaries and PNG figures are included in this repository.

Large intermediate Parquet files and raw datasets are excluded from Git tracking to keep the repository manageable. They are generated or obtained locally as described in the project documentation.

To reproduce the Phase 2 analysis, install the dependencies listed in `requirements.txt` and run the scripts from the project root directory.

### Versioned Releases

- Phase 1: `phase-1`
- Phase 2: `phase-2`