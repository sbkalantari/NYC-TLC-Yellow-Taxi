# NYC TLC Yellow Taxi — Phase 1

پروژه تحلیل اکتشافی داده‌های سفر تاکسی زرد نیویورک برای مارس ۲۰۲۶.

## ساختار
```text
.
├── PROJECT_CARD.md
├── README.md
├── requirements.txt
├── reports/
│   ├── phase1.pdf
│   └── phase1.md
├── notebooks/
│   └── 01_data_understanding.ipynb
├── src/
│   └── audit_data.py
├── data/
│   ├── raw/
│   └── external/
└── outputs/
    ├── figures/
    └── tables/
```

## ۱. تهیه داده‌ها
فایل `yellow_tripdata_2026-03.parquet` را از صفحه رسمی NYC TLC Trip Record Data دریافت کنید و در مسیر زیر قرار دهید:
`data/raw/yellow_tripdata_2026-03.parquet`

برای تحلیل جغرافیایی، فایل shapefile مناطق تاکسی نیویورک و فایل lookup را در مسیرهای زیر قرار دهید:
- `data/external/taxi_zones.shp` به همراه فایل‌های همراه آن مانند `.shx`, `.dbf`, `.prj`
- `data/external/taxi_zone_lookup.csv`

داده‌ها در این بسته گنجانده نشده‌اند؛ کاربر باید آن‌ها را جداگانه تهیه کند.

## ۲. نصب وابستگی‌ها
Python 3.11 یا جدیدتر توصیه می‌شود:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## ۳. اجرای ممیزی داده
از ریشه پروژه:
```bash
python src/audit_data.py
```
این دستور داده را می‌خواند، رکوردهای دارای pickup در مارس ۲۰۲۶ را جدا می‌کند و گزارش ساختار، مقادیر گمشده، رکوردهای تکراری و کنترل‌های کیفیت پایه را در `outputs/tables/` ذخیره می‌کند. اسکریپت داده اصلی را تغییر نمی‌دهد و رکوردها را حذف نمی‌کند.

## ۴. اجرای نوت‌بوک کامل
```bash
jupyter lab
```
سپس `notebooks/01_data_understanding.ipynb` را باز کنید. نوت‌بوک انتظار دارد ریشه پروژه به‌عنوان پوشه کاری والد `notebooks/` باشد. برای جلوگیری از خطای مسیر، نوت‌بوک را از مسیر پروژه و با پوشه کاری `notebooks` اجرا کنید.

## ۵. خروجی‌ها
- `reports/phase1.pdf`: گزارش فاز اول
- `reports/phase1.md`: متن قابل ویرایش گزارش
- `outputs/tables/`: جداول ممیزی، تحلیل زمانی، جغرافیایی و پیش‌بینی
- `outputs/figures/`: نمودارهای تولیدشده

## ۶. محدودیت‌ها
- نتایج گزارش بر مبنای نتایج درج‌شده در نسخه گزارش موجود تهیه شده‌اند؛ برای تحویل نهایی، اجرای کامل نوت‌بوک و تطبیق اعداد توصیه می‌شود.
- داده فقط یک ماه را پوشش می‌دهد و ارزیابی پیش‌بینی روی هفت روز پایانی انجام شده است.
- برخی ستون‌ها دارای مقادیر گمشده هم‌الگو هستند؛ علت آن باید با مستندات رسمی داده بررسی شود.
- مقادیر غیرعادی مالی/زمانی بدون قاعده مستند حذف نمی‌شوند.

## ۷. Git tag
پس از بررسی خروجی‌ها و commit نهایی، از ریشه مخزن اجرا کنید:
```bash
git status
git add README.md PROJECT_CARD.md reports/phase1.pdf reports/phase1.md src/audit_data.py notebooks/01_data_understanding.ipynb requirements.txt
git commit -m "Prepare phase 1 NYC TLC analysis deliverables"
git tag -a phase-1 -m "NYC TLC Phase 1 deliverables"
git tag --list
git show phase-1 --stat
```
اگر commit فاز اول از قبل وجود دارد، ابتدا بررسی کنید که تگ `phase-1` از قبل ساخته نشده باشد. در صورت وجود تگ، آن را بدون بررسی حذف یا بازنویسی نکنید.
