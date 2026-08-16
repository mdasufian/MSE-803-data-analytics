# MSE-803 Week 3 — Activity 1: Basic Data Analytics

Simple data cleaning, missing-value preparation, and descriptive analysis on `Sample_dataset.csv`.

## Files

| File | Description |
|------|-------------|
| `Sample_dataset.csv` | Original messy dataset |
| `Sample_dataset_cleaned.csv` | Cleaned & imputed dataset (ready for analysis) |
| `basic_data_analytics.py` | Python program that cleans, fills missing values, and reports results |

## How to run

```bash
cd week-3/activity-1
python3 basic_data_analytics.py
```

**Requirements:** Python 3 with `pandas` and `numpy`.

---

## 1. Data cleaning

Problems found in the raw file and how they were fixed:

| Problem | Example | Fix |
|---------|---------|-----|
| Commas in numbers | `"30,000"` | Converted to `30000` |
| Word numbers | `"thirty-eight"`, `"sixty five thousand"` | Converted to `38`, `65000` |
| Inconsistent country codes | `AU` vs `AUS` | Standardised to `AUS` |
| Invalid date | `2019-13-01` | Treated as missing |
| Duplicate person | Bob appears twice | Merged into one row |
| Blank / invalid fields | empty Age, Salary as text | Marked as missing, then filled |

**Result after cleaning:** 10 raw rows → **9 rows** (Bob’s duplicates merged).

---

## 2. Preparing missing data (imputation)

Simple classroom rules used to fill blanks:

| Column | Method | Why | Value used |
|--------|--------|-----|------------|
| ID | Next available ID | Keep a unique identifier | `11` (for Eve) |
| Name | Placeholder | Keep the row usable | `"Unknown"` |
| Age | **Median** | Robust to outliers | `29.5` |
| Net worth | **Median** | Robust to outliers | `35,000` |
| Salary | **Median** | Robust to outliers | `62,000` |
| Country | **Mode** (most common) | Best for categories | `NZ` |
| Join Date | **Median date** | Simple date fill | `15/01/2020` |

After imputation: **0 missing values** (100% complete).

### Cleaned dataset

| ID | Name | Age | Net worth | Country | Salary | Join Date |
|----|------|-----|-----------|---------|--------|-----------|
| 1 | Alice | 25.0 | 30,000 | NZ | 55,000 | 15/01/2020 |
| 2 | Bob | 30.0 | 35,000 | NZ | 60,000 | 20/02/2020 |
| 4 | Charlie | 35.0 | 40,000 | AUS | 72,000 | 15/01/2020 |
| 5 | David | 38.0 | 35,000 | NZ | 68,000 | 01/11/2019 |
| 11 | Eve | 29.0 | 22,000 | AUS | 59,000 | 15/01/2020 |
| 7 | Unknown | 40.0 | 55,000 | NZ | 65,000 | 30/05/2018 |
| 8 | Grace | 22.0 | 28,000 | NZ | 64,000 | 25/07/2021 |
| 9 | Heidi | 29.5 | 35,000 | AUS | 62,000 | 25/07/2021 |
| 10 | Ivan | 27.0 | 60,000 | NZ | 58,000 | 15/03/2019 |

---

## 3. Metrics used (what they mean)

| Metric | What it measures | How to interpret |
|--------|------------------|------------------|
| **Count** | Number of values used | After imputation, should equal number of rows |
| **Mean** | Average value | Good overall centre; extreme values can pull it |
| **Median** | Middle value when sorted | Better “typical” value when outliers exist |
| **Mode** | Most common value | Useful for categories; less useful if every number is unique |
| **Min / Max** | Smallest and largest values | Shows extremes |
| **Range** | Max − Min | Easy spread measure; sensitive to outliers |
| **Std Dev** | How far values sit from the mean | Low = consistent; high = widely spread |
| **Variance** | Average squared distance from the mean | Same idea as std dev, but harder to read |
| **Q1 (25%)** | 25% of values are below this | Lower quartile |
| **Q3 (75%)** | 75% of values are below this | Upper quartile |
| **IQR** | Q3 − Q1 | Spread of the middle 50%; robust to outliers |
| **Frequency / %** | Counts and shares by group | Shows which category appears most |
| **Correlation (r)** | Linear link between two numbers (−1 to +1) | Near 0 = weak link; **correlation ≠ causation** |

---

## 4. Key analytical results

### Numeric summary

| Metric | Age | Net worth | Salary |
|--------|-----|-----------|--------|
| Count | 9 | 9 | 9 |
| Mean | 30.61 | 37,777.78 | 62,555.56 |
| Median | 29.50 | 35,000.00 | 62,000.00 |
| Mode | 22.00 | 35,000.00 | 55,000.00 |
| Min | 22.00 | 22,000.00 | 55,000.00 |
| Max | 40.00 | 60,000.00 | 72,000.00 |
| Range | 18.00 | 38,000.00 | 17,000.00 |
| Std Dev | 5.97 | 12,367.07 | 5,294.13 |
| Variance | 35.61 | 152,944,444.44 | 28,027,777.78 |
| Q1 (25%) | 27.00 | 30,000.00 | 59,000.00 |
| Q3 (75%) | 35.00 | 40,000.00 | 65,000.00 |
| IQR | 8.00 | 10,000.00 | 6,000.00 |

### Country distribution

| Country | Count | Percentage |
|---------|-------|------------|
| NZ | 6 | 66.7% |
| AUS | 3 | 33.3% |

### Average salary by country

| Country | Count | Mean salary | Median salary | Std Dev |
|---------|-------|-------------|---------------|---------|
| AUS | 3 | 64,333.33 | 62,000.00 | 6,806.86 |
| NZ | 6 | 61,666.67 | 62,000.00 | 4,844.24 |

### Correlation

|  | Age | Net worth | Salary |
|--|-----|-----------|--------|
| **Age** | 1.000 | 0.387 | 0.629 |
| **Net worth** | 0.387 | 1.000 | 0.107 |
| **Salary** | 0.629 | 0.107 | 1.000 |

**Simple reading:**
- Age and Salary: moderate positive link (`r = 0.629`)
- Age and Net worth: weak–moderate positive link (`r = 0.387`)
- Net worth and Salary: very weak link (`r = 0.107`)

---

## 5. Short takeaways

- Dataset reduced from **10 rows to 9** after merging duplicates.
- Typical **age**: about **30** (median); average **30.6**.
- Typical **salary**: **62,000** (median); average **62,556**.
- Typical **net worth**: **35,000** (median); average **37,778**.
- Most common country: **NZ** (about two-thirds of the sample).
- AUS shows a slightly higher average salary than NZ in this small sample — treat with care because the sample size is tiny.
- Cleaning and imputation were necessary before analysis because the raw file had blanks, text numbers, bad dates, and duplicates.
