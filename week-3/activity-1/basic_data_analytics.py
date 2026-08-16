"""
MSE-803 Week 3 Activity 1 — Data Cleaning + Basic Analytics
------------------------------------------------------------
1. Clean messy values
2. Prepare / fill missing data (imputation)
3. Save a clean dataset
4. Run simple descriptive analysis

Kept simple for a master's data analytics class.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
RAW_PATH = BASE_DIR / "Sample_dataset.csv"
CLEAN_PATH = BASE_DIR / "Sample_dataset_cleaned.csv"

# Simple word → number map for a few dirty values in this sample
WORD_NUMBERS = {
    "thirty-eight": 38,
    "thirty eight": 38,
    "sixty five thousand": 65000,
    "sixty-five thousand": 65000,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def clean_number(value):
    """Convert messy number text into a real number, or NaN if not possible."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower().replace(",", "")
    if text in WORD_NUMBERS:
        return float(WORD_NUMBERS[text])

    try:
        return float(text)
    except ValueError:
        return np.nan


def clean_country(value):
    """Standardise country codes (AU / AUS → AUS, NZ → NZ)."""
    if pd.isna(value):
        return np.nan
    code = str(value).strip().upper()
    if code in {"AU", "AUS", "AUSTRALIA"}:
        return "AUS"
    if code in {"NZ", "NEW ZEALAND"}:
        return "NZ"
    return code


def merge_duplicate_ids(frame):
    """
    Merge rows that share the same ID.
    Example: Bob appears twice with different blanks → one combined row.
    """
    parts = []
    for _, group in frame.groupby(frame["ID"].astype("float"), dropna=False, sort=False):
        if len(group) == 1:
            parts.append(group.iloc[[0]])
            continue

        row = group.iloc[0].copy()
        for col in group.columns:
            if pd.isna(row[col]):
                filled = group[col].dropna()
                if len(filled):
                    row[col] = filled.iloc[0]
        parts.append(pd.DataFrame([row]))

    return pd.concat(parts, ignore_index=True)


def show_missing(frame, title):
    """Print a simple missing-value summary."""
    missing = frame.isna().sum()
    table = pd.DataFrame(
        {
            "Missing count": missing,
            "Missing %": (missing / len(frame) * 100).round(1),
        }
    )
    print(f"\n{title}")
    print(table.to_string())
    print(f"Total missing cells: {int(missing.sum())}")


# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1 — LOAD RAW DATA")
print("=" * 70)

df_raw = pd.read_csv(RAW_PATH)
df_raw.columns = [c.strip() for c in df_raw.columns]

print(f"File: {RAW_PATH.name}")
print(f"Shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
print("\nRaw data:")
print(df_raw.to_string(index=False))
show_missing(df_raw, "Missing values in RAW data:")


# ---------------------------------------------------------------------------
# 2. Clean messy values (not filling yet)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2 — CLEAN MESSY VALUES")
print("=" * 70)
print(
    """
What we fix here
----------------
• Remove commas from numbers (e.g. "30,000" → 30000)
• Convert word numbers (e.g. "thirty-eight" → 38)
• Standardise country codes (AU → AUS)
• Parse dates; invalid dates become missing (e.g. 2019-13-01)
• Merge duplicate ID rows (Bob)
"""
)

df = df_raw.copy()

for col in ["Age", "Net worth", "Salary"]:
    df[col] = df[col].apply(clean_number)

df["Country"] = df["Country"].apply(clean_country)
df["Join Date"] = pd.to_datetime(df["Join Date"], dayfirst=True, errors="coerce")
df = merge_duplicate_ids(df)

# Make Name empty strings into real missing values
df["Name"] = df["Name"].replace(r"^\s*$", np.nan, regex=True)

print("Data after cleaning (before filling missing values):")
print(df.to_string(index=False))
show_missing(df, "Missing values AFTER cleaning (before imputation):")


# ---------------------------------------------------------------------------
# 3. Prepare / fill missing data (imputation)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3 — PREPARE MISSING DATA (IMPUTATION)")
print("=" * 70)
print(
    """
Simple imputation rules used in this activity
---------------------------------------------
• ID          → next available integer ID
• Name        → "Unknown"
• Age         → median age          (middle value; robust to outliers)
• Net worth   → median net worth
• Salary      → median salary
• Country     → mode (most common country)
• Join Date   → median join date    (middle date)

Why median for numbers?
  Median is less affected by extreme values than the mean.
  For a small messy sample, median is a safe classroom choice.
"""
)

df_clean = df.copy()
imputation_log = []

# --- ID ---
missing_id_mask = df_clean["ID"].isna()
if missing_id_mask.any():
    max_id = int(df_clean["ID"].max(skipna=True))
    new_ids = list(range(max_id + 1, max_id + 1 + int(missing_id_mask.sum())))
    df_clean.loc[missing_id_mask, "ID"] = new_ids
    imputation_log.append(f"ID: filled {missing_id_mask.sum()} missing value(s) with {new_ids}")

df_clean["ID"] = df_clean["ID"].astype(int)

# --- Name ---
missing_name = int(df_clean["Name"].isna().sum())
if missing_name:
    df_clean["Name"] = df_clean["Name"].fillna("Unknown")
    imputation_log.append(f"Name: filled {missing_name} missing value(s) with 'Unknown'")

# --- Numeric columns: median ---
for col in ["Age", "Net worth", "Salary"]:
    missing_n = int(df_clean[col].isna().sum())
    if missing_n:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
        imputation_log.append(
            f"{col}: filled {missing_n} missing value(s) with median = {median_val:,.2f}"
        )

# --- Country: mode ---
missing_country = int(df_clean["Country"].isna().sum())
if missing_country:
    mode_country = df_clean["Country"].mode().iloc[0]
    df_clean["Country"] = df_clean["Country"].fillna(mode_country)
    imputation_log.append(
        f"Country: filled {missing_country} missing value(s) with mode = '{mode_country}'"
    )

# --- Join Date: median date ---
missing_date = int(df_clean["Join Date"].isna().sum())
if missing_date:
    valid_dates = df_clean["Join Date"].dropna().sort_values()
    median_date = valid_dates.iloc[len(valid_dates) // 2]
    df_clean["Join Date"] = df_clean["Join Date"].fillna(median_date)
    imputation_log.append(
        f"Join Date: filled {missing_date} missing value(s) with median date = {median_date.date()}"
    )

print("Imputation actions taken:")
for line in imputation_log:
    print(f"  • {line}")

print("\nPrepared / cleaned data (ready for analysis):")
print(df_clean.to_string(index=False))
show_missing(df_clean, "Missing values AFTER imputation:")

# Save cleaned dataset
df_to_save = df_clean.copy()
df_to_save["Join Date"] = df_to_save["Join Date"].dt.strftime("%d/%m/%Y")
df_to_save.to_csv(CLEAN_PATH, index=False)
print(f"\nCleaned dataset saved to: {CLEAN_PATH.name}")


# ---------------------------------------------------------------------------
# 4. Analytics on prepared data
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4 — ANALYTICS ON PREPARED DATA")
print("=" * 70)

df = df_clean

print("\n--- SECTION A: DATA QUALITY ---")
n_rows, n_cols = df.shape
quality = pd.DataFrame(
    {
        "Missing count": df.isna().sum(),
        "Missing %": (df.isna().sum() / n_rows * 100).round(1),
        "Data type": df.dtypes.astype(str),
    }
)
print(quality.to_string())
print(
    f"\nOverall completeness: "
    f"{(1 - df.isna().sum().sum() / (n_rows * n_cols)) * 100:.1f}% of cells have values."
)

print(
    """
How to read this section
------------------------
• Missing count / Missing %
  What it measures: how many values are blank or unusable in each column.
  Interpretation: after imputation, these should be 0 (or very low).

• Completeness
  What it measures: share of all cells that contain a usable value.
  Interpretation: closer to 100% means a fuller dataset for analysis.
"""
)


# ---------------------------------------------------------------------------
# 5. Numeric summary
# ---------------------------------------------------------------------------
print("--- SECTION B: NUMERIC SUMMARY (Age, Net worth, Salary) ---")

numeric_cols = ["Age", "Net worth", "Salary"]
summary_rows = []

for col in numeric_cols:
    series = df[col]
    summary_rows.append(
        {
            "Metric": col,
            "Count": int(series.count()),
            "Mean": round(series.mean(), 2),
            "Median": round(series.median(), 2),
            "Mode": round(series.mode().iloc[0], 2),
            "Min": round(series.min(), 2),
            "Max": round(series.max(), 2),
            "Range": round(series.max() - series.min(), 2),
            "Std Dev": round(series.std(ddof=1), 2),
            "Variance": round(series.var(ddof=1), 2),
            "Q1 (25%)": round(series.quantile(0.25), 2),
            "Q3 (75%)": round(series.quantile(0.75), 2),
            "IQR": round(series.quantile(0.75) - series.quantile(0.25), 2),
        }
    )

summary_df = pd.DataFrame(summary_rows).set_index("Metric")
display_df = summary_df.T.copy()
for col in display_df.columns:
    display_df[col] = display_df[col].apply(lambda x: f"{float(x):,.2f}")
print(display_df.to_string())

print(
    """
Metric explanations (simple language)
-------------------------------------
• Count
  Measures: how many values were used.
  Interpret: after imputation, count should equal the number of rows.

• Mean (average)
  Measures: typical value by adding all values and dividing by count.
  Interpret: good overall centre, but can be pulled by extreme values.

• Median
  Measures: the middle value when numbers are sorted.
  Interpret: better "typical" value when data has outliers.
  Tip: if mean >> median, a few high values are pulling the average up.

• Mode
  Measures: the most frequently occurring value.
  Interpret: useful for categories; for numbers it may matter less if
  every value is unique.

• Min / Max / Range
  Measures: extremes and total spread.
  Interpret: large range means high variability; sensitive to outliers.

• Standard Deviation (Std Dev)
  Measures: how far values sit from the mean.
  Interpret: low = consistent; high = widely spread.

• Variance
  Measures: average squared distance from the mean.
  Interpret: same story as std dev, but harder to read (squared units).

• Quartiles (Q1, Q3) and IQR
  Q1: 25% of values are below this point.
  Q3: 75% of values are below this point.
  IQR: spread of the middle 50% of the data.
  Interpret: robust measure of spread (less affected by outliers).
"""
)


# ---------------------------------------------------------------------------
# 6. Categorical summary
# ---------------------------------------------------------------------------
print("--- SECTION C: CATEGORICAL SUMMARY (Country) ---")
country_counts = df["Country"].value_counts()
country_pct = (df["Country"].value_counts(normalize=True) * 100).round(1)
print(pd.DataFrame({"Count": country_counts, "Percentage %": country_pct}).to_string())

print(
    """
How to read this section
------------------------
• Frequency (Count): how many people in each country group.
• Percentage %: each group's share of the total (adds to 100%).
"""
)


# ---------------------------------------------------------------------------
# 7. Group comparison
# ---------------------------------------------------------------------------
print("--- SECTION D: GROUP COMPARISON (Salary by Country) ---")
grouped = (
    df.groupby("Country")["Salary"]
    .agg(Count="count", Mean="mean", Median="median", Std="std")
    .round(2)
)
print(grouped.to_string())

print(
    """
How to read this section
------------------------
Compares salary centre and spread inside each country.
With a tiny sample, one person can change the average a lot.
"""
)


# ---------------------------------------------------------------------------
# 8. Correlation
# ---------------------------------------------------------------------------
print("--- SECTION E: CORRELATION ---")
corr = df[numeric_cols].corr(method="pearson").round(3)
print(corr.to_string())

print(
    """
How to read this section
------------------------
Pearson correlation (r) measures linear relationship (−1 to +1):
  +1 = perfect positive link
   0 = no linear link
  −1 = perfect negative link
Rough guide: |r| < 0.3 weak, 0.3–0.7 moderate, > 0.7 strong.
Remember: correlation ≠ causation.
"""
)


# ---------------------------------------------------------------------------
# 9. Key takeaways
# ---------------------------------------------------------------------------
print("--- SECTION F: KEY RESULTS ---")
print(f"• Rows after cleaning: {len(df)} (duplicates merged)")
print(f"• Typical age (median): {df['Age'].median():.0f}; average: {df['Age'].mean():.1f}")
print(
    f"• Typical salary (median): {df['Salary'].median():,.0f}; "
    f"average: {df['Salary'].mean():,.0f}"
)
print(
    f"• Typical net worth (median): {df['Net worth'].median():,.0f}; "
    f"average: {df['Net worth'].mean():,.0f}"
)
print(f"• Most common country: {df['Country'].mode().iloc[0]}")
print(f"• Clean file ready for reuse: {CLEAN_PATH.name}")

print("\nDone. Re-run with: python basic_data_analytics.py")
