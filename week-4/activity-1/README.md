# MSE-803 Week 4 — Activity 1: Happiness Dashboard and Data Visualisation

Simple happiness dashboard built from the cleaned World Happiness dataset using **Matplotlib** and **Plotly**.

## GitHub

https://github.com/mdasufian/MSE-803-data-analytics/tree/main/week-4/activity-1

## Files

| File | Description |
|------|-------------|
| `happiness_dashboard.ipynb` | **Main deliverable** — Jupyter notebook dashboard |
| `world_happiness_dataset.csv` | Cleaned dataset from Blackboard (20 countries) |
| `happiness_dashboard.py` | Optional script version of the same analysis |
| `outputs/` | Generated PNG and interactive HTML charts |

## How to run

### Jupyter notebook (recommended)

```bash
cd week-4/activity-1
jupyter notebook happiness_dashboard.ipynb
```

Or open `happiness_dashboard.ipynb` in VS Code / Cursor and **Run All**.

### Python script (optional)

```bash
cd week-4/activity-1
python3 happiness_dashboard.py
```

**Requirements:** Python 3 with `pandas`, `matplotlib`, and `plotly`.

Then open:

- `outputs/matplotlib_happiness_dashboard.png` — static dashboard
- `outputs/plotly_happiness_dashboard.html` — interactive dashboard (browser)
- `outputs/plotly_top3_happiness.html` — interactive top-3 comparison

---

## Approach

1. **Load** the cleaned CSV and confirm there are no missing values.
2. **Aggregate / rank** by sorting on `Happiness_Score` (one row per country) to get:
   - the **three happiest** countries
   - the country with the **lowest** Happiness Score
3. **Visualise** with both libraries:
   - **Matplotlib** — static side-by-side bar dashboard (PNG)
   - **Plotly** — interactive bar dashboard (HTML) plus a standalone top-3 chart
4. For Freedom, show the lowest-happiness country’s `Freedom_to_Make_Choices` next to the **dataset mean** so the score has context.

---

## Findings

### Three happiest countries

| Rank | Country | Happiness Score |
|------|---------|-----------------|
| 1 | Canada | 7.34 |
| 2 | Brazil | 6.98 |
| 3 | Finland | 6.67 |

Canada leads this sample, followed closely by Brazil and Finland.

### Freedom summary — lowest Happiness Score

| Country | Happiness Score | Freedom to Make Choices | Dataset mean Freedom |
|---------|-----------------|-------------------------|----------------------|
| South Africa | 3.53 | 0.90 | 0.66 |

South Africa has the **lowest Happiness Score** in this dataset, but a **relatively high Freedom** score (0.90), above the sample average. In this small sample, low happiness does not automatically mean low freedom.

---

## Which chart is most appropriate — and why?

A **vertical bar chart** is the best fit for this task.

| Option | Why it fits / does not |
|--------|------------------------|
| **Bar chart (chosen)** | Compares a small number of **categories** (countries) on one **numeric** measure. Ranking and differences are easy to see. |
| Line chart | Better for trends over time; we have no time series here. |
| Pie / donut | Poor for precise comparison of magnitudes; hard to compare close values (e.g. Brazil vs Finland). |
| Scatter | Better for relationships between two/three continuous variables, not a simple ranked comparison. |

For **three countries**, bars are clear, labelled, and suitable for both Matplotlib (report/PNG) and Plotly (interactive hover).

---

## Short takeaways

- Dataset: **20 countries**, **0 missing values** — ready for visualisation.
- Top 3 by Happiness: **Canada (7.34), Brazil (6.98), Finland (6.67)**.
- Lowest Happiness: **South Africa (3.53)** with Freedom **0.90** (above the mean).
- **Bar charts** are the most appropriate visualisation for this comparison task.
