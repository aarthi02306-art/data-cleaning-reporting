"""
Data Cleaning & Visualization Project - starter pipeline
Usage: python data_cleaning_visualization.py your_data.csv
Outputs: cleaned_data.csv, cleaning_log.txt, and charts in ./charts/
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
os.makedirs("charts", exist_ok=True)
log = []


def note(msg):
    print(msg)
    log.append(msg)


# 1. LOAD + INSPECT -----------------------------------------------------------
path = sys.argv[1] if len(sys.argv) > 1 else "raw_data.csv"
df = pd.read_csv(path)
note(f"Loaded {path}: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.info())
print(df.describe(include="all").T)

# 2. STANDARDIZE COLUMN NAMES + TEXT -----------------------------------------
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(r"[^\w]+", "_", regex=True))
for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.strip()

# Try converting object columns to dates / numbers where possible
for col in df.select_dtypes(include="object"):
    if "date" in col or "time" in col:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    else:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() > 0.9:  # mostly numeric -> treat as numeric
            df[col] = converted

# 3. DUPLICATES ---------------------------------------------------------------
dups = df.duplicated().sum()
df = df.drop_duplicates()
note(f"Removed {dups} duplicate rows")

# 4. MISSING VALUES -----------------------------------------------------------
missing = df.isna().sum()
note("Missing values before cleaning:\n" + missing[missing > 0].to_string())

plt.figure(figsize=(10, 5))
sns.heatmap(df.isna(), cbar=False, yticklabels=False)
plt.title("Missing Values Map (before imputation)")
plt.tight_layout()
plt.savefig("charts/01_missing_values.png", dpi=150)
plt.close()

# Drop columns that are >50% empty
too_empty = df.columns[df.isna().mean() > 0.5]
df = df.drop(columns=too_empty)
note(f"Dropped columns with >50% missing: {list(too_empty)}")

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

for c in num_cols:
    # median is robust to outliers; use mean if the column is roughly symmetric
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].mode().iloc[0])
note("Imputed numeric columns with median, categorical with mode")

# 5. OUTLIERS (IQR method, capped rather than deleted) -----------------------
fig, axes = plt.subplots(1, min(len(num_cols), 4), figsize=(14, 4), squeeze=False)
for ax, c in zip(axes[0], num_cols[:4]):
    sns.boxplot(y=df[c], ax=ax)
    ax.set_title(f"{c} (before)")
plt.tight_layout()
plt.savefig("charts/02_outliers_before.png", dpi=150)
plt.close()

for c in num_cols:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[c] < low) | (df[c] > high)).sum()
    if n_out:
        df[c] = df[c].clip(low, high)
        note(f"{c}: capped {n_out} outliers to [{low:.2f}, {high:.2f}]")

# 6. SAVE CLEAN DATA + LOG ----------------------------------------------------
df.to_csv("cleaned_data.csv", index=False)
with open("cleaning_log.txt", "w") as f:
    f.write("\n\n".join(log))
note(f"Final shape: {df.shape}")

# 7. VISUALIZATION ------------------------------------------------------------
# Distributions
df[num_cols].hist(figsize=(12, 8), bins=25)
plt.suptitle("Distributions of Numeric Features")
plt.tight_layout()
plt.savefig("charts/03_distributions.png", dpi=150)
plt.close()

# Correlation heatmap
if len(num_cols) > 1:
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("charts/04_correlation.png", dpi=150)
    plt.close()

# Category counts (top 10 of each categorical column, up to 3 columns)
for c in cat_cols[:3]:
    plt.figure(figsize=(8, 4))
    df[c].value_counts().head(10).plot(kind="barh")
    plt.title(f"Top categories: {c}")
    plt.tight_layout()
    plt.savefig(f"charts/05_{c}_counts.png", dpi=150)
    plt.close()

# Trend over time, if a date column exists
date_cols = df.select_dtypes(include="datetime").columns
if len(date_cols) and num_cols:
    d, v = date_cols[0], num_cols[0]
    trend = df.set_index(d)[v].resample("M").mean()
    plt.figure(figsize=(10, 4))
    trend.plot(marker="o")
    plt.title(f"Monthly average {v}")
    plt.tight_layout()
    plt.savefig("charts/06_trend.png", dpi=150)
    plt.close()

print("Done. Check cleaned_data.csv, cleaning_log.txt and the charts/ folder.")
