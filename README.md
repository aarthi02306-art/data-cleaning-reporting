import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# 1. Load data
df = pd.read_csv("your_data.csv")   # <- change this
print(df.shape)
print(df.info())
print(df.describe())

# 2. Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# 3. Duplicates
print("Duplicates:", df.duplicated().sum())
df = df.drop_duplicates()

# 4. Missing values
print(df.isna().sum())
sns.heatmap(df.isna(), cbar=False)
plt.title("Missing values")
plt.show()

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include="object").columns

for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

# 5. Outliers (IQR method)
for c in num_cols:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    print(c, "outliers:", ((df[c] < low) | (df[c] > high)).sum())
    df[c] = df[c].clip(low, high)

# 6. Save cleaned data
df.to_csv("cleaned_data.csv", index=False)

# 7. Visualizations
df[num_cols].hist(figsize=(12, 8), bins=25)
plt.suptitle("Distributions")
plt.show()

plt.figure(figsize=(9, 7))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation heatmap")
plt.show()

for c in cat_cols[:3]:
    plt.figure(figsize=(8, 4))
    df[c].value_counts().head(10).plot(kind="barh")
    plt.title(f"Top categories: {c}")
    plt.show()
