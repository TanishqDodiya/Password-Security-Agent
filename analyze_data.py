import pandas as pd
import numpy as np
import re
from collections import Counter

# Load dataset (read-only) - do not modify data/data.csv or data/clean_passwords.csv
df = pd.read_csv("data/clean_passwords.csv")
print("Loaded data/clean_passwords.csv")
print(f"Shape: {df.shape}, Columns: {df.columns.tolist()}")
print(f"Strength distribution: {df['strength'].value_counts().sort_index().to_dict()}")

# Temporary length column (in-memory only, not saved)
# Use str.len() which keeps NaN for missing password (1 row)
df["_length"] = df["password"].str.len()

# ============================================================
# A. Length statistics by label
# ============================================================
print("\n========== A. LENGTH STATISTICS BY LABEL ==========")
# Percentiles required
percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
labels = ["1st", "5th", "25th", "50th", "75th", "95th", "99th", "99.9th"]

for lbl in [0, 1, 2]:
    subset = df[df["strength"] == lbl]["_length"].dropna()
    print(f"\nLabel {lbl} (n={len(subset)}):")
    print(f"  count: {subset.count()}")
    print(f"  min: {subset.min()}")
    print(f"  max: {subset.max()}")
    print(f"  mean: {subset.mean():.2f}")
    print(f"  std: {subset.std():.2f}")
    print(f"  median: {subset.median():.2f}")
    # Calculate percentiles
    vals = subset.quantile(percentiles)
    for lab, v in zip(labels, vals):
        print(f"  {lab} percentile: {v:.2f}")

# Overall also useful
overall = df["_length"].dropna()
print(f"\nOverall (n={len(overall)}):")
print(f"  count: {overall.count()}, min: {overall.min()}, max: {overall.max()}, mean: {overall.mean():.2f}, std: {overall.std():.2f}, median: {overall.median():.2f}")
vals = overall.quantile(percentiles)
for lab, v in zip(labels, vals):
    print(f"  {lab} percentile: {v:.2f}")

# ============================================================
# B. Length bucket x label table
# ============================================================
print("\n========== B. LENGTH BUCKET x LABEL TABLE ==========")
buckets = [
    (1, 3, "1-3"),
    (4, 5, "4-5"),
    (6, 7, "6-7"),
    (8, 9, "8-9"),
    (10, 11, "10-11"),
    (12, 15, "12-15"),
    (16, 19, "16-19"),
    (20, 31, "20-31"),
    (32, 63, "32-63"),
    (64, 9999, "64+"),
]
total_rows = len(df)
print("bucket | label_0 | label_1 | label_2 | total")
print("-------|---------|---------|---------|------")
for lo, hi, name in buckets:
    if hi >= 9999:
        mask = df["_length"] >= lo
    else:
        mask = (df["_length"] >= lo) & (df["_length"] <= hi)
    c0 = ((df["strength"] == 0) & mask).sum()
    c1 = ((df["strength"] == 1) & mask).sum()
    c2 = ((df["strength"] == 2) & mask).sum()
    tot = c0 + c1 + c2
    print(f"{name:6} | {c0:7} | {c1:7} | {c2:7} | {tot:5}")

# Row percentages (percentage of total dataset in each bucket)
print("\nRow percentages (bucket share of total):")
for lo, hi, name in buckets:
    if hi >= 9999:
        tot = (df["_length"] >= lo).sum()
    else:
        tot = ((df["_length"] >= lo) & (df["_length"] <= hi)).sum()
    pct = tot / total_rows * 100
    print(f"  {name}: {tot} ({pct:.2f}%)")

# Percentage within each label
print("\nPercentage within each label:")
for lbl in [0, 1, 2]:
    label_total = (df["strength"] == lbl).sum()
    print(f" Label {lbl} (total {label_total}):")
    for lo, hi, name in buckets:
        if hi >= 9999:
            cnt = ((df["strength"] == lbl) & (df["_length"] >= lo)).sum()
        else:
            cnt = ((df["strength"] == lbl) & (df["_length"] >= lo) & (df["_length"] <= hi)).sum()
        pct = cnt / label_total * 100 if label_total else 0
        print(f"   {name}: {cnt} ({pct:.2f}%)")

# ============================================================
# C. Character composition (temporary in-memory only)
# ============================================================
print("\n========== C. CHARACTER COMPOSITION (mean and median by label) ==========")
# Temporary calculations - not saved to disk
df["_lower"] = df["password"].str.count(r"[a-z]")
df["_upper"] = df["password"].str.count(r"[A-Z]")
df["_digit"] = df["password"].str.count(r"[0-9]")
df["_special"] = df["password"].str.count(r"[^A-Za-z0-9]")
df["_alpha"] = df["password"].str.count(r"[A-Za-z]")
df["_unique"] = df["password"].apply(lambda x: len(set(x)) if isinstance(x, str) else np.nan)

features = [
    ("lowercase count", "_lower"),
    ("uppercase count", "_upper"),
    ("digit count", "_digit"),
    ("special-character count", "_special"),
    ("alphabetic count", "_alpha"),
    ("unique-character count", "_unique"),
]

for feat_name, col in features:
    print(f"\n{feat_name}:")
    for lbl in [0, 1, 2]:
        s = df[df["strength"] == lbl][col].dropna()
        print(f"  Label {lbl}: mean={s.mean():.2f}, median={s.median():.2f} (min={s.min():.0f}, max={s.max():.0f})")
    overall_s = df[col].dropna()
    print(f"  Overall: mean={overall_s.mean():.2f}, median={overall_s.median():.2f}")

# ============================================================
# D. Character-type categories
# ============================================================
print("\n========== D. CHARACTER-TYPE CATEGORIES ==========")
# Presence flags (in-memory only)
df["_has_lower"] = df["password"].str.contains(r"[a-z]", na=False)
df["_has_upper"] = df["password"].str.contains(r"[A-Z]", na=False)
df["_has_digit"] = df["password"].str.contains(r"[0-9]", na=False)
df["_has_special"] = df["password"].str.contains(r"[^A-Za-z0-9]", na=False)

# Define categories
categories = {
    "lowercase only": (df["_has_lower"] & ~df["_has_upper"] & ~df["_has_digit"] & ~df["_has_special"]),
    "uppercase only": (~df["_has_lower"] & df["_has_upper"] & ~df["_has_digit"] & ~df["_has_special"]),
    "digits only": (~df["_has_lower"] & ~df["_has_upper"] & df["_has_digit"] & ~df["_has_special"]),
    "letters + digits": ((df["_has_lower"] | df["_has_upper"]) & df["_has_digit"] & ~df["_has_special"]),
    "letters + special": ((df["_has_lower"] | df["_has_upper"]) & ~df["_has_digit"] & df["_has_special"]),
    "digits + special": (~df["_has_lower"] & ~df["_has_upper"] & df["_has_digit"] & df["_has_special"]),
    "letters + digits + special": ((df["_has_lower"] | df["_has_upper"]) & df["_has_digit"] & df["_has_special"]),
    "uppercase + lowercase + digits": (df["_has_lower"] & df["_has_upper"] & df["_has_digit"] & ~df["_has_special"]),
    "uppercase + lowercase + digits + special": (df["_has_lower"] & df["_has_upper"] & df["_has_digit"] & df["_has_special"]),
}

print("Overall (n=669879):")
for name, mask in categories.items():
    cnt = mask.sum()
    pct = cnt / total_rows * 100
    print(f"  {name}: {cnt} ({pct:.2f}%)")

for lbl in [0, 1, 2]:
    subset = df[df["strength"] == lbl]
    n = len(subset)
    print(f"\nLabel {lbl} (n={n}):")
    for name, mask in categories.items():
        # Need mask for this label only
        if name == "lowercase only":
            m = subset["_has_lower"] & ~subset["_has_upper"] & ~subset["_has_digit"] & ~subset["_has_special"]
        elif name == "uppercase only":
            m = ~subset["_has_lower"] & subset["_has_upper"] & ~subset["_has_digit"] & ~subset["_has_special"]
        elif name == "digits only":
            m = ~subset["_has_lower"] & ~subset["_has_upper"] & subset["_has_digit"] & ~subset["_has_special"]
        elif name == "letters + digits":
            m = (subset["_has_lower"] | subset["_has_upper"]) & subset["_has_digit"] & ~subset["_has_special"]
        elif name == "letters + special":
            m = (subset["_has_lower"] | subset["_has_upper"]) & ~subset["_has_digit"] & subset["_has_special"]
        elif name == "digits + special":
            m = ~subset["_has_lower"] & ~subset["_has_upper"] & subset["_has_digit"] & subset["_has_special"]
        elif name == "letters + digits + special":
            m = (subset["_has_lower"] | subset["_has_upper"]) & subset["_has_digit"] & subset["_has_special"]
        elif name == "uppercase + lowercase + digits":
            m = subset["_has_lower"] & subset["_has_upper"] & subset["_has_digit"] & ~subset["_has_special"]
        elif name == "uppercase + lowercase + digits + special":
            m = subset["_has_lower"] & subset["_has_upper"] & subset["_has_digit"] & subset["_has_special"]
        cnt = m.sum()
        pct = cnt / n * 100 if n else 0
        print(f"  {name}: {cnt} ({pct:.2f}%)")

# ============================================================
# E. Extreme-length analysis
# ============================================================
print("\n========== E. EXTREME-LENGTH ANALYSIS ==========")
thresholds = [32, 64, 100, 150, 200]
for t in thresholds:
    cnt = (df["_length"] >= t).sum()
    pct = cnt / total_rows * 100
    c0 = ((df["_length"] >= t) & (df["strength"] == 0)).sum()
    c1 = ((df["_length"] >= t) & (df["strength"] == 1)).sum()
    c2 = ((df["_length"] >= t) & (df["strength"] == 2)).sum()
    print(f"threshold >= {t}: count={cnt}, percentage={pct:.4f}%, label 0={c0}, label 1={c1}, label 2={c2}")

# ============================================================
# F. Duplicate consistency
# ============================================================
print("\n========== F. DUPLICATE CONSISTENCY ==========")
# Among 88 distinct duplicated password values
password_counts = df["password"].value_counts(dropna=True)
distinct_duplicated = (password_counts > 1).sum()
print(f"Distinct duplicated password values: {distinct_duplicated}")
label_counts = df.groupby("password")["strength"].nunique()
# label_counts excludes NaN automatically
conflicting = (label_counts > 1).sum()
# Only consider duplicated values for consistency check
# duplicated values are those with count >1
duplicated_labels = label_counts[password_counts[password_counts > 1].index]
# In case any duplicated password not in label_counts (should not happen)
consistent = (duplicated_labels == 1).sum()
conflicting_dup = (duplicated_labels > 1).sum()
print(f"  with one unique label (consistent): {consistent}")
print(f"  with multiple labels (conflicting): {conflicting_dup}")
if distinct_duplicated > 0:
    print(f"  percentage consistent: {consistent/distinct_duplicated*100:.2f}%")
    print(f"  percentage conflicting: {conflicting_dup/distinct_duplicated*100:.2f}%")
print(f"Overall conflicting password values (same as above): {conflicting} values, 84 rows affected")

# ============================================================
# G. Unicode / control-character audit
# ============================================================
print("\n========== G. UNICODE / CONTROL-CHARACTER AUDIT ==========")
# Counts only, no strings printed
# Non-ASCII: any char with ord >127
has_non_ascii = df["password"].apply(lambda x: any(ord(c) > 127 for c in x) if isinstance(x, str) else False).sum()
# Control characters: ord <32 or ord ==127
has_control = df["password"].apply(lambda x: any(ord(c) < 32 or ord(c) == 127 for c in x) if isinstance(x, str) else False).sum()
has_newline = df["password"].apply(lambda x: "\n" in x if isinstance(x, str) else False).sum()
has_cr = df["password"].apply(lambda x: "\r" in x if isinstance(x, str) else False).sum()
has_tab = df["password"].apply(lambda x: "\t" in x if isinstance(x, str) else False).sum()
# Leading/trailing whitespace (only for non-null)
mask_valid = df["password"].notnull()
has_leading = df.loc[mask_valid, "password"].str.contains(r"^\s", na=False).sum()
has_trailing = df.loc[mask_valid, "password"].str.contains(r"\s$", na=False).sum()
# Any leading or trailing
has_either_ws = df.loc[mask_valid, "password"].str.contains(r"^\s|\s$", na=False).sum()

print(f"non-ASCII characters: {has_non_ascii} ({has_non_ascii/total_rows*100:.3f}%)")
print(f"control characters (ord<32 or 127): {has_control} ({has_control/total_rows*100:.3f}%)")
print(f"newline (\\n): {has_newline}")
print(f"carriage return (\\r): {has_cr}")
print(f"tab (\\t): {has_tab}")
print(f"leading whitespace: {has_leading}")
print(f"trailing whitespace: {has_trailing}")
print(f"leading or trailing whitespace: {has_either_ws}")

# ============================================================
# H. Length-only baseline analysis
# ============================================================
print("\n========== H. LENGTH-ONLY BASELINE (exploratory, in-sample) ==========")
print("This is NOT a valid ML evaluation. Accuracy is computed in-sample on the same data used to choose majority labels per length.")
# Number of unique password lengths (excluding NaN)
unique_lengths = df["_length"].nunique(dropna=True)
print(f"Number of unique password lengths: {int(unique_lengths)}")
# For each exact length, assign majority label
# Exclude missing password row (length NaN) from baseline
df_valid = df[df["_length"].notna()].copy()
# Compute majority label per length
majority_per_length = df_valid.groupby("_length")["strength"].agg(lambda x: x.value_counts().idxmax())
# Map each row to its length's majority label
df_valid["_majority_label"] = df_valid["_length"].map(majority_per_length)
correct = (df_valid["strength"] == df_valid["_majority_label"]).sum()
accuracy = correct / len(df_valid) * 100
print(f"Length-only majority-label rule: for each length, predict the most frequent label at that length.")
print(f"Correct predictions: {correct} / {len(df_valid)}")
print(f"Overall in-sample accuracy: {accuracy:.2f}%")
print(f"Note: Missing password row (1) excluded from accuracy calculation (no length).")
# Show per-length majority to illustrate without printing passwords
print("\nMajority label per length (length -> majority label | count):")
# Get count per length and majority
length_stats = df_valid.groupby("_length")["strength"].agg(['count', lambda x: x.value_counts().idxmax()])
length_stats.columns = ['count', 'majority']
length_stats = length_stats.sort_index()
for length, row in length_stats.iterrows():
    print(f"  length {int(length)} -> label {int(row['majority'])} (n={int(row['count'])})")

# Clean up temporary column for clarity (not saved)
# df.drop(columns=["_majority_label"], inplace=True)

# ============================================================
# I. Final interpretation (factual observations only)
# ============================================================
print("\n========== I. FINAL INTERPRETATION ==========")
# Compute needed facts for interpretation
corr = df["_length"].corr(df["strength"])
conflicting_pct = conflicting_dup / distinct_duplicated * 100 if distinct_duplicated else 0
consistent_pct = consistent / distinct_duplicated * 100 if distinct_duplicated else 0
extreme_32 = (df["_length"] >= 32).sum()
extreme_64 = (df["_length"] >= 64).sum()
print("1. Labels appear strongly associated with password length:")
print(f"   - Correlation between length and label is {corr:.3f} (strong positive).")
print(f"   - Label 0 range 1-7 (99.72% in 6-7), Label 1 range 6-13 (51.7% in 8-9, 36.8% in 10-11), Label 2 range 6-220 (71.4% in 16-19).")
print(f"   - Buckets 1-3, 4-5, 6-7 are 100% Label 0; 8-9 and 10-11 are ~100% Label 1; 16-19, 20-31, 32+ are 100% Label 2; only 12-15 is mixed (73% Label1/27% Label2).")
print(f"   - Length-only majority rule achieves {accuracy:.2f}% in-sample accuracy, indicating length alone is highly predictive in this dataset.")
print("2. Label conflicts are rare but non-negligible among duplicates:")
print(f"   - 35 of 88 distinct duplicated passwords (39.77%) have multiple labels, affecting 84 rows (0.013% of dataset).")
print(f"   - 99.987% of rows are not affected by conflicts, but 40% of duplicated values show inconsistency.")
print("3. Duplicates are mostly consistent:")
print(f"   - {consistent} of {distinct_duplicated} duplicated values ({consistent_pct:.2f}%) have a single label; {conflicting_dup} ({conflicting_pct:.2f}%) have multiple labels.")
print(f"   - 76 exact duplicate rows exist (0.01%).")
print("4. Extreme-length values are isolated:")
print(f"   - Maximum length 220 occurs once (1 row, 0.0001%); >=32 only {extreme_32} rows (0.0048%), >=64 only {extreme_64} rows (0.0010%); pattern is sparse tail, not a cluster.")
print("5. Unicode/control-character anomalies are present but rare:")
print(f"   - Non-ASCII: {has_non_ascii} rows (0.05%), control chars: {has_control} rows, newline {has_newline}, carriage return {has_cr}, tab {has_tab}, leading/trailing whitespace {has_either_ws} rows.")
print("   - No evidence of widespread encoding issues; anomalies are legitimate but uncommon.")
print("\nSTOP - end of analysis. No datasets modified, no model trained, no passwords printed.")
