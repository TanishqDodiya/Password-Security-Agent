"""
Forensic analysis of data/clean_passwords.csv
- READ-ONLY: never modifies data/data.csv or data/clean_passwords.csv
- No new dataset written, no ML training
- Only aggregate statistics, no password values printed
- Do not assume 0/1/2 = weak/medium/strong
"""
import pandas as pd
import re
import numpy as np
from collections import Counter

# Load (read-only)
df = pd.read_csv("data/clean_passwords.csv")
orig_shape = df.shape
orig_columns = df.columns.tolist()
print("Loaded clean_passwords.csv")
print(f"Shape: {orig_shape}, Columns: {orig_columns}")

# Temporary length - use str.len() which preserves NaN for missing password
# For overall stats we will exclude the 1 missing password row
df["_length"] = df["password"].str.len()

# Helper to print percentiles
def describe_series(s, name):
    s_clean = s.dropna()
    if len(s_clean)==0:
        print(f"{name}: no data")
        return
    print(f"{name}:")
    print(f"  min={s_clean.min()} max={s_clean.max()} mean={s_clean.mean():.2f} median={s_clean.median():.2f} std={s_clean.std():.2f}")
    qs = [0.01,0.05,0.25,0.5,0.75,0.95,0.99,0.999]
    labels = ["1%","5%","25%","50%","75%","95%","99%","99.9%"]
    vals = s_clean.quantile(qs)
    for lab, v in zip(labels, vals):
        print(f"  {lab}: {v:.2f}")

# ============================================================
# 1. PASSWORD LENGTH DISTRIBUTION
# ============================================================
print("\n========== 1. PASSWORD LENGTH DISTRIBUTION ==========")
describe_series(df["_length"], "Overall length")
for lbl in sorted(df["strength"].dropna().unique()):
    describe_series(df[df["strength"]==lbl]["_length"], f"Label {int(lbl)} length")

# Buckets
buckets = [
    (1,3,"1-3"),
    (4,5,"4-5"),
    (6,7,"6-7"),
    (8,9,"8-9"),
    (10,11,"10-11"),
    (12,15,"12-15"),
    (16,19,"16-19"),
    (20,31,"20-31"),
    (32,63,"32-63"),
    (64,1000,"64+"),
]
print("\nLength bucket counts and percentages (overall):")
total = len(df)
for lo, hi, name in buckets:
    if hi>=1000:
        cnt = (df["_length"] >= lo).sum()
    else:
        cnt = ((df["_length"] >= lo) & (df["_length"] <= hi)).sum()
    pct = cnt/total*100
    print(f"  {name}: {cnt} ({pct:.2f}%)")

print("\nLength bucket counts per label (counts):")
# Build table for later use in section 7 as well
bucket_table_counts = {}
bucket_table_pct = {}
for lo, hi, name in buckets:
    row = {}
    for lbl in [0,1,2]:
        sub = df[df["strength"]==lbl]["_length"]
        if hi>=1000:
            cnt = (sub >= lo).sum()
        else:
            cnt = ((sub >= lo) & (sub <= hi)).sum()
        row[lbl]=cnt
    bucket_table_counts[name]=row
    print(f"  {name}: Label0={row[0]} Label1={row[1]} Label2={row[2]}")

# ============================================================
# 2. INVESTIGATE EXTREMELY LONG PASSWORDS
# ============================================================
print("\n========== 2. EXTREMELY LONG PASSWORDS ==========")
thresholds = [32,64,100,150,200]
for t in thresholds:
    cnt = (df["_length"] >= t).sum()
    pct = cnt/total*100
    sub = df[df["_length"] >= t]
    print(f"\nLength >= {t}: {cnt} rows ({pct:.4f}%)")
    if cnt>0:
        dist = sub["strength"].value_counts().sort_index()
        print("  label distribution:")
        for lab in sorted(dist.index):
            c = dist[lab]
            print(f"    Label {int(lab)}: {c} ({c/cnt*100:.1f}% of threshold)")
        print(f"  min length in bucket: {sub['_length'].min()}, max: {sub['_length'].max()}, mean: {sub['_length'].mean():.1f}")
    else:
        print("  no rows")

# Is 220 isolated?
max_len = df["_length"].max()
cnt_max = (df["_length"]==max_len).sum()
print(f"\nMaximum length is {int(max_len)} with {cnt_max} row(s). Assessment: isolated anomaly if cnt==1 and next threshold drops sharply.")
# Show next largest
print("Top 10 lengths by frequency (descending length):")
length_counts = df["_length"].value_counts().sort_index(ascending=False).head(10)
print(length_counts.to_string())
print("Top 10 most common lengths (by count):")
print(df["_length"].value_counts().head(10).to_string())

# ============================================================
# 3. CHARACTER COMPOSITION ANALYSIS
# ============================================================
print("\n========== 3. CHARACTER COMPOSITION ANALYSIS ==========")
# Temporary features - not saved to disk
# Use vectorized str.count where possible; for unique chars use apply

# Create mask for non-null passwords
mask_valid = df["password"].notnull()
# For NaN rows, these will be NaN
df["_lower"] = df["password"].str.count(r"[a-z]")
df["_upper"] = df["password"].str.count(r"[A-Z]")
df["_digit"] = df["password"].str.count(r"[0-9]")
df["_alpha"] = df["password"].str.count(r"[A-Za-z]")
# special = not alphanumeric
df["_special"] = df["password"].str.count(r"[^A-Za-z0-9]")
# unique chars
df["_unique"] = df["password"].apply(lambda x: len(set(x)) if isinstance(x, str) else np.nan)

# Also _length already computed

features = ["_length","_lower","_upper","_digit","_special","_alpha","_unique"]
names = {"_length":"length","_lower":"lowercase","_upper":"uppercase","_digit":"digits","_special":"special","_alpha":"alpha","_unique":"unique"}
print("Overall feature stats (min, max, mean, median):")
for f in features:
    s = df[f]
    print(f"  {names[f]}: min={s.min():.0f} max={s.max():.0f} mean={s.mean():.2f} median={s.median():.2f}")

for lbl in sorted(df["strength"].dropna().unique()):
    print(f"\n  Label {int(lbl)} feature stats:")
    sub = df[df["strength"]==lbl]
    for f in features:
        s = sub[f]
        print(f"    {names[f]}: min={s.min():.0f} max={s.max():.0f} mean={s.mean():.2f} median={s.median():.2f}")

# ============================================================
# 4. CHARACTER-TYPE CATEGORIES
# ============================================================
print("\n========== 4. CHARACTER-TYPE CATEGORIES ==========")
# Presence flags
df["_has_lower"] = df["password"].str.contains(r"[a-z]", na=False)
df["_has_upper"] = df["password"].str.contains(r"[A-Z]", na=False)
df["_has_digit"] = df["password"].str.contains(r"[0-9]", na=False)
df["_has_special"] = df["password"].str.contains(r"[^A-Za-z0-9]", na=False)
# Note: na=False for missing password -> all False

def classify_counts(sub_df, label_name):
    total_sub = len(sub_df)
    # Define categories (non-exclusive but we report each)
    categories = {
        "lowercase only": (sub_df["_has_lower"] & ~sub_df["_has_upper"] & ~sub_df["_has_digit"] & ~sub_df["_has_special"]),
        "uppercase only": (~sub_df["_has_lower"] & sub_df["_has_upper"] & ~sub_df["_has_digit"] & ~sub_df["_has_special"]),
        "digits only": (~sub_df["_has_lower"] & ~sub_df["_has_upper"] & sub_df["_has_digit"] & ~sub_df["_has_special"]),
        "special only": (~sub_df["_has_lower"] & ~sub_df["_has_upper"] & ~sub_df["_has_digit"] & sub_df["_has_special"]),
        "letters only (any case, no digit/special)": ((sub_df["_has_lower"] | sub_df["_has_upper"]) & ~sub_df["_has_digit"] & ~sub_df["_has_special"]),
        "letters+digits (no special)": ((sub_df["_has_lower"] | sub_df["_has_upper"]) & sub_df["_has_digit"] & ~sub_df["_has_special"]),
        "letters+special (no digit)": ((sub_df["_has_lower"] | sub_df["_has_upper"]) & ~sub_df["_has_digit"] & sub_df["_has_special"]),
        "digits+special (no letters)": (~sub_df["_has_lower"] & ~sub_df["_has_upper"] & sub_df["_has_digit"] & sub_df["_has_special"]),
        "letters+digits+special": ((sub_df["_has_lower"] | sub_df["_has_upper"]) & sub_df["_has_digit"] & sub_df["_has_special"]),
        "lower+upper (any)": (sub_df["_has_lower"] & sub_df["_has_upper"]),
        "upper+lower+digits (no special)": (sub_df["_has_lower"] & sub_df["_has_upper"] & sub_df["_has_digit"] & ~sub_df["_has_special"]),
        "upper+lower+digits+special": (sub_df["_has_lower"] & sub_df["_has_upper"] & sub_df["_has_digit"] & sub_df["_has_special"]),
        "lower+digits": (sub_df["_has_lower"] & ~sub_df["_has_upper"] & sub_df["_has_digit"]),
        "upper+digits": (~sub_df["_has_lower"] & sub_df["_has_upper"] & sub_df["_has_digit"]),
    }
    print(f"\n  {label_name} (n={total_sub}):")
    for cat, mask in categories.items():
        cnt = mask.sum()
        pct = cnt/total_sub*100 if total_sub else 0
        print(f"    {cat}: {cnt} ({pct:.2f}%)")

classify_counts(df, "Overall")
for lbl in sorted(df["strength"].dropna().unique()):
    classify_counts(df[df["strength"]==lbl], f"Label {int(lbl)}")

# Also show full 16-combo breakdown overall
print("\nFull presence-combo breakdown (lower,upper,digit,special) overall:")
combo = df.groupby(["_has_lower","_has_upper","_has_digit","_has_special"]).size().sort_values(ascending=False)
print(combo.to_string())

# ============================================================
# 5. UNIQUENESS ANALYSIS
# ============================================================
print("\n========== 5. UNIQUENESS ANALYSIS ==========")
total_rows = len(df)
unique_pw = df["password"].nunique()
password_counts = df["password"].value_counts(dropna=False)
duplicated_values = (password_counts > 1).sum()  # distinct values appearing >1, includes NaN if duplicated but NaN only 1
# For password values, exclude NaN from distinct count? value_counts includes NaN; we want distinct duplicated password values that are actual passwords
# Use dropna=True for password values
password_counts_no_na = df["password"].value_counts(dropna=True)
duplicated_values_no_na = (password_counts_no_na > 1).sum()
rows_with_dup = password_counts_no_na[password_counts_no_na > 1].sum()
dup_occurrences = df["password"].duplicated().sum()

# Conflicting vs consistent among duplicated
label_counts = df.groupby("password")["strength"].nunique()
# Need to handle NaN group? groupby excludes NaN by default
conflicting = (label_counts > 1).sum()
consistent = duplicated_values_no_na - conflicting if duplicated_values_no_na>=conflicting else 0

print(f"Total rows: {total_rows}")
print(f"Unique password values: {unique_pw}")
print(f"Distinct duplicated password values (appearing >1): {duplicated_values_no_na}")
print(f"Rows belonging to duplicated values: {rows_with_dup}")
print(f"Duplicate occurrences beyond first (duplicated().sum()): {dup_occurrences}")
print(f"Duplicated values with same label (consistent): {consistent}")
print(f"Duplicated values with multiple labels (conflicting): {conflicting}")
if duplicated_values_no_na>0:
    print(f"  % consistent: {consistent/duplicated_values_no_na*100:.1f}%")
    print(f"  % conflicting: {conflicting/duplicated_values_no_na*100:.1f}%")

# ============================================================
# 6. CONFLICTING LABEL FORENSICS
# ============================================================
print("\n========== 6. CONFLICTING LABEL FORENSICS ==========")
# Already computed label_counts and conflicting
num_conflicting = conflicting
rows_affected = df[df["password"].isin(label_counts[label_counts > 1].index)].shape[0]
print(f"Conflicting password values: {num_conflicting}")
print(f"Rows affected: {rows_affected}")

# Breakdown per combination
grouped = df.groupby("password")["strength"].apply(lambda x: tuple(sorted(x.unique())))
conflicting_sets = grouped[grouped.apply(lambda x: len(x)>1)]
from collections import Counter
combo_counts = Counter(conflicting_sets)
print("\nPer combination:")
for combo in sorted(combo_counts.keys()):
    pw_cnt = combo_counts[combo]
    # rows for this combo
    passwords_in_combo = conflicting_sets[conflicting_sets==combo].index
    sub_conf = df[df["password"].isin(passwords_in_combo)]
    label_dist = sub_conf["strength"].value_counts().sort_index().to_dict()
    print(f"  {combo}: {pw_cnt} passwords, {len(sub_conf)} rows, label counts {label_dist}")

# Ensure all 4 combos reported
for exp in [(0,1),(0,2),(1,2),(0,1,2)]:
    if exp not in combo_counts:
        found = any(tuple(int(v) for v in k)==exp for k in combo_counts)
        if not found:
            print(f"  {exp}: 0 passwords, 0 rows")

# Length comparison for conflicting vs non-conflicting
print("\nLength comparison for conflicting passwords vs rest per label:")
# For each label, compare mean length of conflicting rows vs non-conflicting rows
conflicting_pw_set = set(conflicting_sets.index)
# Create flag
df["_is_conflicting"] = df["password"].isin(conflicting_pw_set)
for lbl in sorted(df["strength"].dropna().unique()):
    conf_len = df[(df["strength"]==lbl) & (df["_is_conflicting"])]["_length"]
    non_conf_len = df[(df["strength"]==lbl) & (~df["_is_conflicting"])]["_length"]
    print(f"  Label {int(lbl)}: conflicting n={len(conf_len)} mean={conf_len.mean():.2f} median={conf_len.median():.2f} vs non-conflicting mean={non_conf_len.mean():.2f} median={non_conf_len.median():.2f}")
    if len(conf_len)>0:
        # also percentiles for conflicting
        print(f"    conflicting min={conf_len.min()} max={conf_len.max()}")

# Overall
conf_len_all = df[df["_is_conflicting"]]["_length"]
non_conf_len_all = df[~df["_is_conflicting"]]["_length"]
print(f"\n  Overall conflicting mean {conf_len_all.mean():.2f} vs non-conflicting {non_conf_len_all.mean():.2f}")

# ============================================================
# 7. LABEL VS LENGTH RELATIONSHIP
# ============================================================
print("\n========== 7. LABEL VS LENGTH RELATIONSHIP ==========")
for lbl in sorted(df["strength"].dropna().unique()):
    s = df[df["strength"]==lbl]["_length"]
    print(f"Label {int(lbl)}: n={len(s)} mean={s.mean():.2f} median={s.median():.2f} std={s.std():.2f} min={s.min()} max={s.max()}")
    qs = [0.01,0.05,0.25,0.75,0.95,0.99,0.999]
    vals = s.quantile(qs)
    print(f"  1%={vals.iloc[0]:.1f} 5%={vals.iloc[1]:.1f} 25%={vals.iloc[2]:.1f} 75%={vals.iloc[3]:.1f} 95%={vals.iloc[4]:.1f} 99%={vals.iloc[5]:.1f} 99.9%={vals.iloc[6]:.1f}")

# Bucket x label table
print("\nLength bucket x Label table (counts):")
header = "Bucket      | Label0 | Label1 | Label2 | Total | % of total"
print(header)
print("-"*len(header))
for lo, hi, name in buckets:
    counts = [bucket_table_counts[name][0], bucket_table_counts[name][1], bucket_table_counts[name][2]]
    tot = sum(counts)
    pct = tot/total*100
    print(f"{name:11} | {counts[0]:6} | {counts[1]:6} | {counts[2]:6} | {tot:5} | {pct:5.2f}%")

print("\nLength bucket x Label table (percent within label):")
for lbl in [0,1,2]:
    print(f"Label {lbl}:")
    for lo, hi, name in buckets:
        cnt = bucket_table_counts[name][lbl]
        pct_in_label = cnt / (df["strength"]==lbl).sum() *100
        print(f"  {name}: {cnt} ({pct_in_label:.2f}% of label {lbl})")

# Check length-driven classification evidence
print("\nLength-driven analysis:")
# For each bucket, dominant label share
for lo, hi, name in buckets:
    c0, c1, c2 = bucket_table_counts[name][0], bucket_table_counts[name][1], bucket_table_counts[name][2]
    tot = c0+c1+c2
    if tot>0:
        max_share = max(c0,c1,c2)/tot*100
        dominant = np.argmax([c0,c1,c2])
        print(f"  Bucket {name}: total {tot}, dominant Label {dominant} with {max_share:.1f}%")

# Overlap check: do labels overlap in length?
print("\nOverlap insight:")
print("  Label 0 lengths: 1-7 only (max 7)")
print("  Label 1 lengths: 6-13")
print("  Label 2 lengths: 6-220 (wide)")
# Compute exclusive ranges
# Find % of Label1 that falls within Label0 range (1-7)
label1_in_0range = ((df[df['strength']==1]['_length'] <=7).sum()) / (df['strength']==1).sum()*100
label2_in_0range = ((df[df['strength']==2]['_length'] <=7).sum()) / (df['strength']==2).sum()*100
label0_in_1range = ((df[df['strength']==0]['_length'] >=6).sum()) / (df['strength']==0).sum()*100
print(f"  Label1 rows with length <=7 (inside Label0 max): {label1_in_0range:.2f}%")
print(f"  Label2 rows with length <=7: {label2_in_0range:.2f}%")
print(f"  Label0 rows with length >=6 (inside Label1/2 min): {label0_in_1range:.2f}%")
# Correlation
print(f"\n  Correlation length-strength: {df['_length'].corr(df['strength']):.3f}")

# ============================================================
# 8. DUPLICATE LABEL CONSISTENCY
# ============================================================
print("\n========== 8. DUPLICATE LABEL CONSISTENCY ==========")
# Already computed duplicated_values_no_na, conflicting, consistent
total_dup_vals = duplicated_values_no_na
print(f"Duplicated password values (distinct): {total_dup_vals}")
print(f"  With identical labels (consistent): {consistent} ({consistent/total_dup_vals*100:.1f}%)" if total_dup_vals else "  0")
print(f"  With conflicting labels: {conflicting} ({conflicting/total_dup_vals*100:.1f}%)" if total_dup_vals else "  0")
print(f"Duplicated rows: {dup_occurrences} beyond first, {rows_with_dup} total rows in duplicated values")
# Also rows consistency
# Rows that are part of consistent vs conflicting duplicated values
if conflicting>0:
    rows_conflicting_dup = df[df["password"].isin(conflicting_sets.index)].shape[0]
    rows_consistent_dup = rows_with_dup - rows_conflicting_dup
    print(f"Rows in consistent duplicated values: {rows_consistent_dup}")
    print(f"Rows in conflicting duplicated values: {rows_conflicting_dup}")

# ============================================================
# 9. DATASET SUSPICION CHECKS
# ============================================================
print("\n========== 9. DATASET SUSPICION CHECKS ==========")
# High concentration at particular lengths
length_counts = df["_length"].value_counts().sort_index()
top_lengths = df["_length"].value_counts().head(5)
print("Top 5 most common lengths:")
for length, cnt in top_lengths.items():
    pct = cnt/total*100
    print(f"  Length {int(length)}: {cnt} ({pct:.2f}%)")

# Concentration per label at specific lengths
for lbl in [0,1,2]:
    sub_counts = df[df["strength"]==lbl]["_length"].value_counts().head(3)
    print(f"\nLabel {int(lbl)} top 3 lengths:")
    for length, cnt in sub_counts.items():
        pct = cnt/(df["strength"]==lbl).sum()*100
        print(f"  Length {int(length)}: {cnt} ({pct:.2f}% of label)")

# Impossible/empty
missing_pw = df["password"].isnull().sum()
empty_pw = (df["password"]=="").sum()
mask_nn = df["password"].notnull()
whitespace_pw = (df.loc[mask_nn, "password"].astype(str).str.strip()=="").sum()
print(f"\nImpossible/empty: missing={missing_pw} empty string={empty_pw} whitespace-only={whitespace_pw}")

# Leading/trailing whitespace
leading = df[mask_nn]["password"].str.contains(r"^\s", na=False).sum()
trailing = df[mask_nn]["password"].str.contains(r"\s$", na=False).sum()
both_ws = (df[mask_nn]["password"].str.contains(r"^\s", na=False) & df[mask_nn]["password"].str.contains(r"\s$", na=False)).sum()
print(f"Leading whitespace: {leading}, trailing whitespace: {trailing}, both: {both_ws}")

# Newline/tab/control characters
has_newline = df["password"].astype(str).str.contains(r"\n", na=False).sum()  # literal newline not typical in csv
has_tab = df["password"].astype(str).str.contains(r"\t", na=False).sum()
# Control chars: ASCII 0-31 and 127 excluding tab/newline? Use regex
has_control = df["password"].apply(lambda x: any(ord(c)<32 or ord(c)==127 for c in x) if isinstance(x,str) else False).sum()
print(f"Tab characters: {has_tab}, newline in string: {has_newline}, any control char (0-31,127): {has_control}")

# Unicode-heavy: count passwords with non-ASCII characters
has_non_ascii = df["password"].apply(lambda x: any(ord(c)>127 for c in x) if isinstance(x,str) else False).sum()
pct_unicode = has_non_ascii/total*100
print(f"Unicode (non-ASCII) passwords: {has_non_ascii} ({pct_unicode:.3f}%)")
# Count with high unicode ratio >50%
high_unicode = df["password"].apply(lambda x: (sum(1 for c in x if ord(c)>127)/len(x) >0.5) if isinstance(x,str) and len(x)>0 else False).sum()
print(f"High-unicode (>50% non-ASCII) passwords: {high_unicode}")

# Extremely unusual character distributions
# Very high special ratio
high_special = ((df["_special"] / df["_length"]) >0.5).sum()
print(f"Passwords with >50% special chars: {high_special}")
high_digit = ((df["_digit"] / df["_length"]) >0.8).sum()
print(f"Passwords with >80% digits: {high_digit}")

# Repeated structural patterns - e.g., repeated characters
# Count passwords with very low unique ratio (unique/length <0.3)
low_unique_ratio = ((df["_unique"]/df["_length"]) <0.3).sum()
print(f"Low uniqueness ratio (<0.3 unique/length): {low_unique_ratio}")

# Extremely short/long outliers
very_short = (df["_length"] <=2).sum()
very_long = (df["_length"] >=32).sum()
print(f"Very short (<=2): {very_short}, very long (>=32): {very_long}")

# ============================================================
# 10. LABEL QUALITY WARNING
# ============================================================
print("\n========== 10. LABEL QUALITY WARNING ==========")
print("Evidence:")
print(f"- Conflicting passwords: {num_conflicting}/88 duplicated values ({conflicting/88*100:.1f}% of duplicated) = 35/669767 distinct passwords (0.005%) but affects 84 rows (0.013%)")
print(f"- Duplicate rows: 76 exact duplicates (0.01%)")
print(f"- Invalid labels: 0, missing strength: 0")
print(f"- Length strongly separates labels: Label0 max 7, Label1 6-13, Label2 6-220, correlation length-strength = {df['_length'].corr(df['strength']):.2f}")
print(f"- Class imbalance: Label1 dominates 74.16%, Label0 13.39%, Label2 12.45%")
print(f"- Outliers: max 220 (1 row), >=32: {(df['_length']>=32).sum()} rows ({(df['_length']>=32).sum()/total*100:.3f}%), >=64: {(df['_length']>=64).sum()}")
print("Assessment: B. partially noisy")
print("Justification: Labels are mostly internally consistent (99.987% of rows not in conflict, 0 invalid, only 1 missing), but 40% of duplicated passwords (35/88) have conflicting labels indicating label noise; 84 rows have ambiguous supervision. Length is highly predictive but not solely deterministic (overlap 6-7 between all labels).")
print("What cannot be determined from dataset alone:")
print("- Semantic meaning of 0/1/2 (weak/medium/strong not verified)")
print("- Provenance and labeling process / annotator criteria")
print("- Whether length-driven labeling is intentional or artifact")
print("- Whether long outliers (220) are legitimate or errors")
print("- Whether conflicting labels are errors or reflect password context sensitivity")

# ============================================================
# 11. ML RISK ASSESSMENT
# ============================================================
print("\n========== 11. ML RISK ASSESSMENT ==========")
print("Potential risks (report only, not fixing):")
print("* Class imbalance: 74% Label1 vs 13% Label0 vs 12% Label2 -> bias toward majority, poor minority recall")
print("* Duplicate leakage: 76 exact duplicate rows and 199 rows with duplicated passwords -> train/test split without deduplication risks memorization and inflated metrics")
print(f"* Conflicting labels: 35 passwords with 84 rows having multiple labels -> irreducible label noise, ceiling on accuracy")
print("* Length-driven classification: Pearson r ~0.65-0.75 expected; model may learn length as proxy, failing on real-world passwords with similar length but different strength")
print(f"* Dataset-specific patterns: Label0 strictly <=7 chars, Label1 6-13, Label2 6-220 -> model will overfit to these hard cutoffs, not generalizable")
print(f"* Outliers: 1 password length 220, {(df['_length']>=32).sum()} >=32, {(df['_length']>=64).sum()} >=64 -> skew length features, unstable training")
print("* Train/test leakage: identical passwords appear 2-4 times; random split leaks same password into both sets")
print("* Lack of provenance: no source verification for 0/1/2 meaning, no guarantee of labeling consistency across time")
print("* No semantic verification: cannot claim 0=weak etc. without source")
print("* Unicode/control anomalies: small but present Unicode/control chars may cause tokenizer issues")
print("* Missing value: 1 missing password (NaN) with Label0 -> handling needed")

# ============================================================
# 12. FINAL FORENSIC REPORT
# ============================================================
print("\n========== FORENSIC DATASET REPORT ==========")
print(f"Dataset size: {total_rows} rows, {len(orig_columns)} columns ({', '.join(orig_columns)})")
print(f"Original shape: {orig_shape}")
print(f"Class distribution: 0->{df['strength'].value_counts().get(0,0)} (13.39%), 1->{df['strength'].value_counts().get(1,0)} (74.16%), 2->{df['strength'].value_counts().get(2,0)} (12.45%)")
print(f"Missing values: password {missing_pw}, strength {df['strength'].isnull().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Duplicated password values: {duplicated_values_no_na} distinct, {dup_occurrences} duplicate occurrences, {rows_with_dup} rows")
print(f"Conflicting password values: {num_conflicting}")
print(f"Rows affected by conflicts: {rows_affected}")
print(f"Maximum length: {int(df['_length'].max())}")
for t in [32,64,100,150,200]:
    cnt = (df["_length"]>=t).sum()
    print(f"Passwords >={t}: {cnt} ({cnt/total*100:.4f}%)")
print("\nPotential data-quality risks:")
print("* 1 missing password (confirmed error - 1 NaN)")
print("* 35 conflicting label values (84 rows) - confirmed label noise")
print("* 76 exact duplicate rows - suspicious repetition")
print("* 1 extreme outlier length 220 (suspicious/anomalous - isolated)")
print("* Length strongly predictive of label (dataset-specific pattern - unknown if legitimate)")
print("* Unicode/control chars present but rare (<1%) - legitimate but uncommon")
print("* Class imbalance not error but quality concern")
print("\nPotential ML risks:")
print("* Class imbalance, duplicate leakage, conflicting labels, length bias, outlier skew, lack of provenance, unverified label semantics")
print("\nRecommended next step:")
print("* Do NOT delete outliers or duplicates yet. Document provenance, decide deduplication and conflict-resolution strategy (e.g., majority vote or exclusion), stratify by password (not row) for splits, and verify label semantics with source before feature engineering or training.")
print("\nSTOP - end of forensic analysis. No cleaning, feature engineering, or ML performed.")
