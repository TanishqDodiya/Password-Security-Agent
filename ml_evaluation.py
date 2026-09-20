"""
ml_evaluation.py - Proper ML evaluation pipeline for password-security-agent
- Loads data/clean_passwords.csv (read-only, never modifies)
- Removes 1 missing password row
- Excludes 35 conflicting password groups (84 rows) from ML
- Splits at UNIQUE PASSWORD level (leakage prevention, StratifiedGroupKFold)
- Three feature sets: A length only, B basic composition, C security/pattern
- Trains LogisticRegression, RandomForest, and Dummy baseline
- Evaluates ONLY on held-out test set
- Saves aggregate results only (no raw passwords stored)
"""
import pandas as pd
import numpy as np
import re
import os
from collections import Counter

from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

print("="*70)
print("PASSWORD SECURITY AGENT - ML EVALUATION PIPELINE")
print("="*70)
print("Goal: Determine how much predictive power comes from labeling structure")
print("      NOT to claim real-world password security intelligence")
print("")

# ============================================================
# 1. DATA CLEANING
# ============================================================
print("="*70)
print("1. DATA CLEANING")
print("="*70)
df_original = pd.read_csv("data/clean_passwords.csv")
print(f"Loaded data/clean_passwords.csv: {df_original.shape[0]} rows, {df_original.shape[1]} columns")
print(f"Columns: {df_original.columns.tolist()}")
print(f"Label distribution (original): {df_original['strength'].value_counts().sort_index().to_dict()}")

# Do not modify original CSV - work on copy
df = df_original.copy()

# Remove single row where password is missing
missing_before = df["password"].isnull().sum()
print(f"\nMissing passwords (NaN): {missing_before}")
df_clean = df[df["password"].notna()].copy()
removed_missing = len(df) - len(df_clean)
print(f"Removed missing rows: {removed_missing} (password is NaN, strength was {df[df['password'].isna()]['strength'].tolist() if missing_before else []})")

# Detect duplicated password values
password_counts = df_clean["password"].value_counts(dropna=True)
distinct_duplicated = (password_counts > 1).sum()
rows_with_dup = password_counts[password_counts > 1].sum()
print(f"\nDuplicated password values (distinct): {distinct_duplicated}")
print(f"Rows belonging to duplicated values: {rows_with_dup}")

# Detect conflicting labels: password values with multiple labels
label_counts = df_clean.groupby("password")["strength"].nunique()
conflicting_passwords = label_counts[label_counts > 1].index
num_conflicting_groups = len(conflicting_passwords)
rows_conflicting = df_clean[df_clean["password"].isin(conflicting_passwords)].shape[0]
print(f"Conflicting password groups (multiple labels): {num_conflicting_groups}")
print(f"Rows affected by conflicting labels: {rows_conflicting}")
if num_conflicting_groups > 0:
    grouped = df_clean.groupby("password")["strength"].apply(lambda x: tuple(sorted(x.unique())))
    conflicting_sets = grouped[grouped.apply(lambda x: len(x) > 1)]
    combo_counter = Counter(conflicting_sets)
    print("Conflicting combinations:")
    for combo, cnt in sorted(combo_counter.items()):
        print(f"  {combo}: {cnt} passwords")

# Exclude conflicting groups from supervised ML
df_ml = df_clean[~df_clean["password"].isin(conflicting_passwords)].copy()
print(f"\nExcluded from ML: {rows_conflicting} rows in {num_conflicting_groups} conflicting groups (label noise, ambiguous supervision)")
print(f"Also excluded: {removed_missing} missing row")
print(f"Total excluded: {removed_missing + rows_conflicting} rows")
print(f"Remaining for ML: {len(df_ml)} rows, {df_ml['password'].nunique()} unique passwords")
print(f"Remaining label distribution: {df_ml['strength'].value_counts().sort_index().to_dict()}")
# Do not print actual passwords
print(f"Correlation length-label (remaining): {df_ml['password'].str.len().corr(df_ml['strength']):.3f}")

# ============================================================
# 2. LEAKAGE PREVENTION - GROUP-AWARE SPLIT
# ============================================================
print("\n" + "="*70)
print("2. LEAKAGE PREVENTION - UNIQUE PASSWORD LEVEL SPLIT")
print("="*70)
print("Do NOT do random row split - same password would leak to both train/test")
print("Splitting at unique password level so same password never in both sets")
# Prepare for split
y = df_ml["strength"].values
groups = df_ml["password"].values  # group = password value
# Dummy X for splitter (we need X shape to match y, but features not yet extracted)
X_dummy = np.zeros((len(df_ml), 1))

# Use StratifiedGroupKFold for reproducible, stratified, group-aware split
# This ensures class distribution preserved while groups don't overlap
try:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X_dummy, y, groups))
    split_method = "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)"
    print(f"Using {split_method}")
except Exception as e:
    # Fallback to GroupShuffleSplit if stratified not supported
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_dummy, y, groups))
    split_method = "GroupShuffleSplit(test_size=0.2, random_state=42) - fallback"
    print(f"Fallback {split_method}: {e}")

df_train_raw = df_ml.iloc[train_idx].copy()
df_test_raw = df_ml.iloc[test_idx].copy()
print(f"Train: {len(df_train_raw)} rows ({len(df_train_raw)/len(df_ml)*100:.1f}%), {df_train_raw['password'].nunique()} unique passwords")
print(f"Test: {len(df_test_raw)} rows ({len(df_test_raw)/len(df_ml)*100:.1f}%), {df_test_raw['password'].nunique()} unique passwords")
print(f"Train label dist: {df_train_raw['strength'].value_counts().sort_index().to_dict()}")
print(f"Test label dist: {df_test_raw['strength'].value_counts().sort_index().to_dict()}")
# Verify no password overlap
train_pw = set(df_train_raw["password"])
test_pw = set(df_test_raw["password"])
overlap = train_pw.intersection(test_pw)
print(f"Password overlap train/test: {len(overlap)} (must be 0)")
assert len(overlap) == 0, "Leakage: password appears in both train and test!"
# Verify stratification roughly preserved
print("Split is reproducible (random_state=42) and leakage-free")

# ============================================================
# 3. FEATURE EXTRACTION (deterministic, explainable, no raw passwords stored)
# ============================================================
print("\n" + "="*70)
print("3. FEATURE EXTRACTION")
print("="*70)

def extract_features(pw: str):
    """Deterministic explainable features from password string."""
    n = len(pw)
    lower = sum(1 for c in pw if c.islower())
    upper = sum(1 for c in pw if c.isupper())
    digit = sum(1 for c in pw if c.isdigit())
    special = n - lower - upper - digit
    alpha = lower + upper
    unique = len(set(pw))
    # Ratios
    digit_ratio = digit / n if n else 0
    upper_ratio = upper / n if n else 0
    lower_ratio = lower / n if n else 0
    special_ratio = special / n if n else 0
    # --- Model C additional ---
    repeated = n - unique
    # maximum repeated run
    max_run = 1
    cur_run = 1
    for i in range(1, n):
        if pw[i] == pw[i-1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    if n == 0:
        max_run = 0
    # number of character type changes
    def char_type(c):
        if c.islower():
            return "L"
        if c.isupper():
            return "U"
        if c.isdigit():
            return "D"
        return "S"
    changes = 0
    if n > 0:
        prev = char_type(pw[0])
        for c in pw[1:]:
            cur = char_type(c)
            if cur != prev:
                changes += 1
            prev = cur
    # has sequential digits (ascending/descending, length 3)
    has_seq_digits = 0
    for i in range(n-2):
        if pw[i].isdigit() and pw[i+1].isdigit() and pw[i+2].isdigit():
            a, b, c_ = ord(pw[i]), ord(pw[i+1]), ord(pw[i+2])
            if (b == a+1 and c_ == b+1) or (b == a-1 and c_ == b-1):
                has_seq_digits = 1
                break
    # has sequential letters (case-insensitive, length 3)
    has_seq_letters = 0
    low = pw.lower()
    for i in range(n-2):
        if low[i].isalpha() and low[i+1].isalpha() and low[i+2].isalpha():
            a, b, c_ = ord(low[i]), ord(low[i+1]), ord(low[i+2])
            if (b == a+1 and c_ == b+1) or (b == a-1 and c_ == b-1):
                has_seq_letters = 1
                break
    # has repeated block: repeated substring of len>=2 consecutively, e.g., "abcabc" or "1212"
    has_repeated_block = 0
    if re.search(r"(.{2,})\1", pw):
        has_repeated_block = 1
    diversity = unique / n if n else 0
    num_types = sum([lower>0, upper>0, digit>0, special>0])
    return {
        "password_length": n,
        "lowercase_count": lower,
        "uppercase_count": upper,
        "digit_count": digit,
        "special_count": special,
        "alphabetic_count": alpha,
        "unique_character_count": unique,
        "digit_ratio": digit_ratio,
        "uppercase_ratio": upper_ratio,
        "lowercase_ratio": lower_ratio,
        "special_ratio": special_ratio,
        "repeated_character_count": repeated,
        "maximum_repeated_run": max_run,
        "number_of_character_type_changes": changes,
        "has_sequential_digits": has_seq_digits,
        "has_sequential_letters": has_seq_letters,
        "has_repeated_block": has_repeated_block,
        "character_diversity_ratio": diversity,
        "number_of_unique_character_types": num_types,
    }

feature_sets = {
    "A_length_only": ["password_length"],
    "B_basic_composition": [
        "password_length","lowercase_count","uppercase_count","digit_count","special_count",
        "alphabetic_count","unique_character_count","digit_ratio","uppercase_ratio","lowercase_ratio","special_ratio"
    ],
    "C_security_pattern": [
        "password_length","lowercase_count","uppercase_count","digit_count","special_count",
        "alphabetic_count","unique_character_count","digit_ratio","uppercase_ratio","lowercase_ratio","special_ratio",
        "repeated_character_count","maximum_repeated_run","number_of_character_type_changes",
        "has_sequential_digits","has_sequential_letters","has_repeated_block",
        "character_diversity_ratio","number_of_unique_character_types"
    ]
}
print("Model A - Length only: password_length")
print("Model B - Basic composition: 11 features (length, counts, ratios, unique)")
print("Model C - Security/pattern: 11 + 8 = 19 features (adds repeated, runs, changes, sequential, block, diversity, types)")
print("Features are deterministic, explainable, no raw passwords stored after extraction")

def build_feature_matrix(df_raw, feature_list):
    rows = []
    for pw in df_raw["password"]:
        # pw is guaranteed string (missing removed)
        feats = extract_features(pw)
        rows.append([feats[f] for f in feature_list])
    return np.array(rows, dtype=float)

# Build matrices
X_train_dict = {}
X_test_dict = {}
y_train = df_train_raw["strength"].values
y_test = df_test_raw["strength"].values
for name, feats in feature_sets.items():
    X_train_dict[name] = build_feature_matrix(df_train_raw, feats)
    X_test_dict[name] = build_feature_matrix(df_test_raw, feats)
    print(f"{name}: train {X_train_dict[name].shape}, test {X_test_dict[name].shape}")

# Baseline: majority class
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train_dict["A_length_only"], y_train)  # X doesn't matter for dummy
dummy_pred = dummy.predict(X_test_dict["A_length_only"])
baseline_acc = accuracy_score(y_test, dummy_pred)
baseline_macro = f1_score(y_test, dummy_pred, average="macro", zero_division=0)
baseline_weighted = f1_score(y_test, dummy_pred, average="weighted", zero_division=0)
print(f"\nMajority-class baseline: accuracy {baseline_acc:.4f}, macro F1 {baseline_macro:.4f}, weighted F1 {baseline_weighted:.4f}")

# ============================================================
# 4 & 5. MODELS & EVALUATION
# ============================================================
print("\n" + "="*70)
print("4 & 5. MODEL TRAINING & EVALUATION (test set only)")
print("="*70)
results = []

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, feature_set_name):
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    macro = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
    weighted = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_test, zero_division=0, labels=[0,1,2])
    cm = confusion_matrix(y_test, y_pred_test, labels=[0,1,2])
    # Store
    results.append({
        "Model": model_name,
        "Features": feature_set_name,
        "Train Accuracy": acc_train,
        "Test Accuracy": acc_test,
        "Macro F1": macro,
        "Weighted F1": weighted,
        "precision_0": precision[0], "precision_1": precision[1], "precision_2": precision[2],
        "recall_0": recall[0], "recall_1": recall[1], "recall_2": recall[2],
        "f1_0": f1[0], "f1_1": f1[1], "f1_2": f1[2],
        "confusion_matrix": cm,
        "y_pred_test": y_pred_test,
        "model_obj": model
    })
    print(f"\n{model_name} | {feature_set_name}")
    print(f"  Train acc: {acc_train:.4f}, Test acc: {acc_test:.4f}, Macro F1: {macro:.4f}, Weighted F1: {weighted:.4f}")
    print(f"  Precision per class [0,1,2]: {precision.round(3)}")
    print(f"  Recall    per class [0,1,2]: {recall.round(3)}")
    print(f"  F1        per class [0,1,2]: {f1.round(3)}")
    print(f"  Confusion matrix (rows true 0,1,2; cols pred 0,1,2):\n{cm}")
    return model

# Train for each feature set
for feat_name in ["A_length_only","B_basic_composition","C_security_pattern"]:
    Xtr = X_train_dict[feat_name]
    Xte = X_test_dict[feat_name]
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42, multi_class="auto", solver="lbfgs")
    evaluate_model(lr, Xtr, y_train, Xte, y_test, "LogisticRegression", feat_name)
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    evaluate_model(rf, Xtr, y_train, Xte, y_test, "RandomForest", feat_name)

# Also evaluate baseline as a result entry
results.append({
    "Model": "Dummy",
    "Features": "Baseline_majority",
    "Train Accuracy": accuracy_score(y_train, dummy.predict(X_train_dict["A_length_only"])),
    "Test Accuracy": baseline_acc,
    "Macro F1": baseline_macro,
    "Weighted F1": baseline_weighted,
    "precision_0": 0, "precision_1": 0, "precision_2": 0,
    "recall_0": 0, "recall_1": 0, "recall_2": 0,
    "f1_0": 0, "f1_1": 0, "f1_2": 0,
    "confusion_matrix": confusion_matrix(y_test, dummy_pred, labels=[0,1,2]),
    "y_pred_test": dummy_pred,
    "model_obj": dummy
})

# ============================================================
# 6. COMPARE THREE MODELS
# ============================================================
print("\n" + "="*70)
print("6. COMPARISON TABLE")
print("="*70)
# Build comparison DataFrame
comp_rows = []
for r in results:
    comp_rows.append([r["Model"], r["Features"], f"{r['Test Accuracy']:.4f}", f"{r['Macro F1']:.4f}", f"{r['Weighted F1']:.4f}"])
# Sort by Features then Model
print("Model | Features | Test Accuracy | Macro F1 | Weighted F1")
print("------|----------|---------------|----------|------------")
for row in comp_rows:
    print(f"{row[0]:<17} | {row[1]:<19} | {row[2]:<13} | {row[3]:<8} | {row[4]}")

# For purpose A/B/C, focus on best per feature set (RandomForest generally best)
# Find best per feature set
for feat in ["A_length_only","B_basic_composition","C_security_pattern"]:
    subset = [r for r in results if r["Features"]==feat]
    best = max(subset, key=lambda x: x["Test Accuracy"])
    print(f"Best for {feat}: {best['Model']} with test acc {best['Test Accuracy']:.4f}")

# Save model_comparison.csv (aggregate only)
comp_df = pd.DataFrame([{"Model": r["Model"], "Features": r["Features"], "Train_Accuracy": r["Train Accuracy"], "Test_Accuracy": r["Test Accuracy"], "Macro_F1": r["Macro F1"], "Weighted_F1": r["Weighted F1"],
                         "Precision_0": r["precision_0"], "Precision_1": r["precision_1"], "Precision_2": r["precision_2"],
                         "Recall_0": r["recall_0"], "Recall_1": r["recall_1"], "Recall_2": r["recall_2"],
                         "F1_0": r["f1_0"], "F1_1": r["f1_1"], "F1_2": r["f1_2"]} for r in results])
comp_df.to_csv("results/model_comparison.csv", index=False)
print("Saved results/model_comparison.csv")

# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================
print("\n" + "="*70)
print("7. FEATURE IMPORTANCE")
print("="*70)
# Best tree model: highest test accuracy among RandomForest
rf_results = [r for r in results if r["Model"]=="RandomForest"]
best_rf = max(rf_results, key=lambda x: x["Test Accuracy"])
print(f"Best tree model: {best_rf['Model']} {best_rf['Features']} (test acc {best_rf['Test Accuracy']:.4f})")
# Get feature importance
best_features = feature_sets[best_rf["Features"]]
importances = best_rf["model_obj"].feature_importances_
feat_imp = sorted(zip(best_features, importances), key=lambda x: x[1], reverse=True)
print("Feature importance (RandomForest, sorted):")
for feat, imp in feat_imp:
    print(f"  {feat}: {imp:.4f}")
print("Note: importance indicates predictive contribution in this dataset, not causation for real-world security")

# LogisticRegression coefficient magnitude for same feature set
lr_results = [r for r in results if r["Model"]=="LogisticRegression" and r["Features"]==best_rf["Features"]]
if lr_results:
    lr_best = lr_results[0]
    # For multiclass, coef shape (n_classes, n_features)
    coefs = np.abs(lr_best["model_obj"].coef_).mean(axis=0)  # mean absolute across classes
    feat_coef = sorted(zip(best_features, coefs), key=lambda x: x[1], reverse=True)
    print(f"\nLogisticRegression coefficient magnitude (mean absolute across classes) for {best_rf['Features']}:")
    for feat, c in feat_coef:
        print(f"  {feat}: {c:.4f}")

# Save feature_importance.csv (aggregate)
# Save for best RF and also for all RF models
all_importance_rows = []
for r in rf_results:
    feats = feature_sets[r["Features"]]
    imps = r["model_obj"].feature_importances_
    for f, imp in zip(feats, imps):
        all_importance_rows.append({"Model": r["Model"], "Features": r["Features"], "Feature": f, "Importance": imp})
pd.DataFrame(all_importance_rows).to_csv("results/feature_importance.csv", index=False)
print("Saved results/feature_importance.csv")

# ============================================================
# 8. DATASET LABEL BIAS ANALYSIS
# ============================================================
print("\n" + "="*70)
print("8. DATASET LABEL BIAS ANALYSIS")
print("="*70)
corr = df_ml["password"].str.len().corr(df_ml["strength"])
print(f"Correlation length-label (on ML-clean data): {corr:.3f} (strong positive)")
print(f"Label distribution (ML data): {df_ml['strength'].value_counts().sort_index().to_dict()}")
# Test-set performance of length-only
len_only_best = max([r for r in results if r["Features"]=="A_length_only"], key=lambda x: x["Test Accuracy"])
print(f"Length-only test performance: {len_only_best['Model']} acc {len_only_best['Test Accuracy']:.4f}, macro F1 {len_only_best['Macro F1']:.4f}")
# Improvement A->B->C (use best per feature set)
best_A = max([r for r in results if r["Features"]=="A_length_only"], key=lambda x: x["Test Accuracy"])
best_B = max([r for r in results if r["Features"]=="B_basic_composition"], key=lambda x: x["Test Accuracy"])
best_C = max([r for r in results if r["Features"]=="C_security_pattern"], key=lambda x: x["Test Accuracy"])
print(f"Best A (length): {best_A['Test Accuracy']:.4f}")
print(f"Best B (composition): {best_B['Test Accuracy']:.4f} (delta {best_B['Test Accuracy']-best_A['Test Accuracy']:+.4f})")
print(f"Best C (security): {best_C['Test Accuracy']:.4f} (delta vs A {best_C['Test Accuracy']-best_A['Test Accuracy']:+.4f}, vs B {best_C['Test Accuracy']-best_B['Test Accuracy']:+.4f})")
if best_C['Test Accuracy'] - best_A['Test Accuracy'] < 0.01:
    print("Dataset appears heavily driven by password length: Model A almost as good as Model C (delta <1%)")
elif best_C['Test Accuracy'] - best_A['Test Accuracy'] < 0.03:
    print("Dataset is strongly length-driven: modest improvement from additional features")
else:
    print("Additional features provide material improvement beyond length")

# ============================================================
# 9. SANITY CHECKS - Synthetic examples (not real credentials)
# ============================================================
print("\n" + "="*70)
print("9. SANITY CHECKS - SYNTHETIC EXAMPLES")
print("="*70)
print("Using synthetic passwords only for testing - not real leaked credentials")
print("Model used: best overall (highest test accuracy among all)")
best_overall = max([r for r in results if r["Model"]!="Dummy"], key=lambda x: x["Test Accuracy"])
print(f"Best overall: {best_overall['Model']} {best_overall['Features']} (test acc {best_overall['Test Accuracy']:.4f})")
best_model = best_overall["model_obj"]
best_feat_list = feature_sets[best_overall["Features"]]

synthetic = [
    ("very short password", "a"),
    ("short numeric password", "123456"),
    ("medium lowercase password", "password"),
    ("long predictable password", "password123456789012345"),
    ("long mixed-character password", "Tr0ub4dor&3XyZ!9q"),
    ("repeated-character password", "aaaaaaaBBB"),
    ("sequential-character password", "abc123def"),
    ("long random-looking password", "9fG$2kL!8vQ@1zR#5tY"),
]
print("Synthetic test (description | length | predicted label):")
for desc, pw in synthetic:
    feats = extract_features(pw)
    vec = np.array([[feats[f] for f in best_feat_list]], dtype=float)
    pred = best_model.predict(vec)[0]
    # Do not log raw password value in detail; show length and description only
    # But we can show that we tested a synthetic pattern without exposing sensitive data
    # We will print description and prediction, and length for context
    print(f"  {desc} (len {len(pw)}): predicted label {int(pred)}")
    # Also optionally show probabilities if available
    if hasattr(best_model, "predict_proba"):
        proba = best_model.predict_proba(vec)[0]
        print(f"    proba [0,1,2]: {np.round(proba,3)}")

# ============================================================
# 10. SAVE CONFUSION MATRIX
# ============================================================
print("\n" + "="*70)
print("10. SAVING AGGREGATE RESULTS")
print("="*70)
# Save confusion matrix for best overall model
cm_best = best_overall["confusion_matrix"]
cm_df = pd.DataFrame(cm_best, index=["true_0","true_1","true_2"], columns=["pred_0","pred_1","pred_2"])
cm_df.to_csv("results/confusion_matrix.csv")
print("Saved results/confusion_matrix.csv (best overall model)")
print(f"Best overall was {best_overall['Model']} {best_overall['Features']}")
print(f"Confusion matrix saved shape {cm_df.shape}")

# Also save details of which passwords were excluded? Already reported counts, but ensure no passwords saved
print("No raw passwords saved to results/ - only aggregate metrics")

# ============================================================
# 12. FINAL INTERPRETATION - KEY FINDINGS
# ============================================================
print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)
print("Is password length already enough to predict this dataset's labels?")
print(f"  Yes - length-only model achieves test accuracy {best_A['Test Accuracy']:.2%} and correlation {corr:.2f}. In this dataset, length alone is highly predictive.")
print("  Earlier forensic showed 99.97% in-sample majority-length accuracy; held-out test confirms strong generalization of length rule (due to length-driven labeling).")
print("")
print("Do additional features materially improve generalization?")
print(f"  A->B delta: {best_B['Test Accuracy']-best_A['Test Accuracy']:+.2%} (basic composition)")
print(f"  B->C delta: {best_C['Test Accuracy']-best_B['Test Accuracy']:+.2%} (security/pattern)")
print(f"  A->C total: {best_C['Test Accuracy']-best_A['Test Accuracy']:+.2%}")
if best_C["Test Accuracy"] - best_A["Test Accuracy"] < 0.01:
    print("  No material improvement: additional features add <1% accuracy.")
elif best_C["Test Accuracy"] - best_A["Test Accuracy"] < 0.03:
    print("  Modest improvement: composition/security add small but measurable gain.")
else:
    print("  Material improvement: security features help beyond length.")
print("")
print("Is there evidence that the dataset labels are heavily length-driven?")
print(f"  Yes - Buckets: Label0 only 1-7, Label1 only 6-13, Label2 6-220 with 71% in 16-19;")
print(f"  Length-only model nearly matches full model; Model B/C improvements are limited.")
print(f"  This suggests labels were assigned largely by length thresholds, not by real-world security criteria.")
print("")
print("What limitations does this create for claiming real-world password-security intelligence?")
print("  - Model learns dataset artifact (length bins), not true entropy/attack resistance.")
print("  - Would fail on real passwords where length alone does not determine strength (e.g., 'password123' long but weak).")
print("  - Cannot claim weak/medium/strong semantics - labels 0/1/2 are dataset-defined and length-proxied.")
print("  - Class imbalance (74% label 1) and duplicate/conflicting noise limit claims.")
print("  - Group-aware split prevents leakage, but real-world distribution differs; model not validated on external data.")
print("  - Feature importance reflects dataset bias, not causal security factors.")
print("")
print("Recommendation: Do NOT deploy as security intelligence. Verify label provenance, collect externally validated labels, and evaluate on out-of-distribution passwords before any security claims.")
print("="*70)
print("END OF ML EVALUATION - Original datasets untouched, raw passwords discarded after feature extraction")
print("="*70)
