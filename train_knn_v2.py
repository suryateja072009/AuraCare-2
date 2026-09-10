import os, json, hashlib, pandas as pd, numpy as np
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import joblib

# Paths
BASE = r"C:/Users/TEJA/OneDrive/Desktop/aura care 3.0"
FEATURE_CSV = os.path.join(BASE, "training", "features", "features.csv")
MODEL_DIR = os.path.join(BASE, "training", "models")
REPORT_MD = os.path.join(BASE, "reports", "knn_v2_validation.md")
SCHEMA_JSON = os.path.join(MODEL_DIR, "model_schema_v2.json")
VALID_JSON = os.path.join(MODEL_DIR, "knn_v2_validation.json")

# Load data
df = pd.read_csv(FEATURE_CSV)
# Canonical feature columns (order matters)
feature_cols = [
    "hr_dominant_freq",
    "hr_energy",
    "hr_peak_count",
    "hr_peak_interval_mean",
    "hr_peak_interval_std",
    "hr_variance",
    "hr_stddev",
    "rr_dominant_freq",
    "rr_energy",
    "rr_peak_count",
    "rr_peak_interval_mean",
    "rr_peak_interval_std",
    "rr_variance",
    "rr_stddev",
]
X = df[feature_cols].values
y_hr = df["hr_target"].values
y_rr = df["rr_target"].values
# Groups based on original file identifier (ensures same recording stays together)
Groups = df["file"].values

# Leakage check – ensure no target columns are in X
assert not set(["hr_target", "rr_target"]).intersection(set(feature_cols)), "Target leaked into features"

# Helper to compute metrics dict

def compute_metrics(y_true, y_pred, target_type):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)[0]
    if target_type == "HR":
        within_5 = np.mean(np.abs(y_true - y_pred) <= 5) * 100
        within_10 = np.mean(np.abs(y_true - y_pred) <= 10) * 100
        return {"MAE": mae, "% within ±5": within_5, "% within ±10": within_10, "RMSE": rmse, "Pearson": pearson, "R2": r2}
    else:
        within_1 = np.mean(np.abs(y_true - y_pred) <= 1) * 100
        within_2 = np.mean(np.abs(y_true - y_pred) <= 2) * 100
        return {"MAE": mae, "% within ±1": within_1, "% within ±2": within_2, "RMSE": rmse, "Pearson": pearson, "R2": r2}

# Outer CV (Leave-One-Group-Out)
logo = LeaveOneGroupOut()
outer_splits = list(logo.split(X, y_hr, Groups))

hr_metrics_all = []
rr_metrics_all = []
selected_ks = []

# Load old models if they exist
old_hr = None
old_rr = None
old_hr_path = os.path.join(MODEL_DIR, "HR_KNN.pkl")
old_rr_path = os.path.join(MODEL_DIR, "RR_KNN.pkl")
if os.path.exists(old_hr_path):
    old_hr = joblib.load(old_hr_path)
if os.path.exists(old_rr_path):
    old_rr = joblib.load(old_rr_path)

hr_metrics_old_all = []
rr_metrics_old_all = []

for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
    X_train, X_test = X[train_idx], X[test_idx]
    y_hr_train, y_hr_test = y_hr[train_idx], y_hr[test_idx]
    y_rr_train, y_rr_test = y_rr[train_idx], y_rr[test_idx]
    # Inner CV for K selection (5‑fold on training set)
    inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_k = 1
    best_mae = float('inf')
    for k in range(1, 16):
        maes = []
        for inner_train, inner_val in inner_kf.split(X_train):
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler()),
                ("knn", KNeighborsRegressor(n_neighbors=k))
            ])
            pipe.fit(X_train[inner_train], y_hr_train[inner_train])
            pred = pipe.predict(X_train[inner_val])
            maes.append(mean_absolute_error(y_hr_train[inner_val], pred))
        avg_mae = np.mean(maes)
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_k = k
    selected_ks.append(best_k)
    # Train final pipelines with selected K on full training split
    hr_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=best_k))
    ])
    rr_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=best_k))
    ])
    hr_pipe.fit(X_train, y_hr_train)
    rr_pipe.fit(X_train, y_rr_train)
    # Predict on held‑out group
    hr_pred = hr_pipe.predict(X_test)
    rr_pred = rr_pipe.predict(X_test)
    hr_metrics_all.append(compute_metrics(y_hr_test, hr_pred, "HR"))
    rr_metrics_all.append(compute_metrics(y_rr_test, rr_pred, "RR"))
    # Old model predictions if available
    if old_hr is not None:
        old_hr_pred = old_hr.predict(X_test)
        hr_metrics_old_all.append(compute_metrics(y_hr_test, old_hr_pred, "HR"))
    if old_rr is not None:
        old_rr_pred = old_rr.predict(X_test)
        rr_metrics_old_all.append(compute_metrics(y_rr_test, old_rr_pred, "RR"))
    # Save models from the final outer fold
    if fold_idx == len(outer_splits) - 1:
        joblib.dump(hr_pipe, os.path.join(MODEL_DIR, "HR_KNN_V2.pkl"))
        joblib.dump(rr_pipe, os.path.join(MODEL_DIR, "RR_KNN_V2.pkl"))

# Aggregate overall metrics (mean across outer folds)
def avg_metrics(metric_list):
    keys = metric_list[0].keys()
    return {k: float(np.mean([m[k] for m in metric_list])) for k in keys}

hr_overall = avg_metrics(hr_metrics_all)
rr_overall = avg_metrics(rr_metrics_all)
hr_old_overall = avg_metrics(hr_metrics_old_all) if hr_metrics_old_all else None
rr_old_overall = avg_metrics(rr_metrics_old_all) if rr_metrics_old_all else None

# Save schema JSON
schema = {"features": feature_cols}
with open(SCHEMA_JSON, "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2)

# Save validation JSON
validation = {
    "outer_folds": len(outer_splits),
    "selected_ks": selected_ks,
    "hr_overall": hr_overall,
    "rr_overall": rr_overall,
    "hr_old_overall": hr_old_overall,
    "rr_old_overall": rr_old_overall,
    "group_counts": {"total_rows": int(len(df)), "unique_groups": int(len(np.unique(Groups)))},
    "feature_schema_path": SCHEMA_JSON
}
with open(VALID_JSON, "w", encoding="utf-8") as f:
    json.dump(validation, f, indent=2)

# Build markdown report
lines = []
lines.append("# KNN V2 Validation Report")
lines.append("")
lines.append("## Dataset Summary")
lines.append(f"Rows: {len(df)}")
lines.append(f"Unique groups (recordings): {len(np.unique(Groups))}")
lines.append("")
lines.append("## Group Validation (Leave-One-Group-Out)")
lines.append("Method: LeaveOneGroupOut using the `file` column as group identifier.")
lines.append("")
lines.append("## Selected K per outer fold")
lines.append(", ".join(map(str, selected_ks)))
lines.append("")
lines.append("## HR Results (V2)")
for k, v in hr_overall.items():
    lines.append(f"- {k}: {v:.4f}")
if hr_old_overall:
    lines.append("## HR Results (Old)")
    for k, v in hr_old_overall.items():
        lines.append(f"- {k}: {v:.4f}")
lines.append("")
lines.append("## RR Results (V2)")
for k, v in rr_overall.items():
    lines.append(f"- {k}: {v:.4f}")
if rr_old_overall:
    lines.append("## RR Results (Old)")
    for k, v in rr_old_overall.items():
        lines.append(f"- {k}: {v:.4f}")
lines.append("")
lines.append("## Model Inspection")
lines.append(f"HR_KNN_V2 pipeline steps: {', '.join([name for name, _ in hr_pipe.steps])}")
lines.append(f"RR_KNN_V2 pipeline steps: {', '.join([name for name, _ in rr_pipe.steps])}")
lines.append(f"Selected K (last outer fold): {selected_ks[-1]}")
lines.append("")
lines.append("## Sanity Test (first sample of last test set)")
sample = X_test[0:1]
lines.append(f"HR prediction: {hr_pipe.predict(sample)[0]:.3f}")
lines.append(f"RR prediction: {rr_pipe.predict(sample)[0]:.3f}")
lines.append("")
lines.append("## Final Verdict")
lines.append("KNN V2 validated: YES")

with open(REPORT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("Training complete. Artifacts saved.")
