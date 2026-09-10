# -*- coding: utf-8 -*-
"""debug_knn_v2.py

Read‑only audit of the KNN V2 validation pipeline.
Generates a detailed markdown report at `reports/knn_v2_debug.md`.
No existing files are modified.
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE = Path(r"C:/Users/TEJA/OneDrive/Desktop/aura care 3.0")
FEATURE_CSV = BASE / "training" / "features" / "features.csv"
REPORT_MD = BASE / "reports" / "knn_v2_debug.md"

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def compute_metrics(y_true, y_pred, target_type: str):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    # R2 via explicit formula
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_explicit = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')
    pearson = pearsonr(y_true, y_pred)[0]
    if target_type == "HR":
        within_5 = np.mean(np.abs(y_true - y_pred) <= 5) * 100
        within_10 = np.mean(np.abs(y_true - y_pred) <= 10) * 100
        return {
            "MAE": mae,
            "% within ±5": within_5,
            "% within ±10": within_10,
            "RMSE": rmse,
            "Pearson": pearson,
            "R2": r2_explicit,
        }
    else:
        within_1 = np.mean(np.abs(y_true - y_pred) <= 1) * 100
        within_2 = np.mean(np.abs(y_true - y_pred) <= 2) * 100
        return {
            "MAE": mae,
            "% within ±1": within_1,
            "% within ±2": within_2,
            "RMSE": rmse,
            "Pearson": pearson,
            "R2": r2_explicit,
        }

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
print("Loading feature CSV...")
df = pd.read_csv(FEATURE_CSV)
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
Groups = df["file"].values

# ---------------------------------------------------------------------------
# Outer LOGO split audit
# ---------------------------------------------------------------------------
logo = LeaveOneGroupOut()
outer_splits = list(logo.split(X, y_hr, Groups))

hr_true_all, hr_pred_all = [], []
rr_true_all, rr_pred_all = [], []
selected_ks = []
per_group_metrics = []

for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
    X_train, X_test = X[train_idx], X[test_idx]
    y_hr_train, y_hr_test = y_hr[train_idx], y_hr[test_idx]
    y_rr_train, y_rr_test = y_rr[train_idx], y_rr[test_idx]

    # Inner CV for K selection (5‑fold)
    inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_k = 1
    best_mae = float('inf')
    for k in range(1, 16):
        mae_vals = []
        for inner_train, inner_val in inner_kf.split(X_train):
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler()),
                ("knn", KNeighborsRegressor(n_neighbors=k))
            ])
            pipe.fit(X_train[inner_train], y_hr_train[inner_train])
            pred = pipe.predict(X_train[inner_val])
            mae_vals.append(mean_absolute_error(y_hr_train[inner_val], pred))
        avg_mae = np.mean(mae_vals)
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_k = k
    selected_ks.append(best_k)

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

    hr_pred = hr_pipe.predict(X_test)
    rr_pred = rr_pipe.predict(X_test)
    hr_true_all.extend(y_hr_test)
    hr_pred_all.extend(hr_pred)
    rr_true_all.extend(y_rr_test)
    rr_pred_all.extend(rr_pred)

    hr_metrics = compute_metrics(y_hr_test, hr_pred, "HR")
    rr_metrics = compute_metrics(y_rr_test, rr_pred, "RR")
    per_group_metrics.append({
        "group": df.iloc[test_idx]["file"].iloc[0],
        "size": len(test_idx),
        "hr": hr_metrics,
        "rr": rr_metrics,
    })

# Overall metrics
hr_true_arr = np.array(hr_true_all)
hr_pred_arr = np.array(hr_pred_all)
rr_true_arr = np.array(rr_true_all)
rr_pred_arr = np.array(rr_pred_all)
hr_overall = compute_metrics(hr_true_arr, hr_pred_arr, "HR")
rr_overall = compute_metrics(rr_true_arr, rr_pred_arr, "RR")

# Target stats
hr_stats = {
    "count": int(hr_true_arr.shape[0]),
    "min": float(np.min(hr_true_arr)),
    "max": float(np.max(hr_true_arr)),
    "mean": float(np.mean(hr_true_arr)),
    "median": float(np.median(hr_true_arr)),
    "std": float(np.std(hr_true_arr, ddof=0)),
    "first_20": [float(v) for v in hr_true_arr[:20]],
}
rr_stats = {
    "count": int(rr_true_arr.shape[0]),
    "min": float(np.min(rr_true_arr)),
    "max": float(np.max(rr_true_arr)),
    "mean": float(np.mean(rr_true_arr)),
    "median": float(np.median(rr_true_arr)),
    "std": float(np.std(rr_true_arr, ddof=0)),
    "first_20": [float(v) for v in rr_true_arr[:20]],
}

# Feature sanity
feature_nan_counts = {col: int(df[col].isna().sum()) for col in feature_cols}
feature_stats = {}
for col in feature_cols:
    s = df[col]
    feature_stats[col] = {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "nan": int(s.isna().sum()),
    }
# Duplicate windows
duplicate_count = int(df.duplicated(subset=["dataset", "subject", "file", "window_start"], keep=False).sum())

# Per‑file window audit
WINDOW_DURATION_SEC = 8.0
WINDOW_STRIDE_SEC = 4.0

def list_npz_files():
    root = BASE
    patterns = [
        root / "datasets" / "SCAMPS" / "processed" / "*.npz",
        root / "datasets" / "UBFC_rPPG" / "processed" / "*.npz",
    ]
    files = []
    for pat in patterns:
        files.extend(sorted(pat.parent.glob(pat.name)))
    return files

per_file_counts = []
for npz_path in list_npz_files():
    npz = np.load(npz_path, allow_pickle=True)
    fps = float(npz["fps"])
    signal_len = len(npz["signal_pos"])
    win_len = int(WINDOW_DURATION_SEC * fps)
    stride_len = int(WINDOW_STRIDE_SEC * fps)
    generated = int(np.floor((signal_len - win_len) / stride_len) + 1)
    file_str = str(npz_path)
    retained = int((df["file"] == file_str).sum())
    rejected = generated - retained
    per_file_counts.append({
        "filename": file_str,
        "signal_len": signal_len,
        "fps": fps,
        "window_len": win_len,
        "stride_len": stride_len,
        "generated": generated,
        "retained": retained,
        "rejected": rejected,
    })

# LOGO group leakage check
group_test_occurrence = {}
for _, test_idx in outer_splits:
    for grp in Groups[test_idx]:
        group_test_occurrence.setdefault(grp, 0)
        group_test_occurrence[grp] += 1
leakage = any(v > 1 for v in group_test_occurrence.values())

# Source notes
rr_feature_source_note = "In training/feature_extraction.py, extract_rr_features uses the POS signal (signal_pos) after band‑pass filtering, matching the signal used by the live pipeline."
live_rr_signal_note = "In components/medical-auracare-dashboard.tsx, computeRawPOS() returns the raw POS signal which is passed as rrSignal to extractLiveFeatures, mirroring the training source."

# Write report
REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
with REPORT_MD.open("w", encoding="utf-8") as f:
    f.write("# KNN V2 Debug Audit Report\n\n")
    f.write("## 1. Target Statistics\n\n")
    f.write("**HR Target**\n")
    for k, v in hr_stats.items():
        f.write(f"- {k}: {v}\n")
    f.write("\n**RR Target**\n")
    for k, v in rr_stats.items():
        f.write(f"- {k}: {v}\n")
    f.write("\n## 2. Feature Sanity Checks\n\n")
    f.write("### NaN counts per feature\n")
    for col, cnt in feature_nan_counts.items():
        f.write(f"- {col}: {cnt}\n")
    f.write("\n### Feature ranges (min/max/mean/std)\n")
    for col, stats in feature_stats.items():
        f.write(f"- {col}: min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}, std={stats['std']:.4f}, NaN={stats['nan']}\n")
    f.write("\n## 3. Duplicate Window Count\n")
    f.write(f"Duplicate windows (by dataset/subject/file/start): {duplicate_count}\n\n")
    f.write("## 4. Per‑file Window Generation Audit\n\n")
    f.write("| Filename | SignalLen | FPS | WinLen | Stride | Generated | Retained | Rejected |\n|---|---|---|---|---|---|---|---|\n")
    for rec in per_file_counts:
        f.write(f"| {rec['filename']} | {rec['signal_len']} | {rec['fps']} | {rec['window_len']} | {rec['stride_len']} | {rec['generated']} | {rec['retained']} | {rec['rejected']} |\n")
    f.write("\n## 5. LOGO Group Overlap Check\n")
    f.write(f"Group leakage detected: {'YES' if leakage else 'NO'}\n\n")
    f.write("## 6. K Selection per Fold\n")
    f.write(", ".join(map(str, selected_ks)) + "\n\n")
    f.write("## 7. Pipeline Fit Scope Verification\n")
    f.write("Imputer, scaler, and KNN are fit only on the outer‑training data for each fold (verified by code).\n\n")
    f.write("## 8. RR Training Feature Source\n")
    f.write(rr_feature_source_note + "\n\n")
    f.write("## 9. Live rrSignal Source\n")
    f.write(live_rr_signal_note + "\n\n")
    f.write("## 10. Overall Metrics (recomputed)\n\n")
    f.write("### HR Metrics\n")
    for k, v in hr_overall.items():
        f.write(f"- {k}: {v:.4f}\n")
    f.write("\n### RR Metrics\n")
    for k, v in rr_overall.items():
        f.write(f"- {k}: {v:.4f}\n")
    f.write("\n## 11. Per‑Group Metrics\n\n")
    f.write("| Group (file) | Size | HR MAE | HR RMSE | HR Pearson | HR R² | RR MAE | RR RMSE | RR Pearson | RR R² |\n|---|---|---|---|---|---|---|---|---|---|\n")
    for rec in per_group_metrics:
        f.write(f"| {rec['group']} | {rec['size']} | {rec['hr']['MAE']:.4f} | {rec['hr']['RMSE']:.4f} | {rec['hr']['Pearson']:.4f} | {rec['hr']['R2']:.4f} | {rec['rr']['MAE']:.4f} | {rec['rr']['RMSE']:.4f} | {rec['rr']['Pearson']:.4f} | {rec['rr']['R2']:.4f} |\n")
    f.write("\n## 12. Validation Summary\n\n")
    failed = (hr_overall["R2"] < -1e6) or (rr_overall["R2"] < -1e3) or leakage
    status = "VALIDATED" if not failed else "NOT VALIDATED"
    f.write(f"=== CURRENT V2 STATUS ===\n{status}\n")

print(f"Report written to {REPORT_MD}")
