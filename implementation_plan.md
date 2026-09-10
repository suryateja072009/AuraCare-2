# Implementation Plan for Updated Dataset Pipeline

**Goal**: Re‑architect the AuraCare 3.0 backend to work with the three target datasets (SCAMPS, UBFC‑rPPG, VIPL‑HR) and provide a full CPU‑friendly training / evaluation pipeline while keeping the UI untouched.

### 1. Folder structure changes
- Add raw/ sub‑folders for `datasets/UBFC_rPPG/` and `datasets/VIPL_HR/` (already created placeholders).
- Remove any references to `PURE`, `DIL_RR`, and `COHFACE` from scripts.
- Expected final layout:
```
datasets/
  SCAMPS/
    raw/
    processed/
  UBFC_rPPG/
    raw/
    processed/
  VIPL_HR/
    raw/
    processed/
```

### 2. Pre‑processing (`training/preprocess.py`)
- Keep MediaPipe Face‑Mesh forehead ROI detection.
- Extract **mean RGB** per frame.
- Compute **POS** signal (existing logic) **and** **CHROM** signal (new function).
- Apply Butterworth band‑pass (0.75‑3.5 Hz) to both signals.
- Compute a **simple signal‑quality score** (e.g., variance‑to‑mean ratio of the filtered POS signal).
- Load optional ground‑truth files:
  - `<basename>.hr.txt` for heart‑rate (BPM)
  - `<basename>.rr.txt` for respiratory‑rate (breaths per minute)
- Save to a `.npz` with arrays:
  - `signal_pos`
  - `signal_chrom`
  - `hr_gt`
  - `rr_gt`
  - `quality_score`
- Loop over the three datasets (SCAMPS, UBFC_rPPG, VIPL_HR) only.

### 3. Feature extraction (`training/features.py`)
- Provide a single function `extract_features(npz_path, use_chrom=False)` that loads the `.npz` and computes the listed statistical / spectral features:
  - dominant frequency (via periodogram, restrict to HR band 0.75‑3.5 Hz for HR, respiratory band 0.1‑0.5 Hz for RR)
  - spectral peak power
  - variance, standard deviation, RMS
  - peak count (using `scipy.signal.find_peaks`)
  - average peak interval
  - signal energy (sum of PSD)
  - entropy (Shannon on normalized PSD)
  - skewness, kurtosis (from `scipy.stats`)
- Return a feature vector (list of floats).

### 4. Model training scripts
- **`training/train_hr_model.py`**
  - Load processed data via `features.extract_features(..., use_chrom=False)` for HR.
  - Train two models:
    - Primary: `RandomForestRegressor`
    - Comparison: `KNeighborsRegressor`
  - Save both models under `models/hr_model/` as `rf.pkl` and `knn.pkl`.
- **`training/train_rr_model.py`** (new, analogous)
  - Use the same feature extractor but with respiratory‑band settings.
  - Train RF and KNN, save under `models/rr_model/`.

### 5. Evaluation (`training/evaluate.py`)
- Load the four models (HR‑RF, HR‑KNN, RR‑RF, RR‑KNN).
- For each dataset, predict HR and RR, compute MAE, RMSE, Pearson correlation for **both** models.
- Write a markdown report `reports/accuracy_report.md` with per‑dataset tables and overall averages.
- Include a short note if the target correlations (>0.70) are met.

### 6. Dataset checker (`training/check_datasets.py`)
- Update the constant list to `['SCAMPS', 'UBFC_rPPG', 'VIPL_HR']`.
- For each dataset, report:
  - Raw video count
  - Processed `.npz` count
  - Ready status (`YES` if processed count > 0 and matches raw count, else `NO`).
- Print a nice table.

### 7. Documentation updates
- Overwrite `docs/dataset_setup.md` (already updated) to reference only the three datasets.
- Overwrite `docs/download_links.md` with official download URLs for SCAMPS, UBFC‑rPPG, VIPL‑HR.
- **Create** `docs/download_guide.md` – explains where to download each dataset, storage needed, which subset to start with, and expected extraction layout.
- **Create** `docs/dataset_structure.md` – visual description of the folder hierarchy and the contents of each `.npz` file.

### 8. Artifact generation
- All new files will be placed under the existing workspace (`c:\Users\TEJA\OneDrive\Desktop\aura care 3.0`).
- No modifications to `app/`, `components/`, `pages/`, `package.json`, or any TypeScript/React files.

### 9. Verification plan
- **Unit‑style checks** (via simple script runs):
  - Run `python training/check_datasets.py` – expect three rows, all folders present.
  - Run `python training/preprocess.py` on a small test video (user provides) – verify `.npz` contains the five arrays.
  - Run `python training/train_hr_model.py` and `train_rr_model.py` – ensure model files appear.
  - Run `python training/evaluate.py` – ensure `reports/accuracy_report.md` is generated and contains numeric values.
- Manual sanity check: open a generated `.npz` with NumPy to see arrays.

### Open questions / user review required
> **[IMPORTANT]** Do you want any specific **subset sizes** (e.g., first N videos) for each dataset to be hard‑coded now, or should the scripts process *all* videos present?  
> **[IMPORTANT]** For the signal‑quality score, do you prefer a specific metric (e.g., SNR estimate) or is the variance‑to‑mean ratio acceptable?  
> **[IMPORTANT]** Should the CHROM signal be saved alongside POS for every video, or only when a `--chrom` flag is passed? (Current plan: always compute and store).

---
**User Review Required**
- Approve the overall plan and answer the open questions above.
