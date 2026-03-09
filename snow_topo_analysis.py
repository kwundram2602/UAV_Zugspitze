"""
snow_topo_analysis.py
=====================
Statistical analysis of snow depth vs. topographic parameters.

Workflow:
  1. Load snow depth model + topographic indices (from topo_indices.py output)
  2. Build a random pixel sample (stratified by snow depth quantile)
  3. Fit Random Forest  → feature importance, partial dependence
  4. Export plots and a summary CSV

Prerequisite outputs:
    out/snow_depth_model.tif
    out/indices/slope.tif
    out/indices/tpi_50px.tif           (5 m)
    out/indices/tpi_250px.tif          (25 m)
    out/indices/tpi_500px.tif          (50 m)
    out/indices/windward_index_SW.tif  (wind from 225°)
    out/indices/windward_index_W.tif   (wind from 270°)
    out/indices/windward_index_E.tif   (wind from  90°)
    out/indices/windward_index_SE.tif  (wind from 135°)

Analysis results are written to   out/analysis/

Run topo_indices.py first if the index rasters are missing.
"""
#%%BASICS
import os
import numpy as np
import pandas as pd
import rioxarray as riox
import matplotlib
matplotlib.use("Agg")           # headless; change to "TkAgg" if you want a window
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance
from pathlib import Path


# ─── Configuration ────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
OUT_DIR     = HERE / "out"
INDICES_DIR = os.path.join(OUT_DIR, "indices")    # rasters from topo_indices.py
RESULTS_DIR = os.path.join(OUT_DIR, "analysis")   # plots + CSV from this script

# All TPI scales to include as separate features
TPI_RADII_PX = [50, 250]   # → 5 m, 25 m, 50 m at 10 cm resolution

# Wind directions to include (must match labels from topo_indices.py)
WIND_DIRECTIONS = {"SW": 225, "W": 270}

RELIEFS = ["E", "SE", "SW", "W"]

# Sampling
N_SAMPLE    = 100_000   # max pixels drawn for modelling
RANDOM_SEED = 42

# Snow depth filter (remove outliers / artefacts)
SNOW_MIN_M = 0.05   # below this → likely measurement noise, excluded
SNOW_MAX_M = 6.00   # above this → likely DSM artefact, excluded

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_raster(path: str) -> np.ndarray:
    """Load a single-band raster by full path, return 2-D array (NaN = nodata)."""
    da = riox.open_rasterio(path, masked=True).squeeze()
    return da.values


def print_section(title: str):
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")

os.makedirs(RESULTS_DIR, exist_ok=True)

#%% ─── 1. Load rasters ──────────────────────────────────────────────────────────
print_section("Loading rasters")

snow  = load_raster(os.path.join(OUT_DIR,     "snow_depth_model.tif"))
slope = load_raster(os.path.join(INDICES_DIR, "slope.tif"))
curve = load_raster(os.path.join(INDICES_DIR, "curvature_output.tif"))
tri = load_raster(os.path.join(INDICES_DIR, "ruggedness_tri.tif"))
geomorph = load_raster(os.path.join(INDICES_DIR, "geomorphons.tif"))

tpi_arrays = {}
for r in TPI_RADII_PX:
    key = f"tpi_{r}px"
    tpi_arrays[key] = load_raster(os.path.join(INDICES_DIR, f"{key}.tif"))
    print(f"  Loaded indices/{key}.tif")

wi_arrays = {}
for label in WIND_DIRECTIONS:
    key = f"wi_{label}"
    wi_arrays[key] = load_raster(os.path.join(INDICES_DIR, f"windward_index_{label}.tif"))
    print(f"  Loaded indices/windward_index_{label}.tif")

relief_arrays = {}
for rel in RELIEFS:
    key = f"dir_rel_{rel}"
    relief_arrays[key] = load_raster(os.path.join(INDICES_DIR, f"dir_relief_{rel}.tif"))
    print(f"  Loaded indices/dir_relief_{rel}.tif")

print(f"  Array shape  : {snow.shape}")
print(f"  Total pixels : {snow.size:,}")

#%% ─── 2. Build sample DataFrame ────────────────────────────────────────────────
print_section("Building sample")

#use .ravel() to convert multidimensional arrays into one single dimension
df_dict = {
    "snow_depth": snow.ravel(),
    "slope": slope.ravel(),
    "curve": curve.ravel(),
    "tri": tri.ravel(),
    "geomorph": geomorph.ravel()
}

for key, arr in tpi_arrays.items():
    df_dict[key] = arr.ravel()
for key, arr in wi_arrays.items():
    df_dict[key] = arr.ravel()
for key, arr in relief_arrays.items():
    df_dict[key] = arr.ravel()


df = pd.DataFrame(df_dict)

#feature engineering
print("  Engineering interaction features...")
for label in WIND_DIRECTIONS:
    key = f"wi_{label}"
    inter_key = f"slope_x_{key}"
    # Calculate interaction: Slope * Windward Index
    df[inter_key] = df["slope"] * df[key]
    

FEATURES = (
    ["slope"]
    + ["curve"]
    + ["tri"]
    + ["geomorph"]
    + list(tpi_arrays.keys())
    + list(wi_arrays.keys())
    + list(relief_arrays.keys())
    + [f"slope_x_wi_{l}" for l in WIND_DIRECTIONS]  # New interaction features
)

#create dictionary with key-value pairs for all feature labels
FEATURE_LABELS = {
    "slope": "Slope (°)",
    "curve": "Curvature",
    "tri": "Terrain Ruggedness Index",
    "geomorph": "Geomorphons"
}

for r in TPI_RADII_PX:
    FEATURE_LABELS[f"tpi_{r}px"] = f"TPI {r}px ({r*0.10:.0f} m)"

for label, deg in WIND_DIRECTIONS.items():
    # Das Basis-Feature (wi_SW, wi_W, etc.)
    base_key = f"wi_{label}"
    FEATURE_LABELS[base_key] = f"WI {label} ({deg}°)"
    
    # Das Interaktions-Feature (slope_x_wi_SW, etc.)
    # WICHTIG: Prüfe hier genau, ob du oben "slope_x_wi_" oder "slope_x_" geschrieben hast!
    inter_key = f"slope_x_wi_{label}" 
    FEATURE_LABELS[inter_key] = f"Slope x WI {label}"

for rel in RELIEFS:
    FEATURE_LABELS[f"dir_rel_{rel}"] = f"Directional Relief from {rel}"

feat_labels = [FEATURE_LABELS[f] for f in FEATURES]

#filter df_valid with upper and lower limits
df_valid = df.dropna()
df_valid = df_valid[
    (df_valid["snow_depth"] >= SNOW_MIN_M) &
    (df_valid["snow_depth"] <= SNOW_MAX_M)
]
print(f"  Valid pixels after filter : {len(df_valid):,}")

# Stratified sample: equal number of rows per snow-depth decile
#   → ensures rare deep-snow pixels are represented
df_valid["_decile"] = pd.qcut(df_valid["snow_depth"], q=10, labels=False, duplicates="drop")
per_bin = max(1, N_SAMPLE // df_valid["_decile"].nunique())  # rows to draw per decile
df_sample = (
    df_valid
    .groupby("_decile", group_keys=False)
    .apply(lambda g: g.sample(min(per_bin, len(g)), random_state=RANDOM_SEED))
    .reset_index(drop=True)
)
print(f"  Stratified sample size    : {len(df_sample):,}")


#convert and assign df_sample in to a NumpyArray via .values
X = df_sample[FEATURES].values
y = df_sample["snow_depth"].values

#create test and train dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

#%% ─── 3. Hist Gradient Boosting ──────────────────────────────────────────────
print_section("Section: Hist Gradient Boosting")

#initialize model
hgbr = HistGradientBoostingRegressor(random_state=RANDOM_SEED)

param_grid = {
    'max_iter': [300, 500, 800, 1000],    # Anzahl der Bäume
    'learning_rate': [0.01, 0.05, 0.1],   # Schrittweite (kleiner = stabiler, aber braucht mehr Bäume)
    'max_depth': [5, 10, 20],             # Tiefe der Bäume (Komplexität der Interaktionen)
    'l2_regularization': [0.0, 0.1, 1.0], # Verhindert Overfitting auf Rauschen im DSM
    'min_samples_leaf': [20, 50, 100]     # Wichtig bei 10cm Auflösung wegen Messrauschen
}

print("Hyperparameter Tuning")

grid_search = GridSearchCV(
    estimator=hgbr,
    param_grid=param_grid,
    cv=5, 
    scoring='r2', # Wir optimieren auf das Bestimmtheitsmaß
    n_jobs=-1,    # Nutze alle verfügbaren CPU-Kerne (wichtig auf dem Cluster!)
    verbose=1
)

grid_search.fit(X_train, y_train)

# Bestes Modell extrahieren
hgb_model = grid_search.best_estimator_

print(f"  Best Parameters: {grid_search.best_params_}")
print(f"  Best CV R² Score: {grid_search.best_score_:.4f}")


# 3. Predict and evaluate
y_pred_hgb = hgb_model.predict(X_test)

# 4. Calculate performance metrics
r2_hgb   = r2_score(y_test, y_pred_hgb)
mae_hgb  = mean_absolute_error(y_test, y_pred_hgb)
rmse_hgb = np.sqrt(mean_squared_error(y_test, y_pred_hgb))

print("-" * 30)
print(f"R²   = {r2_hgb:.4f}")
print(f"MAE  = {mae_hgb:.4f} m")
print(f"RMSE = {rmse_hgb:.4f} m")
print("-" * 30)

# 5. Calculate Feature Importances (Permutation Importance)
# Note: n_jobs=-1 parallelizes the importance calculation
result = permutation_importance(
    hgb_model, X_test, y_test, 
    n_repeats=10, 
    random_state=RANDOM_SEED, 
    n_jobs=-1
)
hgb_importances = result.importances_mean

print("Feature Importances (Permutation):")
sorted_idx = np.argsort(hgb_importances)[::-1]
for i in sorted_idx:
    print(f"  {FEATURES[i]:20s}: {hgb_importances[i]:.4f}")

# 6. Save the model
model_path = Path(RESULTS_DIR) / "best_hgb_model.pkl"
joblib.dump(hgb_model, model_path)
print(f"Model saved to → {model_path}")

#%% ─── 4. Summary CSV ───────────────────────────────────────────────────────────
# Create the summary DataFrame
summary = pd.DataFrame({
    "feature"      : FEATURES,
    "feature_label": feat_labels,
    "hgb_importance": hgb_importances,
})

# Add the active hyperparameters to the CSV
# We extract them directly from the model object
model_params = hgb_model.get_params()
params_to_save = ['max_iter', 'learning_rate', 'max_depth', 'max_leaf_nodes', 'l2_regularization']

for p in params_to_save:
    summary[f"param_{p}"] = model_params.get(p)

# Save to CSV
summary_path = os.path.join(RESULTS_DIR, "snow_topo_summary_hgb.csv")
summary.to_csv(summary_path, index=False)

print(f"\nSummary table with HGB parameters saved → {summary_path}")

#%% ─── 5. Plots ─────────────────────────────────────────────────────────────────
print_section("Plotting")

n_feat = len(FEATURES)

# ── 5a. Feature importances + pred-vs-obs ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, max(5, n_feat * 0.5 + 2)))
fig.suptitle(
    "Snow Depth vs. Topographic Parameters – Zugspitze UAV\n"
    f"Summer DSM: 2025-08-20  |  Winter DSM: 2026-01-17  |  "
    f"n = {len(df_sample):,} pixels",
    fontsize=12,
)

# HGB feature importances
ax = axes[0]
feat_imp = pd.Series(hgb_importances, index=feat_labels)
feat_imp.sort_values().plot.barh(ax=ax, color="#2196F3")
ax.set_title("hgb Feature Importances\n(mean decrease in impurity)")
ax.set_xlabel("Importance")
ax.axvline(0, color="k", lw=0.5)

# Predicted vs observed
ax = axes[1]
ax.hexbin(y_test, y_pred_hgb, gridsize=60, cmap="Blues", mincnt=1)
lim = [0, max(y_test.max(), y_pred_hgb.max())]
ax.plot(lim, lim, "r--", lw=1.2, label="1:1 line")
ax.set_title(f"hgb: Predicted vs Observed\nR² = {r2_hgb:.3f}  RMSE = {rmse_hgb:.3f} m")
ax.set_xlabel("Observed snow depth (m)")
ax.set_ylabel("Predicted snow depth (m)")
ax.legend(fontsize=8)

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "snow_topo_analysis.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Model summary plot saved → {plot_path}")

# ── 5b. Hexbin scatter: snow depth vs each predictor ─────────────────────────
n_cols  = 4
n_rows  = int(np.ceil(n_feat / n_cols))
fig2, axes2 = plt.subplots(n_rows, n_cols,
                            figsize=(n_cols * 4, n_rows * 3.5))
axes2_flat = axes2.ravel() if n_feat > 1 else [axes2]
fig2.suptitle("Snow Depth vs. Individual Topographic Predictors", fontsize=12)

for i, feat in enumerate(FEATURES):
    ax = axes2_flat[i]
    hb = ax.hexbin(df_sample[feat], df_sample["snow_depth"],
                   gridsize=50, cmap="viridis", mincnt=1)
    plt.colorbar(hb, ax=ax, label="n pixels")
    ax.set_xlabel(FEATURE_LABELS[feat], fontsize=9)
    ax.set_ylabel("Snow depth (m)", fontsize=9)
    ax.set_title(FEATURE_LABELS[feat], fontsize=9)

# hide unused axes
for j in range(n_feat, len(axes2_flat)):
    axes2_flat[j].set_visible(False)

plt.tight_layout()
scatter_path = os.path.join(RESULTS_DIR, "snow_topo_scatter.png")
plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Scatter grid saved → {scatter_path}")

# ── 5c. Partial dependence plots (hgb) ─────────────────────────────────────────
fig3, axes3 = plt.subplots(n_rows, n_cols,
                            figsize=(n_cols * 4, n_rows * 3.5))
axes3_flat = axes3.ravel() if n_feat > 1 else [axes3]
fig3.suptitle(f"Partial Dependence Plots – Random Forest  (R²={r2_hgb:.3f})",
              fontsize=12)

PartialDependenceDisplay.from_estimator(
    hgb_model, X_test,
    features=list(range(n_feat)),
    feature_names=feat_labels,
    ax=axes3_flat[:n_feat],
)
for j in range(n_feat, len(axes3_flat)):
    axes3_flat[j].set_visible(False)

plt.tight_layout()
pdp_path = os.path.join(RESULTS_DIR, "partial_dependence.png")
plt.savefig(pdp_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Partial dependence plot saved → {pdp_path}")

# ─── Done ─────────────────────────────────────────────────────────────────────
print_section("Results summary")
print(f"  Random Forest R²  = {r2_hgb:.4f}")
print(f"  RMSE              = {rmse_hgb:.4f} m")
print(f"  Most important predictor: {FEATURE_LABELS[FEATURES[np.argmax(hgb_importances)]]}")
print(f"\nAll outputs in: {RESULTS_DIR}")
