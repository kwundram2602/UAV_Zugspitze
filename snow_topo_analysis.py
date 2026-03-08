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

import os
import numpy as np
import pandas as pd
import rioxarray as riox
import matplotlib
matplotlib.use("Agg")           # headless; change to "TkAgg" if you want a window
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import PartialDependenceDisplay
from pathlib import Path


# ─── Configuration ────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
OUT_DIR     = HERE / "out_chunked"
INDICES_DIR = os.path.join(OUT_DIR, "indices")    # rasters from topo_indices.py
RESULTS_DIR = os.path.join(OUT_DIR, "analysis")   # plots + CSV from this script

# All TPI scales to include as separate features
TPI_RADII_PX = [50, 250, 500]   # → 5 m, 25 m, 50 m at 10 cm resolution

# Wind directions to include (must match labels from topo_indices.py)
WIND_DIRECTIONS = {"SW": 225, "W": 270, "E": 90, "SE": 135}

# Sampling
N_SAMPLE    = 1_000_000   # max pixels drawn for modelling
RANDOM_SEED = 42

# Spatial chunk experiment
RUN_CHUNK_EXPERIMENT = True
CHUNK_ROWS = 4
CHUNK_COLS = 4
MIN_CHUNK_SAMPLES = 120_000

# Snow depth filter (remove outliers / artefacts)
SNOW_MIN_M = 0.04   # below this → likely measurement noise, excluded
SNOW_MAX_M = 8.00   # above this → likely DSM artefact, excluded

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_raster(path: str) -> np.ndarray:
    """Load a single-band raster by full path, return 2-D array (NaN = nodata)."""
    da = riox.open_rasterio(path, masked=True).squeeze()
    return da.values


def print_section(title: str):
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def chunkize(shape: tuple[int, int], n_rows: int, n_cols: int) -> np.ndarray:
    """Return a 2-D integer chunk-id map for a raster shape."""
    if n_rows < 1 or n_cols < 1:
        raise ValueError("n_rows and n_cols must both be >= 1")

    height, width = shape
    row_edges = np.linspace(0, height, n_rows + 1, dtype=int)
    col_edges = np.linspace(0, width, n_cols + 1, dtype=int)

    chunk_map = np.empty((height, width), dtype=np.int32)
    chunk_id = 0
    for r in range(n_rows):
        r0, r1 = row_edges[r], row_edges[r + 1]
        for c in range(n_cols):
            c0, c1 = col_edges[c], col_edges[c + 1]
            chunk_map[r0:r1, c0:c1] = chunk_id
            chunk_id += 1

    return chunk_map


def fit_and_score_rf(X_train: np.ndarray, X_test: np.ndarray,
                     y_train: np.ndarray, y_test: np.ndarray,
                     random_seed: int) -> tuple[RandomForestRegressor, np.ndarray, dict]:
    """Train RF and return model, predictions, and score dictionary."""
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=random_seed,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    scores = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    }
    return rf, y_pred, scores

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ─── 1. Load rasters ──────────────────────────────────────────────────────────
    print_section("Loading rasters")

    snow  = load_raster(os.path.join(OUT_DIR,     "snow_depth_model.tif"))
    slope = load_raster(os.path.join(INDICES_DIR, "slope.tif"))

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

    print(f"  Array shape  : {snow.shape}")
    print(f"  Total pixels : {snow.size:,}")

    # ─── 2. Build sample DataFrame ────────────────────────────────────────────────
    print_section("Building sample")

    chunk_map = chunkize(snow.shape, CHUNK_ROWS, CHUNK_COLS)

    df_dict = {
        "snow_depth": snow.ravel(),
        "slope": slope.ravel(),
        "chunk_id": chunk_map.ravel(),
    }
    for key, arr in tpi_arrays.items():
        df_dict[key] = arr.ravel()
    for key, arr in wi_arrays.items():
        df_dict[key] = arr.ravel()

    df = pd.DataFrame(df_dict)

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
        .drop(columns="_decile")
    )
    print(f"  Stratified sample size    : {len(df_sample):,}")

    FEATURES = (
        ["slope"]
        + list(tpi_arrays.keys())
        + list(wi_arrays.keys())
    )

    FEATURE_LABELS = {"slope": "Slope (°)"}
    for r in TPI_RADII_PX:
        FEATURE_LABELS[f"tpi_{r}px"] = f"TPI {r}px ({r*0.10:.0f} m)"
    for label, deg in WIND_DIRECTIONS.items():
        FEATURE_LABELS[f"wi_{label}"] = f"WI {label} ({deg}°)"

    feat_labels = [FEATURE_LABELS[f] for f in FEATURES]

    X = df_sample[FEATURES].values
    y = df_sample["snow_depth"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # ─── 3. Random Forest ─────────────────────────────────────────────────────────
    print_section("Random Forest")

    rf, y_pred_rf, global_scores = fit_and_score_rf(
        X_train, X_test, y_train, y_test, RANDOM_SEED
    )

    r2_rf = global_scores["r2"]
    mae_rf = global_scores["mae"]
    rmse_rf = global_scores["rmse"]

    print(f"  R²   = {r2_rf:.4f}")
    print(f"  MAE  = {mae_rf:.4f} m")
    print(f"  RMSE = {rmse_rf:.4f} m")
    print(f"  Feature importances:")
    for feat, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1]):
        print(f"    {feat:20s}: {imp:.4f}")

    # ─── 3b. Chunk-wise Random Forest experiment ────────────────────────────────
    chunk_summary = None
    if RUN_CHUNK_EXPERIMENT:
        print_section("Chunk-wise Random Forest")
        print(f"  Chunk grid              : {CHUNK_ROWS} x {CHUNK_COLS}")
        print(f"  Min samples per chunk   : {MIN_CHUNK_SAMPLES:,}")

        y_true_all = []
        y_pred_all = []
        chunk_rows = []

        for chunk_id, g in df_sample.groupby("chunk_id"):
            n_chunk = len(g)
            if n_chunk < MIN_CHUNK_SAMPLES:
                chunk_rows.append({
                    "chunk_id": int(chunk_id),
                    "n_samples": int(n_chunk),
                    "r2": np.nan,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "status": "skipped_too_small",
                })
                continue

            Xc = g[FEATURES].values
            yc = g["snow_depth"].values
            Xc_train, Xc_test, yc_train, yc_test = train_test_split(
                Xc, yc, test_size=0.2, random_state=RANDOM_SEED
            )
            _, y_chunk_pred, chunk_scores = fit_and_score_rf(
                Xc_train, Xc_test, yc_train, yc_test, RANDOM_SEED
            )

            y_true_all.append(yc_test)
            y_pred_all.append(y_chunk_pred)
            chunk_rows.append({
                "chunk_id": int(chunk_id),
                "n_samples": int(n_chunk),
                "r2": chunk_scores["r2"],
                "mae": chunk_scores["mae"],
                "rmse": chunk_scores["rmse"],
                "status": "ok",
            })

            print(
                f"  chunk={int(chunk_id):2d}  n={n_chunk:9,}  "
                f"R²={chunk_scores['r2']:.4f}  RMSE={chunk_scores['rmse']:.4f} m"
            )

        chunk_summary = pd.DataFrame(chunk_rows).sort_values("chunk_id").reset_index(drop=True)
        chunk_summary_path = os.path.join(RESULTS_DIR, "snow_topo_chunk_summary.csv")
        chunk_summary.to_csv(chunk_summary_path, index=False)
        print(f"\n  Chunk summary saved      → {chunk_summary_path}")

        valid_eval = chunk_summary[chunk_summary["status"] == "ok"]
        if len(valid_eval) > 0 and y_true_all:
            y_true_concat = np.concatenate(y_true_all)
            y_pred_concat = np.concatenate(y_pred_all)
            chunk_global_r2 = r2_score(y_true_concat, y_pred_concat)
            chunk_global_mae = mean_absolute_error(y_true_concat, y_pred_concat)
            chunk_global_rmse = np.sqrt(mean_squared_error(y_true_concat, y_pred_concat))

            print("\n  Global vs chunked comparison")
            print(f"    Global RF     : R²={r2_rf:.4f}  MAE={mae_rf:.4f} m  RMSE={rmse_rf:.4f} m")
            print(
                f"    Chunked RF    : R²={chunk_global_r2:.4f}  "
                f"MAE={chunk_global_mae:.4f} m  RMSE={chunk_global_rmse:.4f} m"
            )

            cmp_df = pd.DataFrame([
                {"model": "global", "r2": r2_rf, "mae": mae_rf, "rmse": rmse_rf},
                {
                    "model": "chunked_aggregate",
                    "r2": chunk_global_r2,
                    "mae": chunk_global_mae,
                    "rmse": chunk_global_rmse,
                },
            ])
            cmp_path = os.path.join(RESULTS_DIR, "snow_topo_global_vs_chunked.csv")
            cmp_df.to_csv(cmp_path, index=False)
            print(f"  Comparison table saved   → {cmp_path}")
        else:
            print("  No chunk had enough data for evaluation.")

    # ─── 4. Summary CSV ───────────────────────────────────────────────────────────
    summary = pd.DataFrame({
        "feature"      : FEATURES,
        "feature_label": feat_labels,
        "rf_importance": rf.feature_importances_,
    })
    summary_path = os.path.join(RESULTS_DIR, "snow_topo_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary table saved → {summary_path}")

    # ─── 5. Plots ─────────────────────────────────────────────────────────────────
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

    # RF feature importances
    ax = axes[0]
    feat_imp = pd.Series(rf.feature_importances_, index=feat_labels)
    feat_imp.sort_values().plot.barh(ax=ax, color="#2196F3")
    ax.set_title("RF Feature Importances\n(mean decrease in impurity)")
    ax.set_xlabel("Importance")
    ax.axvline(0, color="k", lw=0.5)

    # Predicted vs observed
    ax = axes[1]
    ax.hexbin(y_test, y_pred_rf, gridsize=60, cmap="Blues", mincnt=1)
    lim = [0, max(y_test.max(), y_pred_rf.max())]
    ax.plot(lim, lim, "r--", lw=1.2, label="1:1 line")
    ax.set_title(f"RF: Predicted vs Observed\nR² = {r2_rf:.3f}  RMSE = {rmse_rf:.3f} m")
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

    # ── 5c. Partial dependence plots (RF) ─────────────────────────────────────────
    fig3, axes3 = plt.subplots(n_rows, n_cols,
                                figsize=(n_cols * 4, n_rows * 3.5))
    axes3_flat = axes3.ravel() if n_feat > 1 else [axes3]
    fig3.suptitle(f"Partial Dependence Plots – Random Forest  (R²={r2_rf:.3f})",
                  fontsize=12)

    PartialDependenceDisplay.from_estimator(
        rf, X_test,
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
    print(f"  Random Forest R²  = {r2_rf:.4f}")
    print(f"  RMSE              = {rmse_rf:.4f} m")
    print(f"  Most important predictor: {FEATURE_LABELS[FEATURES[np.argmax(rf.feature_importances_)]]}")
    print(f"\nAll outputs in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
