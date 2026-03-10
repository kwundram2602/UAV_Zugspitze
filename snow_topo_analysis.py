"""
snow_topo_analysis.py
=====================
Statistical analysis of snow depth vs. topographic parameters.

Workflow:
  1. Load snow depth model + topographic indices (from topo_indices.py output)
  2. Build a random pixel sample (stratified by snow depth quantile)
  3. Fit Random Forest or HistGradientBoostingRegressor → feature importance, partial dependence
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
    out/indices/dir_relief_E.tif       (directional relief E)
    out/indices/dir_relief_SE.tif      (directional relief SE)
    out/indices/dir_relief_SW.tif      (directional relief SW)
    out/indices/dir_relief_W.tif       (directional relief W)

Analysis results are written to   out/analysis/

Run topo_indices.py first if the index rasters are missing.
"""

import os
import numpy as np
import pandas as pd
import rioxarray as riox
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")           # headless; change to "TkAgg" if you want a window
import matplotlib.pyplot as plt

from shapely.geometry import box
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import PartialDependenceDisplay,permutation_importance
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
OUT_DIR     = HERE / "out"
INDICES_DIR = os.path.join(OUT_DIR, "indices")    # rasters from topo_indices.py
RESULTS_DIR = os.path.join(OUT_DIR, "analysis")   # plots + CSV from this script

# All TPI scales to include as separate features
TPI_RADII_PX = [50, 250, 500]   # → 5 m, 25 m, 50 m at 10 cm resolution

# Wind directions to include (must match labels from topo_indices.py)
WIND_DIRECTIONS = {"E": 90, "SE": 135}

# Directional relief directions (must match dir_relief_*.tif filenames)
DIR_RELIEF_DIRECTIONS = ["E", "SE", "SW", "W"]

# Model to use: "rf" (RandomForestRegressor) or "hgbr" (HistGradientBoostingRegressor)
MODEL_TYPE = "hgbr"

# Colormap for the predicted-vs-observed hexbin in analysis.png
# Any matplotlib colormap name works, e.g. "Blues", "YlOrRd", "plasma", "viridis"
SCATTER_CMAP = "Blues"

# Sampling
N_SAMPLE    = 6_000_000   # max pixels drawn for modelling
RANDOM_SEED = 42

# Spatial chunk experiment
RUN_CHUNK_EXPERIMENT = True
CHUNK_ROWS = 4
CHUNK_COLS = 4
MIN_CHUNK_SAMPLES = 300_000

# Snow depth filter (remove outliers / artefacts)
SNOW_MIN_M = 0.04   # below this → likely measurement noise, excluded
SNOW_MAX_M = 8.00   # above this → likely DSM artefact, excluded

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_raster(path: str) -> np.ndarray:
    """Load a single-band raster by full path, return 2-D array (NaN = nodata)."""
    da = riox.open_rasterio(path, masked=True).squeeze()
    return da.values


def load_raster_with_meta(path: str) -> tuple[np.ndarray, object, object]:
    """Load a single-band raster, return (2-D array, CRS, affine transform)."""
    da = riox.open_rasterio(path, masked=True).squeeze()
    return da.values, da.rio.crs, da.rio.transform()


def print_section(title: str):
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


def valid_data_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    """
    Return the bounding box of non-NaN pixels as pixel indices.

    Returns
    -------
    (row_min, row_max, col_min, col_max)
        row_max and col_max are exclusive (i.e. suitable for slicing).
    """
    mask = ~np.isnan(arr)
    valid_rows = np.where(np.any(mask, axis=1))[0]
    valid_cols = np.where(np.any(mask, axis=0))[0]
    if valid_rows.size == 0 or valid_cols.size == 0:
        raise ValueError("Array contains no valid (non-NaN) pixels.")
    return (
        int(valid_rows[0]),
        int(valid_rows[-1]) + 1,
        int(valid_cols[0]),
        int(valid_cols[-1]) + 1,
    )


def chunkize(
    shape: tuple[int, int],
    n_rows: int,
    n_cols: int,
    row_start: int = 0,
    row_end:   int | None = None,
    col_start: int = 0,
    col_end:   int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide a raster region into a regular grid of chunks.

    The chunk grid covers only [row_start:row_end, col_start:col_end].
    Pixels outside that region receive chunk_id = -1.

    Parameters
    ----------
    shape                : full raster (height, width)
    n_rows, n_cols       : grid dimensions
    row_start / row_end  : pixel row bounds of the region to divide
                           (row_end exclusive; defaults to full height)
    col_start / col_end  : pixel col bounds of the region to divide
                           (col_end exclusive; defaults to full width)

    Returns
    -------
    chunk_map : ndarray of shape *shape*, dtype int32
        Chunk id per pixel; -1 outside the chunked region.
    row_edges : 1-D int array, length n_rows + 1
        Row-pixel boundaries within [row_start, row_end].
    col_edges : 1-D int array, length n_cols + 1
        Col-pixel boundaries within [col_start, col_end].
    """
    if n_rows < 1 or n_cols < 1:
        raise ValueError("n_rows and n_cols must both be >= 1")

    height, width = shape
    row_end = row_end if row_end is not None else height
    col_end = col_end if col_end is not None else width

    row_edges = np.round(np.linspace(row_start, row_end, n_rows + 1)).astype(int)
    col_edges = np.round(np.linspace(col_start, col_end, n_cols + 1)).astype(int)

    chunk_map = np.full((height, width), -1, dtype=np.int32)
    chunk_id = 0
    for r in range(n_rows):
        r0, r1 = row_edges[r], row_edges[r + 1]
        for c in range(n_cols):
            c0, c1 = col_edges[c], col_edges[c + 1]
            chunk_map[r0:r1, c0:c1] = chunk_id
            chunk_id += 1

    return chunk_map, row_edges, col_edges


def export_chunk_boundaries(
    n_rows: int,
    n_cols: int,
    row_edges: np.ndarray,
    col_edges: np.ndarray,
    crs,
    transform,
    out_path: str,
) -> gpd.GeoDataFrame:
    """
    Build a GeoDataFrame of chunk boundary polygons and save as GeoPackage.

    Each chunk is represented as a rectangular polygon in the raster's CRS.
    The affine *transform* converts (col, row) pixel coordinates to (x, y).

    Returns the GeoDataFrame for optional further use.
    """
    records = []
    chunk_id = 0
    for r in range(n_rows):
        for c in range(n_cols):
            r0, r1 = int(row_edges[r]),  int(row_edges[r + 1])
            c0, c1 = int(col_edges[c]),  int(col_edges[c + 1])

            # Affine transform: (col, row) → (x, y)
            x_min, y_max = transform * (c0, r0)   # top-left  pixel corner
            x_max, y_min = transform * (c1, r1)   # bottom-right pixel corner

            records.append({
                "chunk_id":  chunk_id,
                "chunk_row": r,
                "chunk_col": c,
                "geometry":  box(x_min, y_min, x_max, y_max),
            })
            chunk_id += 1

    gdf = gpd.GeoDataFrame(records, crs=crs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    gdf.to_file(out_path, driver="GPKG")
    print(f"  Chunk boundaries saved → {out_path}")
    return gdf


def fit_and_score(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    random_seed: int,
    model_type: str = "rf",
) -> tuple:
    """Train a model and return (model, predictions, score dict).

    Parameters
    ----------
    model_type : "rf"   → RandomForestRegressor
                 "hgbr" → HistGradientBoostingRegressor
    """
    if model_type == "hgbr":
        model = HistGradientBoostingRegressor(
            max_iter=300,
            max_depth=12,
            min_samples_leaf=20,
            random_state=random_seed,
        )
    else:  # default: rf
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=random_seed,
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    scores = {
        "r2":   r2_score(y_test, y_pred),
        "mae":  mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    }
    return model, y_pred, scores


def make_model_plots(
    label: str,
    out_dir: str,
    model,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    scores: dict,
    FEATURES: list[str],
    feat_labels: list[str],
    FEATURE_LABELS: dict,
    X_test: np.ndarray,
    df_slice: pd.DataFrame,
    n_sample_total: int,
    model_type: str = "rf",
    scatter_cmap: str = "Blues",
):
    """
    Generate and save the three standard diagnostic plots for one model.

    Plots produced (all written to *out_dir*):
      • <label>_analysis.png      – feature importances + predicted vs observed
      • <label>_scatter.png       – hexbin snow depth vs each predictor
      • <label>_partial_dep.png   – partial dependence plots

    Parameters
    ----------
    label          : short identifier, used as filename prefix and in titles
    out_dir        : directory where the three PNG files are written
    model          : trained model (RandomForestRegressor or HistGradientBoostingRegressor)
    y_test / y_pred: hold-out observations and predictions
    scores         : dict with keys 'r2', 'mae', 'rmse'
    FEATURES       : list of feature column names (same order as X)
    feat_labels    : human-readable labels aligned with FEATURES
    FEATURE_LABELS : mapping feature name → human label (for scatter axes)
    X_test         : test feature matrix (for partial dependence)
    df_slice       : DataFrame subset used for this model (for scatter plots)
    n_sample_total : total sample size (for plot title)
    model_type     : "rf" or "hgbr" (affects importance method and labels)
    """
    os.makedirs(out_dir, exist_ok=True)
    n_feat = len(FEATURES)
    r2   = scores["r2"]
    rmse = scores["rmse"]
    model_name = "RF" if model_type == "rf" else "HGBR"

    # ── Feature importances ────────────────────────────────────────────────────
    # RF: built-in impurity-based importances
    # HGBR: permutation importances (no built-in .feature_importances_ guarantee)
    if model_type == "rf":
        imp_values = model.feature_importances_
        imp_title  = "RF Feature Importances\n(mean decrease in impurity)"
    else:
        print("    Computing permutation importances for HGBR …")
        perm = permutation_importance(
            model, X_test, y_test,
            n_repeats=10, random_state=42, n_jobs=-1,
        )
        imp_values = perm.importances_mean
        imp_title  = "HGBR Feature Importances\n(permutation, mean ± std)"

    # ── (a) Feature importances + pred-vs-obs ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, n_feat * 0.5 + 2)))
    fig.suptitle(
        f"Snow Depth vs. Topographic Parameters – {label}\n"
        f"Summer DSM: 2025-08-20  |  Winter DSM: 2026-01-17  |  "
        f"n = {n_sample_total:,} pixels",
        fontsize=12,
    )

    ax = axes[0]
    feat_imp = pd.Series(imp_values, index=feat_labels)
    feat_imp.sort_values().plot.barh(ax=ax, color="#2196F3")
    ax.set_title(imp_title)
    ax.set_xlabel("Importance")
    ax.axvline(0, color="k", lw=0.5)

    ax = axes[1]
    hb = ax.hexbin(y_test, y_pred, gridsize=60, cmap=scatter_cmap, mincnt=1)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("n pixels", fontsize=9)
    lim = [0, max(y_test.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", lw=1.2, label="1:1 line")
    ax.set_title(f"{model_name}: Predicted vs Observed\nR² = {r2:.3f}  RMSE = {rmse:.3f} m")
    ax.set_xlabel("Observed snow depth (m)")
    ax.set_ylabel("Predicted snow depth (m)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path_a = os.path.join(out_dir, f"{label}_analysis.png")
    plt.savefig(path_a, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Model summary plot  → {path_a}")

    # ── (b) Hexbin scatter: snow depth vs each predictor ──────────────────────
    n_plot_cols = 4
    n_plot_rows = int(np.ceil(n_feat / n_plot_cols))
    fig2, axes2 = plt.subplots(
        n_plot_rows, n_plot_cols,
        figsize=(n_plot_cols * 4, n_plot_rows * 3.5),
    )
    axes2_flat = axes2.ravel() if n_feat > 1 else [axes2]
    fig2.suptitle(
        f"Snow Depth vs. Individual Topographic Predictors – {label}",
        fontsize=12,
    )

    for i, feat in enumerate(FEATURES):
        ax = axes2_flat[i]
        hb = ax.hexbin(
            df_slice[feat], df_slice["snow_depth"],
            gridsize=50, cmap="viridis", mincnt=1,
        )
        plt.colorbar(hb, ax=ax, label="n pixels")
        ax.set_xlabel(FEATURE_LABELS[feat], fontsize=9)
        ax.set_ylabel("Snow depth (m)", fontsize=9)
        ax.set_title(FEATURE_LABELS[feat], fontsize=9)

    for j in range(n_feat, len(axes2_flat)):
        axes2_flat[j].set_visible(False)

    plt.tight_layout()
    path_b = os.path.join(out_dir, f"{label}_scatter.png")
    plt.savefig(path_b, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Scatter grid        → {path_b}")

    # ── (c) Partial dependence plots ──────────────────────────────────────────
    fig3, axes3 = plt.subplots(
        n_plot_rows, n_plot_cols,
        figsize=(n_plot_cols * 4, n_plot_rows * 3.5),
    )
    axes3_flat = axes3.ravel() if n_feat > 1 else [axes3]
    fig3.suptitle(
        f"Partial Dependence Plots – {model_name}  (R²={r2:.3f})  – {label}",
        fontsize=12,
    )

    PartialDependenceDisplay.from_estimator(
        model, X_test,
        features=list(range(n_feat)),
        feature_names=feat_labels,
        ax=axes3_flat[:n_feat],
    )
    for j in range(n_feat, len(axes3_flat)):
        axes3_flat[j].set_visible(False)

    plt.tight_layout()
    path_c = os.path.join(out_dir, f"{label}_partial_dep.png")
    plt.savefig(path_c, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Partial dependence  → {path_c}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ─── 1. Load rasters ──────────────────────────────────────────────────────
    print_section("Loading rasters")

    snow, snow_crs, snow_transform = load_raster_with_meta(
        os.path.join(OUT_DIR, "snow_depth_model.tif")
    )
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

    dr_arrays = {}
    for lbl in DIR_RELIEF_DIRECTIONS:
        key = f"dr_{lbl}"
        dr_arrays[key] = load_raster(os.path.join(INDICES_DIR, f"dir_relief_{lbl}.tif"))
        print(f"  Loaded indices/dir_relief_{lbl}.tif")

    print(f"  Array shape  : {snow.shape}")
    print(f"  Total pixels : {snow.size:,}")

    # ─── 2. Build sample DataFrame ────────────────────────────────────────────
    print_section("Building sample")

    row_min, row_max, col_min, col_max = valid_data_bbox(snow)
    print(
        f"  Valid data extent : rows [{row_min}:{row_max}]  "
        f"cols [{col_min}:{col_max}]  "
        f"({row_max - row_min} × {col_max - col_min} px)"
    )

    chunk_map, row_edges, col_edges = chunkize(
        snow.shape, CHUNK_ROWS, CHUNK_COLS,
        row_start=row_min, row_end=row_max,
        col_start=col_min, col_end=col_max,
    )

    # Export chunk boundaries as GeoPackage
    gpkg_path = os.path.join(RESULTS_DIR, "chunk_boundaries.gpkg")
    export_chunk_boundaries(
        CHUNK_ROWS, CHUNK_COLS,
        row_edges, col_edges,
        snow_crs, snow_transform,
        gpkg_path,
    )

    df_dict = {
        "snow_depth": snow.ravel(),
        "slope":      slope.ravel(),
        "chunk_id":   chunk_map.ravel(),
    }
    for key, arr in tpi_arrays.items():
        df_dict[key] = arr.ravel()
    for key, arr in wi_arrays.items():
        df_dict[key] = arr.ravel()
    for key, arr in dr_arrays.items():
        df_dict[key] = arr.ravel()

    df = pd.DataFrame(df_dict)

    df_valid = df.dropna()
    df_valid = df_valid[
        (df_valid["snow_depth"] >= SNOW_MIN_M) &
        (df_valid["snow_depth"] <= SNOW_MAX_M) &
        (df_valid["chunk_id"] >= 0)   # exclude pixels outside the chunked region
    ]
    print(f"  Valid pixels after filter : {len(df_valid):,}")

    # Stratified sample: equal number of rows per snow-depth decile
    #   → ensures rare deep-snow pixels are represented
    df_valid["_decile"] = pd.qcut(df_valid["snow_depth"], q=10, labels=False, duplicates="drop")
    per_bin = max(1, N_SAMPLE // df_valid["_decile"].nunique())
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
        + list(dr_arrays.keys())
    )

    FEATURE_LABELS = {"slope": "Slope (°)"}
    for r in TPI_RADII_PX:
        FEATURE_LABELS[f"tpi_{r}px"] = f"TPI {r}px ({r*0.10:.0f} m)"
    for lbl, deg in WIND_DIRECTIONS.items():
        FEATURE_LABELS[f"wi_{lbl}"] = f"WI {lbl} ({deg}°)"
    for lbl in DIR_RELIEF_DIRECTIONS:
        FEATURE_LABELS[f"dr_{lbl}"] = f"Dir. Relief {lbl}"

    feat_labels = [FEATURE_LABELS[f] for f in FEATURES]

    X = df_sample[FEATURES].values
    y = df_sample["snow_depth"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # ─── 3. Global model ──────────────────────────────────────────────────────
    model_name = "RF" if MODEL_TYPE == "rf" else "HGBR"
    print_section(f"Global {model_name}")

    global_model, y_pred_global, global_scores = fit_and_score(
        X_train, X_test, y_train, y_test, RANDOM_SEED, model_type=MODEL_TYPE
    )

    r2_rf   = global_scores["r2"]
    mae_rf  = global_scores["mae"]
    rmse_rf = global_scores["rmse"]

    print(f"  R²   = {r2_rf:.4f}")
    print(f"  MAE  = {mae_rf:.4f} m")
    print(f"  RMSE = {rmse_rf:.4f} m")

    if MODEL_TYPE == "rf":
        print(f"  Feature importances:")
        for feat, imp in sorted(zip(FEATURES, global_model.feature_importances_), key=lambda x: -x[1]):
            print(f"    {feat:20s}: {imp:.4f}")

    # Global plots
    make_model_plots(
        label="global",
        out_dir=RESULTS_DIR,
        model=global_model,
        y_test=y_test,
        y_pred=y_pred_global,
        scores=global_scores,
        FEATURES=FEATURES,
        feat_labels=feat_labels,
        FEATURE_LABELS=FEATURE_LABELS,
        X_test=X_test,
        df_slice=df_sample,
        n_sample_total=len(df_sample),
        model_type=MODEL_TYPE,
        scatter_cmap=SCATTER_CMAP,
    )

    # Global summary CSV — importance column computed inside make_model_plots already;
    # here we store whatever is available (impurity for RF, NaN placeholder for HGBR)
    if MODEL_TYPE == "rf":
        imp_col = global_model.feature_importances_
    else:
        imp_col = [np.nan] * len(FEATURES)

    summary = pd.DataFrame({
        "feature"      : FEATURES,
        "feature_label": feat_labels,
        "importance"   : imp_col,
    })
    summary_path = os.path.join(RESULTS_DIR, "snow_topo_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  Summary table saved → {summary_path}")

    # ─── 3b. Chunk-wise model ─────────────────────────────────────────────────
    chunk_summary = None
    if RUN_CHUNK_EXPERIMENT:
        print_section(f"Chunk-wise {model_name}")
        print(f"  Chunk grid              : {CHUNK_ROWS} x {CHUNK_COLS}")
        print(f"  Min samples per chunk   : {MIN_CHUNK_SAMPLES:,}")

        chunks_out_dir = os.path.join(RESULTS_DIR, "chunks")

        y_true_all = []
        y_pred_all = []
        chunk_rows = []

        for chunk_id, g in df_sample.groupby("chunk_id"):
            n_chunk = len(g)
            print(f"\n  ── chunk {int(chunk_id):02d}  (n = {n_chunk:,}) ──")

            if n_chunk < MIN_CHUNK_SAMPLES:
                print(f"     Skipped (< {MIN_CHUNK_SAMPLES:,} samples)")
                chunk_rows.append({
                    "chunk_id": int(chunk_id),
                    "n_samples": int(n_chunk),
                    "r2":   np.nan,
                    "mae":  np.nan,
                    "rmse": np.nan,
                    "status": "skipped_too_small",
                })
                continue

            Xc = g[FEATURES].values
            yc = g["snow_depth"].values
            Xc_train, Xc_test, yc_train, yc_test = train_test_split(
                Xc, yc, test_size=0.2, random_state=RANDOM_SEED
            )

            chunk_model, y_chunk_pred, chunk_scores = fit_and_score(
                Xc_train, Xc_test, yc_train, yc_test, RANDOM_SEED, model_type=MODEL_TYPE
            )

            print(
                f"     R²={chunk_scores['r2']:.4f}  "
                f"MAE={chunk_scores['mae']:.4f} m  "
                f"RMSE={chunk_scores['rmse']:.4f} m"
            )

            # Per-chunk plots in own subdirectory
            chunk_plot_dir = os.path.join(chunks_out_dir, f"chunk_{int(chunk_id):02d}")
            make_model_plots(
                label=f"chunk_{int(chunk_id):02d}",
                out_dir=chunk_plot_dir,
                model=chunk_model,
                y_test=yc_test,
                y_pred=y_chunk_pred,
                scores=chunk_scores,
                FEATURES=FEATURES,
                feat_labels=feat_labels,
                FEATURE_LABELS=FEATURE_LABELS,
                X_test=Xc_test,
                df_slice=g,
                n_sample_total=n_chunk,
                model_type=MODEL_TYPE,
                scatter_cmap=SCATTER_CMAP,
            )

            # Per-chunk feature importance CSV
            if MODEL_TYPE == "rf":
                chunk_imp_col = chunk_model.feature_importances_
            else:
                chunk_imp_col = [np.nan] * len(FEATURES)

            chunk_imp = pd.DataFrame({
                "feature"      : FEATURES,
                "feature_label": feat_labels,
                "importance"   : chunk_imp_col,
            })
            chunk_imp.to_csv(
                os.path.join(chunk_plot_dir, f"chunk_{int(chunk_id):02d}_summary.csv"),
                index=False,
            )

            y_true_all.append(yc_test)
            y_pred_all.append(y_chunk_pred)
            chunk_rows.append({
                "chunk_id": int(chunk_id),
                "n_samples": int(n_chunk),
                "r2":   chunk_scores["r2"],
                "mae":  chunk_scores["mae"],
                "rmse": chunk_scores["rmse"],
                "status": "ok",
            })

        chunk_summary = pd.DataFrame(chunk_rows).sort_values("chunk_id").reset_index(drop=True)
        chunk_summary_path = os.path.join(RESULTS_DIR, "snow_topo_chunk_summary.csv")
        chunk_summary.to_csv(chunk_summary_path, index=False)
        print(f"\n  Chunk summary saved → {chunk_summary_path}")

        # Annotate GeoPackage with chunk metrics
        try:
            gdf = gpd.read_file(gpkg_path)
            gdf = gdf.merge(
                chunk_summary[["chunk_id", "n_samples", "r2", "mae", "rmse", "status"]],
                on="chunk_id",
                how="left",
            )
            gdf.to_file(gpkg_path, driver="GPKG")
            print(f"  Chunk boundaries updated with metrics → {gpkg_path}")
        except Exception as e:
            print(f"  Warning: could not update GeoPackage with metrics: {e}")

        valid_eval = chunk_summary[chunk_summary["status"] == "ok"]
        if len(valid_eval) > 0 and y_true_all:
            y_true_concat = np.concatenate(y_true_all)
            y_pred_concat = np.concatenate(y_pred_all)
            chunk_global_r2   = r2_score(y_true_concat, y_pred_concat)
            chunk_global_mae  = mean_absolute_error(y_true_concat, y_pred_concat)
            chunk_global_rmse = np.sqrt(mean_squared_error(y_true_concat, y_pred_concat))

            print("\n  Global vs chunked comparison")
            print(f"    Global {model_name:<6}: R²={r2_rf:.4f}  MAE={mae_rf:.4f} m  RMSE={rmse_rf:.4f} m")
            print(
                f"    Chunked {model_name:<5}: R²={chunk_global_r2:.4f}  "
                f"MAE={chunk_global_mae:.4f} m  RMSE={chunk_global_rmse:.4f} m"
            )

            cmp_df = pd.DataFrame([
                {"model": "global",           "r2": r2_rf,           "mae": mae_rf,           "rmse": rmse_rf},
                {"model": "chunked_aggregate", "r2": chunk_global_r2, "mae": chunk_global_mae, "rmse": chunk_global_rmse},
            ])
            cmp_path = os.path.join(RESULTS_DIR, "snow_topo_global_vs_chunked.csv")
            cmp_df.to_csv(cmp_path, index=False)
            print(f"  Comparison table saved → {cmp_path}")
        else:
            print("  No chunk had enough data for evaluation.")

    # ─── Done ─────────────────────────────────────────────────────────────────
    print_section("Results summary")
    print(f"  Model type               : {MODEL_TYPE.upper()}")
    print(f"  Global {model_name} R²          = {r2_rf:.4f}")
    print(f"  RMSE                     = {rmse_rf:.4f} m")
    if MODEL_TYPE == "rf":
        best_feat = FEATURE_LABELS[FEATURES[np.argmax(global_model.feature_importances_)]]
        print(f"  Most important predictor : {best_feat}")
    print(f"\nAll outputs in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
