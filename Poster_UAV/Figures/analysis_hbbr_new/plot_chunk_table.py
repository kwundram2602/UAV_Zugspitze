import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import os

# --- Paths ---
BASE_DIR = r"D:\EAGLE\Zugspitze\UAV_Zugspitze\Poster_UAV\Figures\analysis_hbbr_new"
SUMMARY_CSV = os.path.join(BASE_DIR, "snow_topo_chunk_summary.csv")
CHUNKS_DIR  = os.path.join(BASE_DIR, "chunks")
OUT_PDF     = os.path.join(BASE_DIR, "chunk_metrics_table.pdf")
OUT_SVG     = os.path.join(BASE_DIR, "chunk_metrics_table.svg")

TOP_N = 3   # top features per chunk

# ── colour palette ──────────────────────────────────────────────────────────
C_HEADER    = "#1b3f6e"      # dark navy header
C_ROW_A     = "#e8f0f8"      # light blue row
C_ROW_B     = "#f7fafd"      # near-white row
C_FEAT_BG   = "#d0e4f5"      # feature-importance bar fill
C_FEAT_NONE = "#f0f4f8"      # bar background
C_TEXT      = "#1a1a2e"
C_WHITE     = "#ffffff"
C_ACCENT    = "#2e86c1"      # bright accent (bar, top-1 highlight)

# R² → blue intensity (low R² = light, high = deep blue)
def r2_color(r2, vmin=0.93, vmax=0.995):
    t = np.clip((r2 - vmin) / (vmax - vmin), 0, 1)
    light = np.array(mcolors.to_rgb("#cfe2f3"))
    dark  = np.array(mcolors.to_rgb("#1b5e9e"))
    rgb = (1 - t) * light + t * dark
    return mcolors.to_hex(rgb)

# ── load data ────────────────────────────────────────────────────────────────
df      = pd.read_csv(SUMMARY_CSV)
n_skip  = (df["status"] != "ok").sum()
df_ok   = df[df["status"] == "ok"].copy()
df_ok["chunk_id"] = df_ok["chunk_id"].astype(int)
df_ok   = df_ok.sort_values("chunk_id").reset_index(drop=True)
df_all  = df_ok.copy()   # all ok chunks for mean row
df_ok   = df_ok.head(4)

# ── build per-chunk data ─────────────────────────────────────────────────────
records = []
for _, row in df_ok.iterrows():
    cid   = int(row["chunk_id"])
    fcsv  = os.path.join(CHUNKS_DIR, f"chunk_{cid:02d}", f"chunk_{cid:02d}_summary.csv")
    feats = []
    if os.path.exists(fcsv):
        fdf   = pd.read_csv(fcsv)
        top   = fdf.nlargest(TOP_N, "importance")
        imp_max = fdf["importance"].max()
        for rank, (_, fr) in enumerate(top.iterrows()):
            feats.append({
                "label": fr["feature_label"],
                "imp":   fr["importance"],
                "rel":   fr["importance"] / imp_max,   # relative to chunk max
                "rank":  rank,
            })
    records.append({
        "cid":    cid,
        "n":      int(row["n_samples"]),
        "r2":     row["r2"],
        "mae":    row["mae"],
        "rmse":   row["rmse"],
        "feats":  feats,
    })

# ── mean row (all ok chunks) ─────────────────────────────────────────────────
mean_rec = {
    "cid":   None,
    "n":     None,
    "r2":    df_all["r2"].mean(),
    "mae":   df_all["mae"].mean(),
    "rmse":  df_all["rmse"].mean(),
    "feats": [],
}

# ── figure layout ────────────────────────────────────────────────────────────
n_rows = len(records) + 1   # +1 for mean row
ROW_H  = 1.26      # inches per data row
HDR_H  = 0.963     # header row height
FIG_W  = 13.0

fig_h  = HDR_H + ROW_H * n_rows + 1.138  # + title + footer

fig = plt.figure(figsize=(FIG_W, fig_h), facecolor=C_WHITE)

# column definitions: (left_x, width, label, align)
COLS = [
    (0.00,  0.07,  "ID",              "center"),
    (0.07,  0.10,  "N Samples",       "right"),
    (0.17,  0.09,  "R²",              "center"),
    (0.26,  0.08,  "MAE (m)",         "right"),
    (0.36,  0.08,  "RMSE (m)",        "right"),
    (0.44,  0.56,  f"Top {TOP_N} Predictors  (feature importance)", "left"),
]

MARGIN_L = 0.03
MARGIN_R = 0.03
COL_SEP  = 0.008   # thin vertical gap before feature col
AVAIL_W  = 1.0 - MARGIN_L - MARGIN_R

def col_x(ci):
    return MARGIN_L + COLS[ci][0] * AVAIL_W

def col_w(ci):
    return COLS[ci][1] * AVAIL_W

def row_y(ri):
    """Bottom y of row ri (0 = header)."""
    total = HDR_H + ROW_H * n_rows
    # place from top, convert to axes fraction later
    return 1.0 - (HDR_H + ROW_H * ri) / fig_h - 0.35 / fig_h

# we'll use plain axes with manual patches + text
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

def fig_to_ax(x_in, y_in):
    return x_in, y_in

def add_rect(x, y, w, h, color, zorder=1, lw=0, ec="none"):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="square,pad=0",
        facecolor=color, edgecolor=ec, linewidth=lw, zorder=zorder,
    ))

# ── convert row/col to figure coordinates ────────────────────────────────────
title_h   = 0.70 / fig_h
footer_h  = 0.438 / fig_h
content_h = 1.0 - title_h - footer_h

def ry(ri):
    """bottom y in axes coords for row ri (0=header)"""
    # rows stack from top downward
    top = 1.0 - title_h
    h   = HDR_H / fig_h if ri == 0 else ROW_H / fig_h
    y   = top - (HDR_H / fig_h) - (ri) * (ROW_H / fig_h)
    if ri == 0:
        y = top - HDR_H / fig_h
    else:
        y = top - HDR_H / fig_h - (ri - 1 + 1) * ROW_H / fig_h
    return y

def row_bottom(ri):
    top = 1.0 - title_h
    if ri == 0:
        return top - HDR_H / fig_h
    return top - HDR_H / fig_h - ri * ROW_H / fig_h

def row_height_ax(ri):
    return HDR_H / fig_h if ri == 0 else ROW_H / fig_h

# ── title ────────────────────────────────────────────────────────────────────
ax.text(0.5, 1.0 - title_h / 2,
        "Chunk Model Performance & Key Predictors",
        ha="center", va="center", fontsize=21, fontweight="bold",
        color=C_HEADER, transform=ax.transAxes)

# ── header row ───────────────────────────────────────────────────────────────
hb = row_bottom(0)
hh = row_height_ax(0)
add_rect(MARGIN_L, hb, AVAIL_W, hh, C_HEADER, zorder=2)

for ci, (_, cw, label, align) in enumerate(COLS):
    cx = col_x(ci)
    cw_ = col_w(ci)
    mid_x = cx + cw_ / 2
    pad = 0.008
    tx  = cx + pad if align == "left" else cx + cw_ - pad if align == "right" else mid_x
    ax.text(tx, hb + hh / 2, label,
            ha=align, va="center", fontsize=14, fontweight="bold",
            color=C_WHITE, transform=ax.transAxes, zorder=3)

# thin separator line before feature col
sep_x = col_x(5) - COL_SEP / 2
ax.plot([sep_x, sep_x], [row_bottom(n_rows), row_bottom(0) + row_height_ax(0)],
        color="#7fb3d3", lw=0.8, transform=ax.transAxes, zorder=4)

# ── data rows ────────────────────────────────────────────────────────────────
for ri, rec in enumerate(records):
    actual_ri = ri + 1
    rb = row_bottom(actual_ri)
    rh = row_height_ax(actual_ri)
    bg = C_ROW_A if ri % 2 == 0 else C_ROW_B
    add_rect(MARGIN_L, rb, AVAIL_W, rh, bg, zorder=1)

    def txt(ci, text, bold=False, color=C_TEXT, size=14, align=None):
        al = align or COLS[ci][3]
        cx = col_x(ci)
        cw_ = col_w(ci)
        pad = 0.008
        tx = cx + pad if al == "left" else cx + cw_ - pad if al == "right" else cx + cw_ / 2
        ax.text(tx, rb + rh / 2, text,
                ha=al, va="center", fontsize=size,
                fontweight="bold" if bold else "normal",
                color=color, transform=ax.transAxes, zorder=3)

    # Chunk label
    txt(0, f"{rec['cid']:02d}", bold=True, color=C_ACCENT)

    # N Samples
    txt(1, f"{rec['n']:,}")

    # R² — coloured badge
    r2c = r2_color(rec["r2"])
    badge_w = col_w(2) * 0.72
    badge_h = rh * 0.60
    badge_x = col_x(2) + (col_w(2) - badge_w) / 2
    badge_y = rb + (rh - badge_h) / 2
    add_rect(badge_x, badge_y, badge_w, badge_h, r2c, zorder=2, lw=0.4, ec="#a0bdd4")
    r2_txt_col = C_WHITE if rec["r2"] > 0.965 else C_TEXT
    ax.text(badge_x + badge_w / 2, rb + rh / 2,
            f"{rec['r2']:.2f}",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            color=r2_txt_col, transform=ax.transAxes, zorder=4)

    # MAE / RMSE
    txt(3, f"{rec['mae']:.2f}")
    txt(4, f"{rec['rmse']:.2f}")

    # Feature importance mini-bars
    if rec["feats"]:
        fx0    = col_x(5) + 0.006
        f_avail = col_w(5) - 0.012
        bar_max_w = f_avail * 0.32
        label_w   = f_avail * 0.58
        val_w     = f_avail * 0.10

        sub_h = rh / (TOP_N + 0.6)
        for k, feat in enumerate(rec["feats"]):
            fy = rb + rh - (k + 1) * sub_h - sub_h * 0.05
            fh = sub_h * 0.72

            # label
            ax.text(fx0, fy + fh / 2, feat["label"],
                    ha="left", va="center", fontsize=12,
                    color=C_TEXT, transform=ax.transAxes, zorder=3)

            # bar background
            bar_x = fx0 + label_w
            add_rect(bar_x, fy + fh * 0.15, bar_max_w, fh * 0.70, C_FEAT_NONE, zorder=2)
            # bar fill
            bar_fill = bar_max_w * feat["rel"]
            bar_col  = C_ACCENT if k == 0 else "#6aacda"
            add_rect(bar_x, fy + fh * 0.15, bar_fill, fh * 0.70, bar_col, zorder=3)

            # value
            ax.text(bar_x + bar_max_w + 0.004, fy + fh / 2,
                    f"{feat['imp']:.2f}",
                    ha="left", va="center", fontsize=11,
                    color="#555555", transform=ax.transAxes, zorder=3)

# ── mean row ─────────────────────────────────────────────────────────────────
mean_ri = len(records) + 1
rb = row_bottom(mean_ri)
rh = row_height_ax(mean_ri)
add_rect(MARGIN_L, rb, AVAIL_W, rh, "#1b3f6e", zorder=2)   # dark header colour

def mean_txt(ci, text, align=None):
    al = align or COLS[ci][3]
    cx = col_x(ci)
    cw_ = col_w(ci)
    pad = 0.008
    tx = cx + pad if al == "left" else cx + cw_ - pad if al == "right" else cx + cw_ / 2
    ax.text(tx, rb + rh / 2, text,
            ha=al, va="center", fontsize=14, fontweight="bold",
            color=C_WHITE, transform=ax.transAxes, zorder=4)

mean_txt(0, f"Mean", align="center")
mean_txt(1, f"n\u202f=\u202f{len(df_all)}", align="center")   # n = 11
mean_txt(2, f"{mean_rec['r2']:.2f}", align="center")
mean_txt(3, f"{mean_rec['mae']:.2f}")
mean_txt(4, f"{mean_rec['rmse']:.2f}")

# ── outer border ─────────────────────────────────────────────────────────────
top_y  = row_bottom(0) + row_height_ax(0)
bot_y  = row_bottom(n_rows)
rect_h = top_y - bot_y
ax.add_patch(mpatches.FancyBboxPatch(
    (MARGIN_L, bot_y), AVAIL_W, rect_h,
    boxstyle="square,pad=0",
    facecolor="none", edgecolor="#2e86c1", linewidth=1.2,
    transform=ax.transAxes, zorder=5,
))


# ── save ─────────────────────────────────────────────────────────────────────
fig.savefig(OUT_PDF, bbox_inches="tight", dpi=200, facecolor=C_WHITE)
fig.savefig(OUT_SVG, bbox_inches="tight", facecolor=C_WHITE)
print(f"Saved:\n  {OUT_PDF}\n  {OUT_SVG}")
plt.show()
