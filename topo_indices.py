"""
===============
Compute topographic indices from the summer DSM (snow-free baseline) for
snow distribution analysis:

  - Slope  [degrees]
  - Aspect [degrees, CW from North]
  - TPI    at 3 spatial scales (5 m, 25 m, 50 m radius)
  - Windward Index  = cos(wind_from – aspect) × sin(slope_rad)
                      > 0 → windward / exposed
                      < 0 → leeward  / sheltered
                      computed for 4 wind directions: SW, W, E, SE

"""

import os
from pathlib import Path                          
import numpy as np
import whitebox_workflows as wbw
from whitebox_workflows import WbEnvironment
import rioxarray as riox
import whitebox as wbt

# ─── User configuration ───────────────────────────────────────────────────────
def main():
    HERE        = Path(__file__).resolve().parent
    print(Path(__file__))
    print(HERE)
    SUMMER_DSM  = HERE / "uas_data" / "2025-08-20" / "2025-08-20_dsm_10cm_32632.tif"
    OUT_DIR     = HERE / "out"
    INDICES_DIR = OUT_DIR / "indices"                 
    # TPI neighbourhood radii in pixels (10 cm resolution → 1 px = 0.10 m)
    #   50 px  =  5 m   
    #  250 px  = 25 m   
    #  500 px  = 50 m   
    TPI_RADII_PX = [50, 250, 500]

    # Wind directions to evaluate (degrees from North, clockwise)
    # Key = short label used in output filename,  Value = direction in degrees
    WIND_DIRECTIONS = {
        "SW": 225,   # Föhn / SW inflow
        "W" : 270,   # dominant westerly
        "E" : 90,    # easterly (Zugspitze east face exposure)
        "SE": 135,   # south-easterly
    }

    # ──────────────────────────────────────────────────────────────────────────────

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(INDICES_DIR, exist_ok=True)

    wbe = WbEnvironment()

    dsm_dir  = str(SUMMER_DSM.parent)                # ← FIX 3: use Path methods + str()
    dsm_name = SUMMER_DSM.name                        #    for whitebox compatibility


    # ── 1. Slope ──────────────────────────────────────────────────────────────────
    print("Computing slope …")
    wbe.working_directory = dsm_dir
    dsm   = wbe.read_raster(dsm_name)
    slope = wbe.slope(dsm, units="degrees")
    wbe.working_directory = str(INDICES_DIR)          # ← ensure str for wbe
    wbe.write_raster(slope, "slope.tif")
    print("  → out/indices/slope.tif")


    # ── 2. Aspect ─────────────────────────────────────────────────────────────────
    print("Computing aspect …")
    wbe.working_directory = dsm_dir
    dsm    = wbe.read_raster(dsm_name)
    aspect = wbe.aspect(dsm)
    wbe.working_directory = str(INDICES_DIR)
    wbe.write_raster(aspect, "aspect.tif")
    print("  → out/indices/aspect.tif")


    # ── 3. TPI at multiple scales ─────────────────────────────────────────────────
    for r in TPI_RADII_PX:
        label = f"tpi_{r}px"
        print(f"Computing TPI radius {r} px ({r * 0.10:.0f} m) …")
        wbe.working_directory = dsm_dir
        dsm = wbe.read_raster(dsm_name)
        tpi = wbe.relative_topographic_position(dsm, r, r)
        wbe.working_directory = str(INDICES_DIR)
        wbe.write_raster(tpi, f"{label}.tif")
        print(f"  → out/indices/{label}.tif")


    # ── 4. Windward Index for each wind direction ─────────────────────────────────
    #   WI = cos(wind_from – aspect) × sin(slope)
    #   flat pixels → sin(0) = 0 → WI = 0 (neutral)

    slope_da  = riox.open_rasterio(os.path.join(INDICES_DIR, "slope.tif"),  masked=True).squeeze()
    aspect_da = riox.open_rasterio(os.path.join(INDICES_DIR, "aspect.tif"), masked=True).squeeze()

    # NumPy-Trigonometrie (np.sin, np.cos) erwartet Bogenmaß (Radiant), nicht Grad.
    # Whitebox liefert Slope in Grad (0° = eben, 90° = senkrecht) und
    # Aspect in Grad (0°/360° = Nord, im Uhrzeigersinn).
    # Ohne Umrechnung wäre z.B. np.sin(45) ≈ 0.851 statt sin(45°) = 0.707 –
    # der Wert würde als ~45 Radiant ≈ 2578° interpretiert.
    slope_rad  = np.deg2rad(slope_da.values)
    aspect_rad = np.deg2rad(aspect_da.values)

    for label, deg in WIND_DIRECTIONS.items():
        print(f"Computing Windward Index {label} ({deg}°) …")
        wind_rad = np.deg2rad(deg)
        wi_arr   = np.cos(wind_rad - aspect_rad) * np.sin(slope_rad)
        wi_da    = slope_da.copy(data=wi_arr)
        wi_da.attrs["long_name"] = f"Windward Index from {label} ({deg} deg N)"
        out_name = f"windward_index_{label}.tif"
        wi_da.rio.to_raster(os.path.join(INDICES_DIR, out_name))
        print(f"  → out/indices/{out_name}")

    # ── 5. Directional Relief for each wind direction ───────────────────────────────
    for label, azimuth in WIND_DIRECTIONS.items():
        output_path = os.path.join(INDICES_DIR, f"dir_relief_{label}.tif")
        
        # Whitebox Tool: Directional Relief
        wbt.directional_relief(
            dem = str(SUMMER_DSM),
            output = str(output_path),
            azimuth = azimuth,
            max_dist = 100.0 # Suchradius in Metern (bei 10cm Auflösung sehr wichtig!)
        )
        print(f"  Directional Relief for {label} ({azimuth}°) calculated.")
    print("\nDone. All topographic indices written to:", INDICES_DIR)
    
if __name__ == "__main__":
    main()