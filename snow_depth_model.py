import os
import whitebox_workflows as wbw
from whitebox_workflows import download_sample_data, show, WbEnvironment
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray as riox

print(os.getcwd())
wbe = WbEnvironment()
wbe.working_directory = os.path.join(os.getcwd(), "uas_data")
print(wbe.working_directory)

dates = os.listdir("./uas_data/")
print(dates)
folders = [os.path.join(os.getcwd(), "uas_data", date) for date in dates]
print(folders)

dsms = {}
rgbs = {}
snow_m = []
for date, folder in zip(dates, folders):
    files = os.listdir(folder)
    dsm_files = [f for f in files if "dsm" in f.lower()]
    dsms[date] = dsm_files
    rgb_files = [f for f in files if "rgb" in f.lower()]
    rgbs[date] = rgb_files


print(dsms)
print(rgbs)

summer_date = "2025-08-20"
winter_date = "2026-01-17"
wbe.working_directory = os.path.join(os.getcwd(), "uas_data", summer_date)
print(wbe.working_directory)
summer_lidar_dsm = dsms[summer_date][0]
print(summer_lidar_dsm)
wbe_summer_lidar_dsm = wbe.read_raster(summer_lidar_dsm)

wbe.working_directory = os.path.join(os.getcwd(), "uas_data", winter_date)
print(wbe.working_directory)
winter_lidar_dsm = dsms[winter_date][0]
print(winter_lidar_dsm)
wbe_winter_lidar_dsm = wbe.read_raster(winter_lidar_dsm)

snow_depth_model = wbe_winter_lidar_dsm - wbe_summer_lidar_dsm
output_dir = os.path.join(os.getcwd(), "out")
os.makedirs(output_dir, exist_ok=True)
wbe.working_directory = output_dir
wbe.write_raster(snow_depth_model, "snow_depth_model.tif")
wbe.raster_histogram(snow_depth_model, "snow_depth_histogram.html")
