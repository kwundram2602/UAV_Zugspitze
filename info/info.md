Dear all,

you can find the Zugspitze data for your final project here:
https://gigamove.rwth-aachen.de/de/download/c0846f1339658dfb11a04be44db51a70

The dataset contains orthomosaics and digital surface models for 8 field campaigns, as well as snow depth measurements for 5 field campaigns. You can choose which datasets you want to use for your analysis, but you don't need to use all of them. 

For the orthomosaics the band availability varies, as we were not always able to collect data due to bad weather (as you experienced as well). For some dates we only have three bands (blue, green, red) based on data collected with the Zenmuse L1 LiDAR sensor that we also used to create the DSMs. On other dates we have six bands, with the following order: blue, green, red, red-edge, near-infrared, thermal (LWIR). The last band is always an alpha band, since data is not always available for all pixels. 

You can read more about the AltumPT sensor here: https://support.micasense.com/hc/en-us/articles/4419868608407-Altum-PT-Integration-Guide If you want to use the thermal data, for example, you will need to transform the thermal band to °C as follows: (LWIR / 100) - 273.15.

All datasets have been cropped to a subset of the study areas for which we had the highest data coverage. The data was also resampled to 10cm resolution to reduce the file size, but if your computer has problems handling the data you can also resample to a lower resolution. 

If you want to use any other datasets beside the orthomosaics and DSMs provided here, let me know.

Best regards,
Elio 
