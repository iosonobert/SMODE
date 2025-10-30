import os, sys
import re
from datetime import datetime
import numpy as np

import glob, os
import rasterio, pandas as pd
import exiftool

instrument = sys.argv[1]  # Change this as needed
print(f'Processing instrument: {instrument}')
if instrument is None:
    raise ValueError("Instrument argument is required.")

## Change this
EXIF_TOOL_full = r'/software/projects/pawsey0106/azulberti/Image-ExifTool-12.69/exiftool'

imgsearch = '*/*.tif'

# outzarr  = os.path.join(field_dir, f'processed_data/{camera}/{camera}_V3_DASK.zarr')
def parse_exif(fname):
    ## Simple function to extract EXIF metadata from a single file
    with exiftool.ExifTool(EXIF_TOOL_full) as et:
        metadata = et.execute_json(fname)
        df_meta  = pd.DataFrame(metadata)

    return df_meta

def pull_times(ds_exif):
    # Simple function fo parse the date and time from the exif data
    dts = []
    for filenames in ds_exif['File:FileName'].values:
        print(filenames)
        # Regex to extract the YYYYMMDD and HHMMSS parts
        match = re.search(r"_(\d{8})_(\d{6})_", filenames)
        if match:
            date_str, time_str = match.groups()
            dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        # print(dt)  # 2023-04-18 00:19:51
        dts.append(dt)

    dts = np.array(dts)
    return dts

basedir = os.environ["MYSCRATCH"] + "/" + instrument
print(f'Base directory: {basedir}')

# List only subdirectories, skip anything ending in .gz
imgdirs = [
    os.path.join(basedir, d) for d in os.listdir(basedir)
    if os.path.isdir(os.path.join(basedir, d)) and not d.endswith(".gz")
          ]

print(f'Found {len(imgdirs)} subdirectories in {basedir}.')

for imgdir in imgdirs:
    print(f'Processing directory: {imgdir}')
    netcdfname = os.path.join(basedir, imgdir + '_exif_data.nc')
    print(f'   Target netcdf: {netcdfname}')
    
    if os.path.exists(netcdfname):
        print(f'   {netcdfname} already exists, skipping...')
        continue

    files = glob.glob(imgdir + '/*.tif')

    tiff_files = glob.glob(f"{imgdir}{imgsearch}", recursive=True)
    print(len(tiff_files), ' found.\nConverting...')

    print('Reading EXIF data...')
    df = []

    for ff in tiff_files:
        df.append(parse_exif(ff))
        if len(df) % 100 == 0:
            print(f'   {len(df)} files processed...')
            
        if len(df) >2:
            pass # Debug step
    
    if len(df) == 0:
        print(f'   No files found in {imgdir}, skipping...')
        continue
        
    print('EXIF data read...')
    print(len(df), ' files processed.')

    exif = pd.concat(df)
    ds_exif = exif.to_xarray().rename({'index':'time'})

    print('Parsing and sorting by times')
    ds_exif['time'] = pull_times(ds_exif)

    # Sort the dataset by time
    order = ds_exif['time'].values.argsort()
    ds_exif = ds_exif.isel(time=order)

    ds_exif.to_netcdf(netcdfname)
    print(f'   Saved to {netcdfname}')

    print(ds_exif)
    print('Done.')
