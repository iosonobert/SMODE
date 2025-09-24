import os

import glob, os
import rasterio, pandas as pd
import exiftool

## Change this
EXIF_TOOL_full = r'/software/projects/pawsey0106/azulberti/Image-ExifTool-12.69/exiftool'

imgsearch = '*/*.tif'

# outzarr  = os.path.join(field_dir, f'processed_data/{camera}/{camera}_V3_DASK.zarr')
def parse_exif(fname):
    with exiftool.ExifTool(EXIF_TOOL_full) as et:
        metadata = et.execute_json(fname)
        df_meta  = pd.DataFrame(metadata)

    return df_meta

basedir = os.environ["MYSCRATCH"] + "/DOPPVIS"

# List only subdirectories, skip anything ending in .gz
imgdirs = [
    os.path.join(basedir, d) for d in os.listdir(basedir)
    if os.path.isdir(os.path.join(basedir, d)) and not d.endswith(".gz")
          ]

print(f'Found {len(imgdirs)} subdirectories in {basedir}.')

for imgdir in imgdirs:
    print(f'Processing directory: {imgdir}')
    netcdfname = os.path.join(basedir, imgdir + '_exif_data.nc')
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

    print('EXIF data read...')
    print(len(df), ' files processed.')

    exif = pd.concat(df)
    ds_exif = exif.to_xarray().rename({'index':'time'})

    ds_exif.to_netcdf(netcdfname)
    print(f'   Saved to {netcdfname}')

    print(ds_exif)
    print('Done.')