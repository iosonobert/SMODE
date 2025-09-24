imgdir = './SMODE_IOP2_MASS_DOPPVIS_20230427_005616_055-20230427_010241_055'

import glob, os
import rasterio, pandas as pd

files = glob.glob(imgdir + '/*.tif')

## Change this
import exiftool
EXIF_TOOL_full = '/usr/local/bin/exiftool'
EXIF_TOOL_full = r'/home/andrew/Image-ExifTool-12.69/exiftool'

imgsearch = '*/*.tif'

# outzarr  = os.path.join(field_dir, f'processed_data/{camera}/{camera}_V3_DASK.zarr')
def parse_exif(fname):
    with exiftool.ExifTool(EXIF_TOOL_full) as et:
        metadata = et.execute_json(fname)
        df_meta  = pd.DataFrame(metadata)

    return df_meta

tiff_files = glob.glob(f"{imgdir}{imgsearch}", recursive=True)
print(len(tiff_files), ' found.\nConverting...')

print('Reading EXIF data...')
df = []
for ff in tiff_files:
    df.append(parse_exif(ff))
    

exif = pd.concat(df)
ds_exif = exif.to_xarray().rename({'index':'time'})

print(ds_exif)
print('Done.')