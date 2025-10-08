import os
import re
from datetime import datetime
import numpy as np
import xarray as xr
import glob, os
import rasterio, pandas as pd
import exiftool
import matplotlib.pyplot as plt
import aerial_imagery.geotiff

### INPUTS ###
box_size  = 256*4
image_seq = 8 # Spacing between images in image pairs

exif_nc_dir  = '/scratch/pawsey0106/azulberti/DOPPVIS'

# These ones are East-West flight lines
exif_nc_stub = 'DoppVis_20230418_000132_054-20230418_001344_555'

# These ones are West-East flight lines
exif_nc_stub = 'DoppVis_20230418_001542_055-20230418_002726_555'

print('Starting image processing...')

output_dir = os.path.join(exif_nc_dir, 'processed_images')

exif_nc    = os.path.join(exif_nc_dir, f'{exif_nc_stub}_exif_data.nc')
nc_outfile = os.path.join(output_dir, f'{exif_nc_stub}_extracted_boxes_[{image_seq}_{box_size}].nc')


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if os.path.exists(nc_outfile):
    print(f'   {nc_outfile} already exists...')
    pass

os.path.exists(exif_nc)
ds_exif = xr.open_dataset(exif_nc)

print(ds_exif)

# files = ds_exif['File:FileName'].values
# first_image = files[0]

def get_theta(ds_exif):
    # Extracting the angle of the flight line 
    ModelTiePoint = ds_exif['EXIF:ModelTiePoint'].values
    PixelScale    = ds_exif['EXIF:PixelScale'].values

    Easting_tl    = np.array([float(mtp.split(' ')[3]) for mtp in ModelTiePoint])
    Northing_tl   = np.array([float(mtp.split(' ')[4]) for mtp in ModelTiePoint])
    scale_x       = np.array([float(mtp.split(' ')[0]) for mtp in PixelScale])
    scale_y       = np.array([float(mtp.split(' ')[1]) for mtp in PixelScale])
    width         = ds_exif['EXIF:ImageWidth'].values
    height        = ds_exif['EXIF:ImageHeight'].values

    Easting_br = Easting_tl + scale_x * width
    Northing_br = Northing_tl - scale_y * height

    # Now fit a line to Easting_br and Northing_br  to get the angle
    A = np.vstack([Easting_br, np.ones(len(Easting_br))]).T
    m, c = np.linalg.lstsq(A, Northing_br, rcond=None)[0]
    theta = np.arctan(m) * 180 / np.pi

    print(f'Flight line angle (theta): {theta} degrees')

    return theta, Easting_br, Northing_br

theta, Easting_br, Northing_br = get_theta(ds_exif)
# Now we need to determine the flight direction 
if np.mean(Easting_br[0:10]) > np.mean(Easting_br[-10:]):
    flight_direction = 'E-W'
    raise Exception("Don't wan't to deal with this case yet")
else:
    flight_direction = 'W-E'
    print('Flight direction is W-E')
    # raise Exception("Should be right")

all_exif_nc    = os.path.join(exif_nc_dir, f'*_exif_data.nc')
all_exif_nc_files = glob.glob(all_exif_nc)
print(f'Found {len(all_exif_nc_files)} exif nc files in {exif_nc_dir}')

# print(f'First image in the dataset: {first_image}')

# def read_image(fname):
#     full_path = os.path.join(exif_nc_dir, exif_nc_stub)
#     full_path = os.path.join(full_path, fname)
#     print(f'Reading image: {full_path}')
#     with rasterio.open(full_path) as src:
#         img = src.read(1)  # Read the first band
#         profile = src.profile  # Get image metadata
#     return img, profile

# print(first_image)
# print(type(first_image))
# print(first_image.dtype)
# read_image(first_image)

# print('Image processing complete.')

def get_data(ds_exif, index, image_seq=1):
    # Add an option to be given file 1, such as in the event of series loading

    file  = ds_exif['File:FileName'].values[index]  # Just take the first file for demonstration
    file2 = ds_exif['File:FileName'].values[index+image_seq]  # Just take the first file for demonstration

    data, extent   = _get_data(file)
    data2, extent2 = _get_data(file2)

    time  = ds_exif['time'].values[index]
    time2 = ds_exif['time'].values[index+image_seq]

    return data, extent, data2, extent2, time, time2

def _get_data(file):

    print(f'Getting data for file: {file}')
    full_path = os.path.join(exif_nc_dir, exif_nc_stub)
    full_path = os.path.join(full_path, file)

    with rasterio.open(full_path) as src:
        # Read the first band of the raster data
        # If your GeoTIFF has multiple bands, you can specify a different band index
        data = src.read(1)

        # Get the spatial extent (bounding box) for correct plotting
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    data_ = data.copy().astype('float')  # Convert to float for NaN support
    data_[data_ == src.nodata] = float('nan')  # Replace nodata

    return data_, extent

def find_box(data, extent, data2, extent2, box_size=256*12):
    # Find a random box in the first image and locate it's pixel coordinates in the second image
    import numpy as np
    import random

    max_attemps = 150

    h, w = data.shape
    h2, w2 = data2.shape

    # assert h == h2 and w == w2, "Images must have the same dimensions for this example"

    happy = False
    attempts = 0
    while not happy:
        attempts += 1
        
        y = random.randint(0, h - box_size)
        x = random.randint(0, w - box_size)

        print(f"Trying box at y={y}, x={x}")

        box = data[y:y+box_size, x:x+box_size]

        # Convert pixel coordinates to geographic coordinates
        lon1 = extent[0] + (x / w) * (extent[1] - extent[0])
        lat1 = extent[3] - (y / h) * (extent[3] - extent[2])
        lon2 = extent[0] + ((x + box_size) / w) * (extent[1] - extent[0])
        lat2 = extent[3] - ((y + box_size) / h) * (extent[3] - extent[2])
        # Convert geographic coordinates back to pixel coordinates in the second image
        x2_1 = int((lon1 - extent2[0]) / (extent[1] - extent2[0]) * w2)
        y2_1 = int((extent2[3] - lat1) / (extent2[3] - extent2[2]) * h2)
        x2_2        = int((lon2 - extent2[0]) / (extent2[1] - extent2[0]) * w2)
        y2_2        = int((extent2[3] - lat2) / (extent2[3] - extent2[2]) * h2)

        box2 = data2[y2_1:y2_2, x2_1:x2_2]

        if np.isfinite(box).all() and np.isfinite(box2).all():
            happy = True

        if attempts > max_attemps:
            print(f"   Could not find a valid box after {max_attemps} attempts, returning NaNs for box2")
            return box, None, (y, y+box_size, x, x+box_size), [None]*4, (lon1, lat1, lon2, lat2)

    print(f"Found valid box after {attempts} attempts:")
    print(f"    Box in image 1 at pixel coordinates: y={y}:{y+box_size}, x={x}:{x+box_size}")
    print(f"    Box in image 2 at geographic coordinates: lon1={lon1}, lat1={lat1}, lon2={lon2}, lat2={lat2}")
    print(f"    Box in image 2 at pixel coordinates: y={y2_1}:{y2_2}, x={x2_1}:{x2_2}")

    return box, box2, (y, y+box_size, x, x+box_size), (y2_1, y2_2, x2_1, x2_2), (lon1, lat1, lon2, lat2)


# Give`` me the actial lat/lon extent of the box
# I'M HAVING TO DROP BY ONE GRID CELL HERE, CHECK THIS LATER
def interpolate_boxes(box, box2, lon1, lat1, lon2, lat2):
    
    dx = (lon2 - lon1) / box.shape[1]
    dy = (lat2 - lat1) / box.shape[0]

    assert dx == -dy

    lat_grid  = np.linspace(lat1, lat2-dy, box.shape[0])
    lon_grid  = np.linspace(lon1, lon2-dx, box.shape[1])

    lat_grid2 = np.linspace(lat1, lat2-dy, box2.shape[0])
    lon_grid2 = np.linspace(lon1, lon2-dx, box2.shape[1])

    # Interpolate box2 onto the grid of box1
    from scipy.interpolate import griddata
    lon_grid, lat_grid   = np.meshgrid(lon_grid, lat_grid)
    lon_grid2, lat_grid2 = np.meshgrid(lon_grid2, lat_grid2)
    # box2_interp          = griddata((lon_grid2.flatten(), lat_grid2.flatten()), box2.flatten(), (lon_grid, lat_grid), method='linear')

    myGrid = aerial_imagery.geotiff.RegGrid([lon1, lon2], [lat1, lat2], dx, dy)

    boxI = myGrid.griddata(lon_grid, lat_grid, box)
    box2I = myGrid.griddata(lon_grid2, lat_grid2, box2)

    return boxI, box2I, dx, dy


big_N = len(ds_exif)-1
big_N = 50 

boxes1, boxes2 = [], []
times1, times2 = [], []
min_lons, min_lats, max_lons, max_lats = [], [], [], []
time_processed = []

for i in range(0, big_N, 1):
    print(f'Processing image pair index: {i} and {i+image_seq}')
    data, extent, data2, extent2, time, time2 = get_data(ds_exif, i, image_seq)

    print(f'get_data line complete for times {time} and {time2}.')

    box, box2, (y1, y2, x1, x2), (y21, y22, x21, x22), (lon1, lat1, lon2, lat2) = find_box(data, extent, data2, extent2, box_size=box_size)
    print((lon1, lat1, lon2, lat2))

    dx = (lon2 - lon1) / box.shape[1]
    dy = (lat2 - lat1) / box.shape[0]
    print(f"dx={dx}, dy={dy}")

    if True:
        plt.figure(figsize=(10, 12)) # Adjust figure size as needed
        plt.suptitle('GeoTIFF Visualization')

        ax = plt.subplot(2, 2, 1)
        plt.imshow(data, cmap='Spectral', extent=extent, origin='upper') # 'viridis' is a common colormap
        plt.colorbar(label='Pixel Value') # Add a color bar to indicate pixel values
        plt.title('Image 1')
        plt.xlabel('Easting')
        plt.ylabel('Northing')
        yl, xl = plt.ylim(), plt.xlim()
        plt.plot([lon1, lon2, lon2, lon1, lon1], [lat1, lat1, lat2, lat2, lat1], 'r-')

        ax = plt.subplot(2, 2, 2)
        plt.imshow(data2, cmap='Spectral', extent=extent2, origin='upper') # 'viridis' is a common colormap
        plt.colorbar(label='Pixel Value') # Add a color bar to indicate pixel values
        plt.title('Image 2')
        plt.xlabel('Easting')
        plt.ylabel('Northing')
        plt.xlim(xl)
        plt.ylim(yl)
        plt.plot([lon1, lon2, lon2, lon1, lon1], [lat1, lat1, lat2, lat2, lat1], 'r-', label='Randomly placed 1.5 km for spectral calcs')

        # image_name = os.path.join(output_dir, f'{exif_nc_stub}_search_box_[pair_{i:03d}].png')

        # print(f'Saved image to {image_name}')

    if not box2 is None:
        ll_ext = (lon1, lon2, lat1, lat2)

        # Plot the extracted boxes
        # plt.figure(figsize=(10, 4))
        ax = plt.subplot(2, 2, 3)
        plt.suptitle(f'Extracted Boxes are [{box.shape[0]} x {box.shape[1]}] pixels i.e. {box.shape[0]*dx} m')
        plt.imshow(box, cmap='Spectral', origin='upper', extent=ll_ext) # 'viridis' is
        plt.colorbar(label='Pixel Value') # Add a color bar to indicate pixel values
        plt.title('Extracted Box from Image 1' )
        plt.xlabel('Easting')
        plt.ylabel('Northing')

        ax = plt.subplot(2, 2, 4)
        plt.imshow(box2, cmap='Spectral', origin='upper', extent=ll_ext) # 'viridis'
        plt.colorbar(label='Pixel Value') # Add a color bar to indicate pixel values
        plt.title('Extracted Box from Image 2')
        plt.xlabel('Easting')
        plt.ylabel('Northing')
        plt.tight_layout()
        
        image_name = os.path.join(output_dir, f'{exif_nc_stub}_search_box_[pair_{i:03d}].png')
        
        plt.tight_layout()
        plt.savefig(image_name)

    print('find_box line complete.')


    if box2 is None:
        box2 = np.full_like(box, np.nan)

    else:
        # I think we need better interpolation than this SFODA one
        box, box2, dx, dy = interpolate_boxes(box, box2, lon1, lat1, lon2, lat2)

    assert box.shape == box2.shape, "Extracted boxes must have the same shape, or interp needed"

        
    print('interpolate_boxes line complete.')

    
    # Now I want to stack these boxes into a 4D array for processing later
    boxes1.append(box)
    boxes2.append(box2)
    times1.append(time)
    times2.append(time2)
    min_lons.append(lon1)
    min_lats.append(lat1)
    max_lons.append(lon2)
    max_lats.append(lat2)
    time_processed.append(datetime.utcnow())

    print('Just logged times of {} and {}'.format(time, time2))

boxes1 = np.stack(boxes1)
boxes2 = np.stack(boxes2)
times1 = np.array(times1)
times2 = np.array(times2)
min_lons = np.array(min_lons)
min_lats = np.array(min_lats)
max_lons = np.array(max_lons)
max_lats = np.array(max_lats)
time_processed = np.array(time_processed)

print(times1)
print(times2)

# Now make an nc file with these boxes and times
# Also include the lat/lon extents
# Also include the attrs suxch as dx, dy for reference
ds_boxes = xr.Dataset(
    {
        'box1': (('pair', 'y', 'x'), boxes1),
        'box2': (('pair', 'y', 'x'), boxes2),
        'time1': (('pair',), times1),
        'time2': (('pair',), times2),
        'min_lon': (('pair',), [lon1]*boxes1.shape[0]),
        'min_lat': (('pair',), [lat1]*boxes1.shape[0]),
        'max_lon': (('pair',), [lon2]*boxes1.shape[0]),
        'max_lat': (('pair',), [lat2]*boxes1.shape[0]),
        'time_processed': (('pair',), time_processed.astype('datetime64[ns]')),
    },
    coords={
        'pair': np.arange(boxes1.shape[0]),
        'y': np.arange(boxes1.shape[1]),
        'x': np.arange(boxes1.shape[2]),
    }
)

ds_boxes.attrs['extracted_box_size'] = box_size
ds_boxes.attrs['dx'] = dx
ds_boxes.attrs['dy'] = dy

ds_boxes.to_netcdf(nc_outfile)
print(f'Saved extracted boxes to {nc_outfile}')
print(f'gOT UP TO FRAME {i} with no issues')

print(min_lons)