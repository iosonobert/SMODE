import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import pandas as pd
import math 

import scipy.signal
from scipy.signal import detrend
from scipy.ndimage import distance_transform_edt

from shapely.geometry import shape, MultiPolygon, GeometryCollection
from shapely.ops import unary_union

import rasterio
from rasterio.features import shapes
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

from affine import Affine

def speed_test():
    t0 = time.time()
    arr = np.random.rand(200, 1969, 1983).astype(np.float32)
    print("Seconds to make matrix:", time.time()-t0)
    t0 = time.time()
    np.sum(arr, axis=0)
    print("Seconds to sum matrix:", time.time()-t0)

folder = f''

# These ones are West-East flight lines
exif_nc_stub = 'SMODE_IOP2_MASS_VIDEO_20230419_001018_120-20230419_002139_680'

parent_folder = '/mnt/c/Users/00071913/OneDrive - UWA/Zulberti/my_projects/SMODE/LWIR/'

folder = f'{parent_folder}/{exif_nc_stub}'
files = sorted(glob.glob(os.path.join(folder, '*.L1.tif')))

first_file = 10
file = files[first_file]

os.path.exists(file)

def load_me(file):
    """
    Load a single GeoTIFF file using rasterio.
    Returns the data array, the rasterio dataset object, and the profile metadata.
    """
    with rasterio.open(file) as src:
        data = src.read(1)
        profile = src.profile

    return data, src, profile   


def get_grid_CRS(src1, srcN):
    """Return a massieve grid in the CRS plane that encompasses both src1 and srcN.
    """

    left = min(src1.bounds.left, srcN.bounds.left)
    right = max(src1.bounds.right, srcN.bounds.right)
    bottom = min(src1.bounds.bottom, srcN.bounds.bottom)
    top = max(src1.bounds.top, srcN.bounds.top)

    res = src1.res[0]

    width_0N = int(np.ceil((right - left) / res))
    height_0N = int(np.ceil((top - bottom) / res))

    transform_0N = from_origin(left, top, res, res)

    return transform_0N, width_0N, height_0N


def centroid_of_bounds(bounds):
    left, bottom, right, top = bounds
    return ( (left + right) / 2.0, (bottom + top) / 2.0 )

def rotation_matrix(theta_rad):
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    # rotation by +theta: [ [c, -s], [s, c] ]
    return np.array([[c, -s],
                     [s,  c]])

def rotate_points(xs, ys, theta_rad):
    """Rotate points (xs, ys) by theta_rad around origin (0,0).
       Returns arrays of x', y'."""
    R = rotation_matrix(theta_rad)
    pts = np.vstack([xs, ys])
    rotated = R.dot(pts)
    return rotated[0, :], rotated[1, :]


def build_transect_grid_transform_and_size(files, pad_m=100.0, res=None):
    """files: list of files (first, last at least).
       pad_m: padding in meters (or same units as CRS).
       res: desired pixel size (if None, average source resolution used)."""
    
    _, src0, _ = load_me(files[0])
    _, srcN, _ = load_me(files[-1])
    dsets = [src0, srcN]

    # collect all corner coords of datasets
    corners_x = []
    corners_y = []
    resolutions = []
    for ds in dsets:
        left, bottom, right, top = ds.bounds
        corners_x.extend([left, left, right, right])
        corners_y.extend([bottom, top, bottom, top])
        # approximate resolution: use transform.a and transform.e (abs)
        res_x = abs(ds.transform.a)
        res_y = abs(ds.transform.e)
        resolutions.append(max(res_x, res_y))
    # compute centroids of first & last for transect direction
    x0, y0 = centroid_of_bounds(dsets[0].bounds)
    x1, y1 = centroid_of_bounds(dsets[-1].bounds)
    dx = x1 - x0
    dy = y1 - y0
    angle_rad = math.atan2(dy, dx)  # angle of transect in radians (CRS coords)
    angle_deg = math.degrees(angle_rad)

    # rotate all corners by -angle to align transect with x-axis
    # rotation by -angle: apply rotation_matrix(-angle)
    xs_rot, ys_rot = rotate_points(np.array(corners_x), np.array(corners_y), -angle_rad)
    min_xr, max_xr = xs_rot.min(), xs_rot.max()
    min_yr, max_yr = ys_rot.min(), ys_rot.max()

    # pad and resolution
    min_xr -= pad_m
    max_xr += pad_m
    min_yr -= pad_m
    max_yr += pad_m
    if res is None:
        res = float(np.mean(resolutions))

    width = int(math.ceil((max_xr - min_xr) / res))
    height = int(math.ceil((max_yr - min_yr) / res))

    # Build destination transform: pixel -> original CRS
    # 1) pixel -> rotated coords: Affine(res, 0, min_xr, 0, -res, max_yr)
    t_pixel_to_rot = Affine(res, 0.0, min_xr, 0.0, -res, max_yr)
    # 2) rotated coords -> original CRS: rotation by +angle
    t_rot_to_orig = Affine.rotation(math.degrees(angle_rad))
    # Combined transform: pixel -> original CRS
    dst_transform = t_rot_to_orig * t_pixel_to_rot

    return {
        "dst_transform": dst_transform,
        "dst_width": width,
        "dst_height": height,
        "angle_rad": angle_rad,
        "angle_deg": angle_deg,
        "resolution": res,
        "bounds_rotated": (min_xr, min_yr, max_xr, max_yr)
    }


def reproject_to_target(src_array, src_transform, src_crs, target_shape, target_transform, target_crs):
    dst = np.zeros(target_shape, dtype=src_array.dtype) * np.nan
    reproject(
        source=src_array,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear,
        dst_nodata=np.nan
    )
    return dst



def pre_feather(files, transform_0N, width_0N, height_0N, verbose=False):

    _, src0, _ = load_me(files[0])
    _, srcN, _ = load_me(files[-1])

    # Would be better to preallocate the np arrays
    raw_data_list = []
    mask_stack   = np.empty((len(files), height_0N, width_0N), dtype=np.float32)
    data_stack   = np.empty((len(files), height_0N, width_0N), dtype=np.float32)
    weight_stack = np.empty((len(files), height_0N, width_0N), dtype=np.float32)

    all_minima = np.inf
    all_minima_resampled = np.inf
    # for i, (file, data, src) in enumerate(zip(flies_ss__, datas, srcs)):

    nan_before_resample = True
    t0 = time.time()

    print(f'Pre-processing files with verbose={verbose} [warning in place if file takes much longer than usual]')
    all_file_dt = []

    for i, file in enumerate(files):

        if verbose:
            print("   ", i, file)

        t1 = time.time()
        data, src, profile = load_me(file)
        data = data.astype(np.float32)

        # Nan the data before reproject to exclude edge data becoming half ghosted in resampling
        if nan_before_resample:
            mask           = data           != src.nodata
            data[~mask]    = np.nan
        else:
            raise NotImplementedError("I removed this option")
        
        data_resampled = reproject_to_target(
            data,
            src.transform,
            src.crs,
            (height_0N, width_0N),
            transform_0N,
            src.crs
        )

        data_resampled = data_resampled.astype(np.float32)

        if nan_before_resample:
            mask_resampled1 = data_resampled != src.nodata # This doesn't work if you nan nodata before resampling
            mask_resampled2 = ~np.isnan(data_resampled) 
            mask_resampled = mask_resampled1 & mask_resampled2
        else:
            mask_resampled = data_resampled != src.nodata # This doesn't work if you nan nodata before resampling
        
        data_resampled[~mask_resampled] = np.nan

        all_minima = min(all_minima, np.nanmin(data))
        all_minima_resampled = min(all_minima_resampled, np.nanmin(data_resampled))

        if nan_before_resample:    # This is really slow in big masks, but necessary if we nan before resampling
            weight_resampled = distance_transform_edt(mask_resampled)
        else: 
            weight = distance_transform_edt(mask)
            weight_resampled = reproject_to_target(
                weight,
                src.transform,
                src.crs,
                (height_0N, width_0N),
                transform_0N,
                src.crs
            )

        mask_stack[i, :, :]   = mask_resampled.astype(np.uint8)
        data_stack[i, :, :]   = data_resampled
        weight_stack[i, :, :] = weight_resampled

        file_dt = time.time()-t1
        if verbose:
            print("    ", f"Seconds to pre-process 1 files:", file_dt)

        all_file_dt.append(file_dt)
        mean_prev = np.mean(all_file_dt)
        if i == 10:
            mean_prev10 = mean_prev
            print(f"Note: after {i} files, mean of previous times is {mean_prev10:.2f}s.")
            print(f"Total time estimate is then {mean_prev10 * len(files) :.2f} seconds.")
            print(f"       (subsequent warnings if any indicate files taking much longer than this)")
        elif i > 10:
            update_perc = 1.1
            if mean_prev / mean_prev10 > update_perc or mean_prev10 / mean_prev > update_perc:
                mean_prev10 = mean_prev
                print('MEAN TIME UPDATED')
                print(f"Note: after {i} files, UPDATED mean of previous times is {mean_prev10:.2f}s.")
                print(f"Total time estimate is then {mean_prev * len(files) :.2f} seconds.")
                print(f"       (subsequent warnings if any indicate files taking much longer than this)")

        if i > 10:
            # Check if time of file is much greater than median of previous times
            median_prev = np.median(all_file_dt)
            if file_dt > 3 * median_prev:   
                print(f"WARNING: file {i} {file} took {file_dt:.2f}s which is much greater than median of previous times {median_prev:.2f}s.")

        raw_data_list.append(data)

    print(f"Seconds to pre-process {len(files)} file:", time.time()-t0)

    return mask_stack, data_stack, weight_stack, all_minima, all_minima_resampled, raw_data_list

def feather(data_stack, weight_stack):
    """Just uses np.nansum for now. Can replace with safe_sum if needed later for very large arrays.
    """

    nt, h, w = data_stack.shape
    t0 = time.time()

    sum_of_weight = np.nansum(weight_stack, axis=0)

    sum_of_data          = np.nansum(data_stack, axis=0) 
    weighted_sum_of_data = np.nansum(data_stack * weight_stack, axis=0) 
    weighted_avg_of_data = np.nansum(data_stack * weight_stack, axis=0) / sum_of_weight

    print(f"Seconds to feather [nt x h x w] = [{nt} x {h} x {w}] arrays:", time.time()-t0)

    # take the gradients of that too

    return sum_of_weight, sum_of_data, weighted_sum_of_data, weighted_avg_of_data

def check_ghosts(all_minima, all_minima_resampled, data_stack, weighted_sum_of_data, weighted_avg_of_data):
    
    print("  ", "The smallest value across all data is:", all_minima)
    print("  ", "The smallest value across all resampled data is:", all_minima_resampled)
    print("  ")
    print("  ", "Min value in data_stack:", np.nanmin(data_stack))
    print("  ", "Min value in sum_of_data:", np.nanmin(weighted_avg_of_data))
    print("  ", "Min weighted_sum_of_data:", np.nanmin(weighted_sum_of_data))
    print("  ", "Min weighted_avg_of_data:", np.nanmin(weighted_avg_of_data))


def get_geotiff_filename(folder, parent_folder, exif_nc_stub, proc_type='weighted_avg', append_str=''):
    filename = os.path.join(folder, f'{parent_folder}/[{exif_nc_stub}]_[{proc_type}]_[{append_str}].tif')
    return filename

def save_geotiffs(weighted_avg_of_data, sum_of_weight, folder, parent_folder, exif_nc_stub, crs, height_0N, width_0N, transform_0N, append_str=''):
    t0 = time.time()
    weighted_avg_file = get_geotiff_filename(folder, parent_folder, exif_nc_stub, proc_type='weighted_avg', append_str=append_str)

    with rasterio.open(weighted_avg_file, 'w', 
        driver="GTiff",
        height=height_0N,
        width=width_0N,
        count=1,
        dtype=rasterio.float32,
        crs=crs,
        transform=transform_0N,
        compress="lzw",
        nodata=np.nan,
    ) as dst:
        dst.write(weighted_avg_of_data, 1)
    print(f"  Seconds to save weighted_avg_of_data GeoTIFF:", time.time()-t0)

    if not sum_of_weight is None:
        t0 = time.time()
        output_file = get_geotiff_filename(folder, parent_folder, exif_nc_stub, proc_type='sum_of_weight', append_str=append_str)

        with rasterio.open(output_file, 'w', 
            driver="GTiff",
            height=height_0N,
            width=width_0N,
            count=1,
            dtype=rasterio.float32,
            crs=crs,
            transform=transform_0N,
            compress="lzw",
            nodata=np.nan,
        ) as dst:
            dst.write(sum_of_weight, 1)
        print(f"  Seconds to save sum_of_weight GeoTIFF:", time.time()-t0)

    # # And the regular average too
    # output_file = os.path.join(folder, f'{parent_folder}/{exif_nc_stub}_avg_[{start}_{end}_{skip}]_v2.tif')

    # with rasterio.open(output_file, 'w', 
    #     driver="GTiff",
    #     height=height_0N,
    #     width=width_0N,
    #     count=1,
    #     dtype=rasterio.float32,
    #     crs=crs,
    #     transform=transform_0N,
    #     compress="lzw",
    #     nodata=np.nan,
    # ) as dst:
    #     dst.write(avg_of_data, 1)

    return weighted_avg_file