#!/usr/bin/env python 

"""
Script to generate a rotated, high-resolution bathymetry file for MITgcm simulations
focused on the Grande-Anse terminal (Port Saguenay, Québec).

This script performs the following steps:
1. Loads a high-resolution bathymetric survey around the Grande-Anse wharf (in MTM Zone 7).
2. Loads a lower-resolution bathymetric map of the Saguenay fjord (LAT, LON, Z from .mat file).
3. Converts all coordinates to MTM (EPSG:2949) and rotates them so that the wharf becomes aligned
   with the southern edge of the model grid.
4. Interpolates both datasets onto a common high-resolution grid (1 m).
5. Merges the datasets by preserving the Grande-Anse data where available.
6. Applies a digitized land mask (created manually) to exclude non-ocean regions.
7. Produces a masked bathymetry ready to be exported to MITgcm (.bin files).

Input files:
- High-res .xyz survey file (x, y in meters, z in meters)
- Wharf coordinates (.dat file with 2 points)
- Low-res .mat file with LAT, LON, Z variables
- Digitized shoreline mask (.dat file from DigitTable)

Output:
- 2D bathymetry array aligned with the rotated wharf and suitable for MITgcm
- Optional visualization to inspect bathymetry and masking

Author: Sandy Gregorio
Date: June 2025
"""

### === Import modules === ###
import pandas                 as pd
import numpy                  as np
import matplotlib.pyplot      as plt
import readwrite_input_mitgcm as rw
from scipy.io                 import loadmat
from bathy_tools              import (
    rotate_topography,
    rotate_coordinates,
    interpolate_to_grid,
    apply_mask,
    convert_latlon_to_mtm
)

### === Define file paths === ###
rdir   = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/topo_GA/'
grid   = rdir + 'CA0011760.8527_BATHY_GRANDEANSE_2023-09-27_CUBE_0.3m.xyz'
wharf  = rdir + 'Wharf_Grande-Anse.dat'
fmask  = rdir + 'Grande-Anse_mask.dat'
fjord  = rdir + 'Fjord_50m.mat' 

### === Load input data === ###
# Load Grande-Anse bathymetry (MTM Zone 7 projection)
df     = pd.read_csv(grid, sep='\s+', header=None, names=['x', 'y', 'z'])

# Load wharf coordinates (2 points)
df_w   = pd.read_csv(wharf, sep='\s+', header=None, names=['x', 'y', 'val'])
x1, y1 = df_w['x'][0], df_w['y'][0]
x2, y2 = df_w['x'][1], df_w['y'][1]
x_center = (x1 + x2) / 2
y_center = (y1 + y2) / 2
origin = np.array([x_center, y_center])

# Load low-resolution fjord bathymetry from .mat file (LAT, LON, Z)
mat    = loadmat(fjord)
LAT    = mat['LAT']
LON    = mat['LON']
Z      = mat['Z']  
# WGS84 → MTM zone 7 (EPSG:2949)
X, Y   = convert_latlon_to_mtm(LON, LAT)

# Load digitized land mask (from DigitTable in Matlab)
mask_df     = pd.read_csv(fmask, sep='\s+', header=None, names=['x', 'y', 'val'])
mask_coords = mask_df[['x', 'y']].values

### === Process bathymetry === ###
# Rotate Grande-Anse bathymetry
grid_x, grid_y, grid_depth_ga, theta, df_rot = rotate_topography(df, x1, y1, x2, y2, origin=origin)

# Rotate fjord bathymetry into same reference frame
coords_big     = np.column_stack((X.ravel(), Y.ravel()))
coords_big_rot = rotate_coordinates(coords_big, origin, theta)

# Define final model domain (relative to rotated origin)
domain_half_width = 1000  # in meters
res = 1.0
x_hr = np.arange(-domain_half_width, domain_half_width + res, res)
y_hr = np.arange(-100, 1900 + res, res)
grid_x, grid_y = np.meshgrid(x_hr, y_hr)

# Interpolate both datasets onto final grid
z_hr_ga    = interpolate_to_grid(df_rot.values, df['z'], grid_x, grid_y)
z_hr_fjord = interpolate_to_grid(coords_big_rot, Z.ravel(), grid_x, grid_y)

# Combine high-res Grande-Anse and low-res fjord data
z_combined = np.where(~np.isnan(z_hr_ga), z_hr_ga, z_hr_fjord)

# Apply land mask (rotated)
mask_coords_rot = rotate_coordinates(mask_coords, origin, theta)
mask = apply_mask(grid_x, grid_y, mask_coords_rot)
grid_depth_masked = np.where(mask, np.nan, z_combined)

### === Visualization === ###
plt.ion()
plt.figure(figsize=(10, 6))

# Bathymetry
c = plt.pcolormesh(grid_x, grid_y, grid_depth_masked, shading='auto', cmap='viridis')
# Mask
plt.plot(mask_coords_rot[:, 0], mask_coords_rot[:, 1], 'r-', lw=2, label='Digitized shoreline')
plt.colorbar(c, label='Depth (m)')
plt.title('Masked bathymetry (land excluded)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')
#plt.legend()
plt.tight_layout()

### define file to save IC
dir1   = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/MITgcm_runs/ISW4-CTDF14/APE1e5_3D/'
### save variable into binary filename
#rw.write_to_binary(grid_depth, dir1 + 'bathy.bin', precision='double')
#rw.write_to_binary(diffx, dir1 + 'dx.bin', precision='double')



