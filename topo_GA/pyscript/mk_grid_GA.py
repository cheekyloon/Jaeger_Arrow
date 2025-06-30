#!/usr/bin/env python 

"""
Script to generate a rotated, high-resolution bathymetry file for MITgcm simulations
focused on the Grande-Anse terminal (Port Saguenay, Québec).

This script performs the following steps:
1. Loads a high-resolution bathymetric survey around the Grande-Anse wharf (in MTM Zone 7).
2. Loads a lower-resolution bathymetric map of the Saguenay Fjord (LAT, LON, Z from .mat file).
3. Converts all coordinates to MTM Zone 7 (EPSG:2949) and rotates the domain so that
   the wharf aligns with the southern edge of the model grid.
4. Interpolates both datasets onto a common high-resolution grid (1 m spacing).
5. Merges the datasets, preserving the high-resolution survey where available.
6. Applies a digitized shoreline mask to exclude land areas from the domain.
7. Produces a masked bathymetry field ready for use in MITgcm (.bin format).

Input files:
- High-resolution .xyz survey file (x, y in meters, z in meters relative to NMMB)
- Wharf coordinates (.dat file with 2 reference points)
- Low-resolution .mat file with LAT, LON, Z variables from CHS NONNA bathymetry
- Digitized shoreline mask (.dat file created manually using DigitTable)

Vertical references:
- High-resolution survey is referenced to mean low water level (NMMB),
  which is 2.76 meters below CGVD28.
- CHS NONNA bathymetry is referenced to chart datum (CD).
  [Source: https://open.canada.ca/data/en/dataset/d3881c4c-650d-4070-bf9b-1e00aabf0a1d]

Output:
- 2D masked bathymetry array aligned with the rotated domain
- Optional visualization to inspect interpolation and masking steps

Author: Sandy Gregorio
Date: June 2025
"""

### === Import modules === ###
import pandas                 as pd
import numpy                  as np
import matplotlib.pyplot      as plt
import readwrite_input_mitgcm as rw
from scipy.io                 import loadmat
from scipy.spatial            import cKDTree
from bathy_tools              import (
    compute_rotation_angle,
    rotate_coordinates,
    interpolate_to_grid,
    apply_mask,
    convert_latlon_to_mtm
)

### === Define file paths === ###
rdir    = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/topo_GA/'
HR_grid = rdir + 'CA0011760.8527_BATHY_GRANDEANSE_2023-09-27_CUBE_0.3m.xyz'
LR_grid = rdir + 'Fjord_50m.mat' 
wharf   = rdir + 'Wharf_Grande-Anse.dat'
fmask   = rdir + 'Grande-Anse_mask.dat'
fcoast  = rdir + 'Saguenay_coastline.mat'

### === Load input data === ###
# High-resolution Grande-Anse bathymetry (projected in MTM Zone 7)
df = pd.read_csv(HR_grid, sep='\s+', header=None, names=['x', 'y', 'z'])

# Low-resolution fjord bathymetry (LAT, LON, Z)
mat = loadmat(LR_grid)
LAT = mat['LAT']
LON = mat['LON']
Z   = mat['Z']
X, Y = convert_latlon_to_mtm(LON, LAT)  # WGS84 → MTM zone 7

# Wharf endpoints (2 points, to define orientation)
df_w = pd.read_csv(wharf, sep='\s+', header=None, names=['x', 'y', 'val'])
x1, y1 = df_w['x'][0], df_w['y'][0]
x2, y2 = df_w['x'][1], df_w['y'][1]

# Digitized mask from Matlab DigitTable
mask_df     = pd.read_csv(fmask, sep='\s+', header=None, names=['x', 'y', 'val'])
mask_coords = mask_df[['x', 'y']].values

# Saguenay coastline (lat_coastline_Sag, lon_coastline_Sag)
mat = loadmat(fcoast)
LAT = mat['lat_coastline_Sag']
LON = mat['lon_coastline_Sag']
X_coast, Y_coast = convert_latlon_to_mtm(LON, LAT)

### === Prepare coordinate transformations === ###
# Compute rotation angle to align the wharf horizontally
theta = compute_rotation_angle(x1, y1, x2, y2)

# Define rotation origin (midpoint of the wharf)
origin = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

# Define bounding box around model domain in original coordinates
x_min, x_max = origin[0] - 1100, origin[0] + 1100
y_min, y_max = origin[1] - 500, origin[1] + 2200

# Filter fjord bathymetry to local domain
mask_in_fjord = (
    (X >= x_min) & (X <= x_max) &
    (Y >= y_min) & (Y <= y_max)
)
X_sub = X[mask_in_fjord]
Y_sub = Y[mask_in_fjord]
Z_sub = Z[mask_in_fjord]

# Filter coastline contours to local domain
mask_coast0 = (
    (X_coast[:, 0] >= x_min) & (X_coast[:, 0] <= x_max) &
    (Y_coast[:, 0] >= y_min) & (Y_coast[:, 0] <= y_max)
)
mask_coast1 = (
    (X_coast[:, 1] >= x_min) & (X_coast[:, 1] <= x_max) &
    (Y_coast[:, 1] >= y_min) & (Y_coast[:, 1] <= y_max)
)
X0_sub, Y0_sub = X_coast[:, 0][mask_coast0], Y_coast[:, 0][mask_coast0]
X1_sub, Y1_sub = X_coast[:, 1][mask_coast1], Y_coast[:, 1][mask_coast1]


### === Rotate and interpolate bathymetry === ###
# Rotate high-resolution bathymetry
coords_ga     = df[['x', 'y']].values
coords_ga_rot = rotate_coordinates(coords_ga, origin, -theta)
df['x_rot'], df['y_rot'] = coords_ga_rot[:, 0], coords_ga_rot[:, 1]

# Rotate low-resolution fjord bathymetry
coords_fjord     = np.column_stack((X_sub, Y_sub))
coords_fjord_rot = rotate_coordinates(coords_fjord, origin, -theta)

# Define final model grid (in rotated coordinates)
res = 1.0
domain_half_width = 1000
y_0   = 300
y_end = 1900
x_hr, y_hr = np.arange(-domain_half_width, domain_half_width + res, res), np.arange(-y_0, y_end + res, res)
grid_x, grid_y = np.meshgrid(x_hr, y_hr)

# Interpolate both datasets onto final grid
z_hr_ga    = interpolate_to_grid(df[['x_rot', 'y_rot']].values, df['z'].values, grid_x, grid_y) - 2.76
#z_hr_ga    = interpolate_to_grid(df[['x_rot', 'y_rot']].values, df['z'].values, grid_x, grid_y)
z_hr_fjord = interpolate_to_grid(coords_fjord_rot, Z_sub, grid_x, grid_y)

# Combine Grande-Anse and fjord bathymetry
z_combined = np.where(~np.isnan(z_hr_ga), z_hr_ga, z_hr_fjord)


### === Apply land mask === ###

# Rotate the digitized shoreline mask (quay contour)
mask_coords_rot = rotate_coordinates(mask_coords, origin, -theta)

# Rotate extracted coastline contours from the .mat file
contour_0_rot = rotate_coordinates(np.column_stack((X0_sub, Y0_sub)), origin, -theta)  # North shore
contour_1_rot = rotate_coordinates(np.column_stack((X1_sub, Y1_sub)), origin, -theta)  # South shore

# --- Merge the digitized quay segment with the southern shoreline ---

# Build a KDTree for the southern shoreline to find nearest points
tree = cKDTree(contour_1_rot)
distances, indices = tree.query(mask_coords_rot)

# Manually select the quay segment from the digitized mask (validated range)
quay_segment = mask_coords_rot[24:50][::-1]  # Reversed to match shoreline orientation

# Get corresponding index range on the southern shoreline
i_start = indices[24]
i_end   = indices[49]

# Ensure correct slicing order (west to east)
if i_start > i_end:
    i_start, i_end = i_end, i_start

# Extract shoreline segments before and after the quay
# Note: contour_1_rot is assumed to be ordered west → east
before_quay = contour_1_rot[:i_start]      # Segment before quay (west side)
after_quay  = contour_1_rot[i_end+1:]      # Segment after quay (east side)

# Combine the full southern shoreline including the quay
merged_shoreline = np.vstack([
    before_quay,
    quay_segment,
    after_quay
])

# --- Build a closed polygon enclosing the water region ---

# The polygon follows:
# 1. The southern shoreline eastward (with quay inserted),
# 2. A vertical connector up to the northern shore (east side),
# 3. The northern shoreline westward (reversed),
# 4. A vertical connector down to the southern shore (west side),
# forming a closed rectangular-like loop.

# Define southern and northern shorelines
south = merged_shoreline
north = contour_0_rot[::-1]  # Reversed to maintain clockwise orientation

# Add vertical connectors at both ends of the polygon
pt_east = [south[-1][0], north[0][1]]   # Eastern connector (top-right corner)
pt_west = [south[0][0], north[-1][1]]   # Western connector (top-left corner)

# Combine all segments to form the final polygon
polygon_coords = np.vstack([
    south,        # Step 1: southern shore (incl. quay)
    pt_east,      # Step 2: vertical connector up (east)
    north,        # Step 3: northern shore (westward)
    pt_west       # Step 4: vertical connector down (west)
])

# --- Apply the mask based on the polygon ---
mask_polygon      = apply_mask(grid_x, grid_y, polygon_coords)
grid_depth_masked = np.where(mask_polygon, z_combined, np.nan)

### === Visualization === ###
plt.ion()
plt.figure(figsize=(10, 6))

# Bathymetry
c = plt.pcolormesh(grid_x, grid_y, grid_depth_masked, shading='auto', cmap='viridis')
plt.colorbar(c, label='Depth (m)')

# Show masks
#plt.plot(mask_coords_rot[:, 0], mask_coords_rot[:, 1], 'm-', lw=2, label='Digitized shoreline')
plt.plot(contour_0_rot[:, 0], contour_0_rot[:, 1], 'r-', lw=2, label='Shoreline 0')
#plt.plot(contour_1_rot[:, 0], contour_1_rot[:, 1], 'b-', lw=2, label='Shoreline 1')
plt.plot(merged_shoreline[:, 0], merged_shoreline[:, 1], 'k-', lw=2, label='Shoreline 1')
# Axis and labels
plt.xlim(-1100, 1100)
plt.ylim(-500, 2200)
plt.title('Masked bathymetry (land excluded)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')
plt.tight_layout()

### define file to save IC
dir1   = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/MITgcm_runs/ISW4-CTDF14/APE1e5_3D/'
### save variable into binary filename
#rw.write_to_binary(grid_depth, dir1 + 'bathy.bin', precision='double')
#rw.write_to_binary(diffx, dir1 + 'dx.bin', precision='double')



