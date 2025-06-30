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
z_hr_ga    = interpolate_to_grid(df[['x_rot', 'y_rot']].values, df['z'].values, grid_x, grid_y)
z_hr_fjord = interpolate_to_grid(coords_fjord_rot, Z_sub, grid_x, grid_y)

# Combine Grande-Anse and fjord bathymetry
z_combined = np.where(~np.isnan(z_hr_ga), z_hr_ga, z_hr_fjord)

### === Apply land mask === ###

# Rotate the digitized shoreline mask (quay contour)
mask_coords_rot = rotate_coordinates(mask_coords, origin, -theta)

# Rotate extracted coastline contours (from .mat file)
contour_0_rot = rotate_coordinates(np.column_stack((X0_sub, Y0_sub)), origin, -theta)  # North shore
contour_1_rot = rotate_coordinates(np.column_stack((X1_sub, Y1_sub)), origin, -theta)  # South shore

# --- Merge the digitized quay segment with southern shoreline ---

# Build a KDTree from the rotated southern shoreline (contour_1_rot)
tree = cKDTree(contour_1_rot)

# Query the closest point on the shoreline for each digitized mask point
distances, indices = tree.query(mask_coords_rot)

# Select the segment of the digitized mask that corresponds to the wharf
# (manually validated range: mask_coords_rot[24:50])
quay_segment = mask_coords_rot[24:50]

# Get corresponding indices on the southern shoreline
i_start = indices[24]
i_end   = indices[49]

# Ensure the order is increasing for proper slicing
if i_start > i_end:
    i_start, i_end = i_end, i_start

# Extract sections of the southern shoreline before and after the quay
before_quay = contour_1_rot[i_end+1:]
after_quay  = contour_1_rot[:i_start]

# Combine all parts to form the merged southern shoreline
merged_shoreline = np.vstack([
    before_quay,
    quay_segment,
    after_quay
])

# --- Construct the water polygon between north and south shorelines ---

# Define shorelines
south = merged_shoreline          # Southern shoreline (with quay)
north = contour_0_rot[::-1]       # Northern shoreline (reversed to close the polygon clockwise)

# Add artificial vertical connectors to close the polygon as a rectangle
pt_east = [south[-1][0], north[0][1]]   # Eastern corner (vertical segment)
pt_west = [south[0][0], north[-1][1]]   # Western corner (vertical segment)

# Stack the full polygon: south → east connector → north → west connector
polygon_coords = np.vstack([
    south,        # Step 1: walk along southern shoreline
    pt_east,      # Step 2: connect up to north shore (east side)
    north,        # Step 3: walk along northern shoreline (westward)
    pt_west       # Step 4: connect down to starting point (west side)
])

# --- Apply the polygon mask to exclude land regions ---
mask_polygon      = apply_mask(grid_x, grid_y, polygon_coords)
grid_depth_masked = np.where(mask_polygon, z_combined, np.nan)



# Merge digitized shoreline mask with coastline contours
# Construire l'arbre pour la rive sud
tree = cKDTree(contour_1_rot)
# Chercher le plus proche voisin pour chaque point du quai
distances, indices = tree.query(mask_coords_rot)
# Trouver les deux plus proches
nearest_two = np.argsort(distances)
# using
# Points du quai les plus proches
closest_mask_points = mask_coords_rot[nearest_two]



i_start = indices[2]
i_end   = indices[26]

# Partie avant (rive sud avant le quai)
before_quay = contour_1_rot[:i_start]

# Partie centrale : le quai (segment digitalisé)
quay_segment = mask_coords_rot[2:27]  # [2:26+1]

# Partie après (rive sud après le quai)
after_quay = contour_1_rot[i_end+1:]

# Merger les segments
merged_shoreline = np.vstack([
    before_quay,
    quay_segment,
    after_quay
])

# Build a closed polygon representing the water area between the northern and southern shores.
# The polygon is constructed by:
# 1. Following the southern shoreline eastward (contour_1_rot),
# 2. Adding a vertical connection up to the northern shore (east side),
# 3. Following the northern shoreline westward (contour_0_rot[::-1]),
# 4. Adding a vertical connection back down to the southern shore (west side),
# to form a rectangular envelope around the water domain.

south = merged_shoreline                 # Southern shoreline (eastward)
north = contour_0_rot[::-1]           # Northern shoreline (westward)

# Define artificial vertical corner points to close the polygon as a rectangle
pt_east = [south[-1][0], north[0][1]]  # Top-right corner (east)
pt_west = [south[0][0], north[-1][1]]  # Top-left corner (west)

# Stack all points to form the closed polygon
polygon_coords = np.vstack([
    south,        # Step 1: walk along southern shore
    pt_east,      # Step 2: connect upward to northern shore (east)
    north,        # Step 3: walk along northern shore (reversed)
    pt_west       # Step 4: connect downward to southern shore (west)
])

# Create mask from the polygon and apply it
mask_polygon      = apply_mask(grid_x, grid_y, polygon_coords)
grid_depth_masked = np.where(mask_polygon, z_combined, np.nan)

### === Visualization === ###
plt.ion()
plt.figure(figsize=(10, 6))

# Bathymetry
c = plt.pcolormesh(grid_x, grid_y, grid_depth_masked, shading='auto', cmap='viridis')
plt.colorbar(c, label='Depth (m)')

# Show masks
plt.plot(mask_coords_rot[:, 0], mask_coords_rot[:, 1], 'm-', lw=2, label='Digitized shoreline')
plt.plot(contour_0_rot[:, 0], contour_0_rot[:, 1], 'r-', lw=2, label='Shoreline 0')
plt.plot(contour_1_rot[:, 0], contour_1_rot[:, 1], 'b-', lw=2, label='Shoreline 1')
plt.plot(closest_mask_points[2, 0], closest_mask_points[2, 1], 'o', markersize=8, label='Closest mask→shore')
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



