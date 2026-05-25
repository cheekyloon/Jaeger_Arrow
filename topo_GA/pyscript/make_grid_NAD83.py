#!/usr/bin/env python 

# Load .xyz file from Port Saguenay
# with the Grande-Anse topgraphy
# The file is three column x,y,z
# The projection of x and y is
# NAD83 CSRS / MTM Zone 7
# Équidistance : 1 mètres
# This file convert the NAD83 projection
# to the WGS84 Lat/Lon projection

### Load modules
import pandas                 as pd
import numpy                  as np
import readwrite_input_mitgcm as rw
import matplotlib.pyplot      as plt
from matplotlib.path          import Path
from scipy.interpolate        import griddata
from pyproj                   import Proj, Transformer
#import cartopy.crs       as ccrs
#import cartopy.feature   as cfeature

### set root path where to load the bathymetry
rdir  = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/topo_GA/'
### file to load
fname1 = 'CA0011760.8527_BATHY_GRANDEANSE_2023-09-27_CUBE_0.3m.xyz'
fname2 = 'CA0011760.8527_BATHY_GRANDEANSE_2023-09-27_MINIMUM_5m.xyz'
fdir1 = rdir + fname1
fdir2 = rdir + fname2

### === 1. Charger les données === ###
# Load the .xyz file
# Assuming the file has no headers and space-separated values
# sep='\s+' split the data by whitespace characters (spaces or tabs)
df = pd.read_csv(fdir1, sep='\s+', header=None, names=['x', 'y', 'z'])

# Convert MTM (Zone 7) coordinates to Latitude/Longitude
# Define the MTM Zone 7 (CSRS) (EPSG code 2949) and the target WGS84 (EPSG code 4326) CRS
mtm_proj = Proj("EPSG:2949")  # NAD83 / MTM zone 7
wgs84_proj = Proj("EPSG:4326")  # WGS84 (Lat/Lon)

# Create a transformer object to transform between MTM Zone 7 (NAD83) and WGS84 (Lat/Lon)
transformer = Transformer.from_proj(mtm_proj, wgs84_proj)

# Apply the transformation for each x, y coordinate
#df['latitude'], df['longitude'] = transformer.transform(df['x'].values, df['y'].values)

### === 2. Build a regular grid === ###
# resolution 
reso     = 1
# extrema
x_min, x_max = int(df['x'].min()), int(df['x'].max())+1 
y_min, y_max = int(df['y'].min()), int(df['y'].max())+1
# grid
x_grid = np.arange(x_min, x_max+reso, reso)
y_grid = np.arange(y_min, y_max+reso, reso)
grid_x, grid_y = np.meshgrid(x_grid, y_grid)
# transform in degree
#grid_lat, grid_lon = transformer.transform(grid_x, grid_y)

### === 3. Interpolation === ###
grid_depth = griddata(
    (df['x'], df['y']),   # Input coordinates
    df['z'],              # Input depths
    (grid_x, grid_y),     # Grid coordinates
    method='linear',      # Or 'nearest' or 'cubic'
    fill_value=0          # Default depth for missing points
)

# Grande-Anse_mask.dat obtenu en digitalisant avec Matlab 
fmask = 'Grande-Anse_mask.dat'
mask_df = pd.read_csv(rdir + fmask, sep='\s+', header=None, names=['x', 'y', 'val'])

# Extraire uniquement les colonnes x et y
mask_coords = mask_df[['x', 'y']].values

# Si besoin : fermer le polygone
#if not np.allclose(mask_coords[0], mask_coords[-1]):
#    mask_coords = np.vstack([mask_coords, mask_coords[0]])

mask_path = Path(mask_coords)

# On vérifie pour chaque point de la grille s’il est dans le polygone
points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
mask = mask_path.contains_points(points).reshape(grid_x.shape)

# Masquer les points en dehors du polygone
grid_depth_masked = np.where(mask, 0, grid_depth)

### === 6. Visualisation === ###
plt.ion()
plt.figure(figsize=(10, 6))
c = plt.pcolormesh(x_grid, y_grid, grid_depth_masked, shading='auto', cmap='viridis')
plt.plot(mask_coords[:, 0], mask_coords[:, 1], 'r-', label='Contour digitalisé')
#plt.plot(mask_df['x'], mask_df['y'], 'r-', lw=2, label="Contour digitalisé")
plt.colorbar(c, label='Profondeur (m)')
plt.title('Bathymétrie masquée (zones terrestres exclues)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')
plt.tight_layout()


### define file to save IC
dir1   = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/MITgcm_runs/ISW4-CTDF14/APE1e5_3D/'
### save variable into binary filename
#rw.write_to_binary(grid_depth, dir1 + 'bathy.bin', precision='double')
#rw.write_to_binary(diffx, dir1 + 'dx.bin', precision='double')

#plt.ion()
## Set up the map projection
#fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 5))
## Plot the depth data
#mesh = ax.pcolormesh(df['longitude'].values, df['latitude'].values, df['z'].values, cmap='viridis', transform=ccrs.PlateCarree())

## Add map features
#ax.add_feature(cfeature.COASTLINE)
#ax.add_feature(cfeature.BORDERS, linestyle=':')
#ax.add_feature(cfeature.LAND, facecolor='lightgray')
#ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

## Add a color bar
#cbar = plt.colorbar(mesh, orientation='vertical', pad=0.05, aspect=40)
#cbar.set_label('Depth (m)')

## Set titles and labels
#ax.set_title('Depth Map')
#ax.set_xlabel('Longitude')
#ax.set_ylabel('Latitude')


