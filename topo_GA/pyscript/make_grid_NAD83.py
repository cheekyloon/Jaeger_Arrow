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
import pandas as pd
import numpy  as np
from scipy.interpolate import griddata
from pyproj import Proj, Transformer
#import matplotlib.pyplot as plt
#import cartopy.crs       as ccrs
#import cartopy.feature   as cfeature

### set root path
rdir  = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/topo_GA/'
### file to load
fname = 'CA0011760.8527_BATHY_GRANDEANSE_2023-09-27_CUBE_0.3m.xyz'
fdir  = rdir + fname

# Load the .xyz file
# Assuming the file has no headers and space-separated values
# sep='\s+' split the data by whitespace characters (spaces or tabs)
df = pd.read_csv(fdir, sep='\s+', header=None, names=['x', 'y', 'z'])

# Convert MTM (Zone 7) coordinates to Latitude/Longitude
# Define the MTM Zone 7 (CSRS) (EPSG code 2949) and the target WGS84 (EPSG code 4326) CRS
mtm_proj = Proj("EPSG:2949")  # NAD83 / MTM zone 7
wgs84_proj = Proj("EPSG:4326")  # WGS84 (Lat/Lon)

# Create a transformer object to transform between MTM Zone 7 (NAD83) and WGS84 (Lat/Lon)
transformer = Transformer.from_proj(mtm_proj, wgs84_proj)

# Apply the transformation for each x, y coordinate
#df['latitude'], df['longitude'] = transformer.transform(df['x'].values, df['y'].values)

# Build a regular grid
reso     = 1
x_grid = np.arange(round(df['x'].min()), round(df['x'].max())+reso, reso)
y_grid = np.arange(round(df['y'].min()), round(df['y'].max())+reso, reso)
grid_x, grid_y = np.meshgrid(x_grid, y_grid)
grid_depth = griddata(
    (df['x'], df['y']),   # Input coordinates
    df['z'],              # Input depths
    (grid_x, grid_y),     # Grid coordinates
    method='linear',      # Or 'nearest' or 'cubic'
    fill_value=0          # Default depth for missing points
)

grid_lat, grid_lon = transformer.transform(grid_x, grid_y)


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


