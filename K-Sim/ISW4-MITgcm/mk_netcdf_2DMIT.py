#!/usr/bin/env python 

### create xarray of currents 
### extracted from 2D MITgcm
### in a horizontal square box 
### with resolution  
### dx = dy = dz and timestep dt 
### ideally the boat should feel 10 dt 
### if the boat width = 25 m
### => dt = 2.5 s

### import modules
import numpy     as np
import xarray    as xr
from scipy.interpolate import interp1d

# horizontal resolution
dx   = 2 
# Because my grid along x has a resolution of 0.5 m
# I need to extract data along x every 4 data
nx   = 4
# vertical resolution 
dz   = 2
# timestep 
dt   = 15
# Because deltaT = 0.5 s, I need to extract my
# data along the time every 3 data
nt   = 3
# domain for the wave
# length
LISW = 250
# depth 
zmax = -20.25
# choose time index to start 
# just before ISW
TISW = 910

### file to get MITgcm simulations
# root directory
rdir = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/MITgcm_runs/ISW4-CTDF14/'
# experiment directory
edir = 'APE1e5/'
# grid file
gMIT = f"{rdir}{edir}grid.glob.nc" 
# diagnostics file
fMIT = f"{rdir}{edir}dynDiag.{edir[:-1]}.nc" 

### load grid
with xr.open_dataset(gMIT) as grid:
    Z_values = grid.Z.values
 
### Open dataset and extract u-velocity 
with xr.open_dataset(fMIT) as ds:
    u = (
        ds.UVELMASS.isel(Y=0)
        .rename({"Zmd000100": "Z"})
        .assign_coords({"Z": Z_values})
        .sel(T=slice(TISW, None, nt), Xp1=slice(0, LISW, nx), Z=slice(0, zmax))
    )

# create z-axis with dz = 2 m 
zISW  = np.arange(0,-20-dz,-dz)
# Initialize u 
u_i = np.empty((len(u), len(zISW), len(u.Xp1)))

# Loop over each time step to interpolate along z
for tt in range(len(u)):
    for xx in range(len(u.Xp1)):
        # Interpolate along the z axis for each (T, X) slice
        f = interp1d(u.Z.values, u.isel(T=tt, Xp1=xx).values, kind='linear', fill_value='extrapolate')
        u_i[tt, :, xx] = f(zISW)

# create a 4D matrix with shape (T,Z,Y,X)
# such as ISW bumps the wharf perpendicularly
# the wharf is parallel to longitude
u4d = np.tile(u_i[:, :, :, np.newaxis], (1, 1, 1, len(u.Xp1)))

### contruct grid in lat and lon
### we want to work in degree
### 1 degree = 60 '
### 1 ' of latitude = 1852 m
# Position of wharf center
### Jonathan gave me for longitudes
### 70 W 50.007
### 70 W 49.7743
### and latitudes
### 48 N 24.1074
### 48 N 24.1015
lon_wharf   = -70.8315
lat_wharf   =  48.40174
# resolution in lat and lon
dlat        = dx/(60*1852)
dlon        = dlat/np.cos(np.radians(lat_wharf))
# compute min and max of lon and lat
lon_min_ISW = lon_wharf - (LISW/(2*dx))*dlon
lon_max_ISW = lon_wharf + (LISW/(2*dx))*dlon
lat_min_ISW = lat_wharf 
lat_max_ISW = lat_wharf + (LISW/dx)*dlat 
# compute grid
lon_ISW     = np.linspace(lon_min_ISW, lon_max_ISW, len(u.Xp1))
lat_ISW     = np.linspace(lat_min_ISW, lat_max_ISW, len(u.Xp1))

### save the data in a netcdf file
### define output filename
fname = f"/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/K-Sim/netcdf/ISW4-{edir[:-1]}-dxdy{dx}mdz{dz}mdt{dt}s.nc" 
### Convert data to xarray Dataset and save to NetCDF
ds_out = xr.Dataset(
    {
        "v": (("time", "depth", "lat", "lon"), u4d),
    },
    coords={
        "lat": lat_ISW,
        "lon": lon_ISW,
        "depth": zISW,
        "time": np.arange(0, len(u4d) * dt, dt),
    },
    attrs={
        "description": f"Four internal waves propagating perpendicular to the wharf. "
                       f"The horizontal resolution is {dx} m over a grid of {LISW} m by {LISW} m. "
                       f"The vertical resolution is {dz} m. The timestep is {dt} s."
    },
)

# Save using xarray 
ds_out.to_netcdf(fname)

