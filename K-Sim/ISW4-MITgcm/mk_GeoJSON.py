#!/usr/bin/env python 

### create GeoJSON file from a netcdf file 

### import modules
import numpy     as np
import netCDF4   as nc
import json
import math 

## vertical resolution
dz = 2
## timestep
dt = 15

## root directory
rdir = "/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/K-Sim/"
## netcdf filename to convert in geojson
exp  = f"ISW4-APE1e5-dxdy2mdz{dz}mdt{dt}s"
fnc  = f"{rdir}netcdf/{exp}.nc"

## open netcdf file and load variables
with nc.Dataset(fnc, 'r') as ds:
    t = ds.variables['time'][:]
    lon = ds.variables['lon'][:]
    lat = ds.variables['lat'][:]
    depth = ds.variables['depth'][:]

    ## Create the GeoJSON structure
    geojson_data = {
        "type": "FeatureCollection",
        "name": exp,
        "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
        "features": []
    }

    ## Initialize ID counter 
    fid = 1

    ## Loop over spatial coordinates
    for ii, lon_val in enumerate(lon):
        for jj, lat_val in enumerate(lat):
            # Initialize a list to collect time, speed, direction for all time step for each depth 
            current_str = []

            for zz, depth_val in enumerate(depth):
                # Depth
                # MITgcm uses negative depth, K-Sim requires positive depth
                depth_label = f"{depth[zz]}" if zz == 0 else f"{-depth[zz]}"
                # Initialize a list to collect time, speed, direction for each time step 
                time_speed_dir_list = []

                ## Loop over time
                for tt, time_val in enumerate(t):
                    # U component is zero
                    U = 0 
                    # V component from dataset 
                    V = ds['v'][tt, zz, jj, ii].data 
                    # Current direction in radians 
                    DIR = np.pi/2 - math.atan2(V, U)
                    # Add to the time-speed-direction list with the time in s 
                    time_speed_dir_list.append(f"{int(time_val)},{abs(np.round(V, 2))},{DIR}")

                # Join time-speed-direction entries into a single string for this depth
                current_str.append(f"{depth_label}[" + ",".join(time_speed_dir_list) + "]")

            feature = {
                "type": "Feature",
                "properties": {
                    "fid": fid,
                    "Period": int(t[-1] + dt),
                    "Current": "|".join(current_str),
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon_val, lat_val],
                },
            }

            geojson_data["features"].append(feature)
            fid += 1

## Function to convert NumPy data types to standard Python types
def convert_numpy_types(obj):
    if isinstance(obj, (np.ndarray, list)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    return obj

## Convert data and save GeoJSON file
geojson_data_clean = convert_numpy_types(geojson_data)

# Write the geojson data to a file
fgeo = f"{rdir}/GeoJSON/{exp}.GeoJSON"
with open(fgeo, "w") as geojson_file:
    json.dump(geojson_data_clean, geojson_file)


