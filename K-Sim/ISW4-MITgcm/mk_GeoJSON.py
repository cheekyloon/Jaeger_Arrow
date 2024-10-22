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
### netcdf filename to convert in geojson
exp  = "ISW4-APE1e5-dxdy2mdz" + str(dz) + "mdt" + str(dt) + "s"
fnc  = rdir + "netcdf/" + exp + ".nc"

### open netcdf file and load variables
ds    = nc.Dataset(fnc, 'r')
t     = ds.variables['time'][:]
lon   = ds.variables['lon'][:]
lat   = ds.variables['lat'][:]
depth = ds.variables['depth'][:]

# Create the GeoJSON structure
geojson_data = {
    "type": "FeatureCollection",
    "name": exp,
    "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
    "features": []
}

# Initialize loop over coordinates
fid = 1
# Loop over the xarray dataset to populate the features
for ii in range(len(lon)):
    for jj in range(len(lat)):

        # Loop over depth
        for zz in range(len(depth)):
            if zz == 0:
               # Initialize collecting current values for the surface depth
               current_str = f"{depth[zz]} ["
            else:
               # Collecting current values for this depth
               # MITgcm has negative depth while K-Sim takes positive depth
               current_str = current_str + "|" + f"{-depth[zz]}" + "[" 
            
            # Initialize a list to collect time, speed, direction for each time step 
            time_speed_dir_list = []

            # Loop over time
            for tt in range(len(t)):
                # Access the currents in m/s
                U  = 0 
                V  = ds['v'][tt, zz, jj, ii].data
                # find direction of currents in radians
                DIR = np.pi/2 - math.atan2(V,U)  
                # Add to the time-speed-direction list with the time in s 
                time_speed_dir_list.append(f"{t[tt]},{abs(np.round(V,2))},{DIR}")

            # Join time-speed-direction entries into a single string for this depth
            current_str += ','.join(time_speed_dir_list) + "]" 

        feature = {
            "type": "Feature",
            "properties": {
             "fid": int(fid),
             "Period": int(t[-1]+15),
             "Current": current_str
            },
            "geometry": {
             "type": "Point",
             "coordinates": [lon[ii], lat[jj]]
            }
        }
        geojson_data["features"].append(feature)
        fid += 1

### close netcdf file
ds.close()

# Write the geojson data to a file
fgeo = rdir + GeoJSON + exp + ".GeoJSON"
with open(fgeo, "w") as geojson_file:
    json.dump(geojson_data, geojson_file, indent=4)

