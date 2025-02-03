#!/Users/sandy/miniconda3/bin/python

# This script convert rsk to NetCDF format 
# for ISW occuring on the 20th of september 

# import modules
from pyrsktools           import RSK
import numpy              as np
import pandas             as pd
import xarray             as xr
import datetime

###################################
# define RSK directory 
dirRSK     = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/Instruments/RBR/RBR_Solo-Duet/2023/'
# define RSK files name 
fileRSK    = ['230463_20231101_1957.rsk', '230462_20231101_1952.rsk', \
             '230461_20231101_1947.rsk', '230460_20231101_1942.rsk', \
             '230459_20231101_1936.rsk', '230458_20231101_1931.rsk', \
             '230457_20231101_1925.rsk', '230456_20231101_1920.rsk', \
             '230455_20231101_1902.rsk', '230454_20231101_1912.rsk', \
             '230453_20231101_1907.rsk', '230452_20231101_1916.rsk', \
             '230451_20231101_1843.rsk', '230450_20231101_1837.rsk', \
             '230890_20231101_1849.rsk']
# Define RSK z-axis
zSolo      = np.array([19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 7, 5, 3, 1])
# beginning and end time of event
t0         = pd.to_datetime('2023-09-20 17:05:00')
tend       = pd.to_datetime('2023-09-20 17:35:00')

# initialize list to store data
data_list      = []
# Loop through each file and extract data
for i, filename in enumerate(fileRSK):
    path = dirRSK + filename
    with RSK(path) as rsk:
        # load the data
        rsk.readdata()

        # Get time and temperature 
        time_data = rsk.data['timestamp']
        temp_data = rsk.data['temperature']

        # Convert to DataFrame
        # Add pressure column if it exists
        if 'pressure' in rsk.data.dtype.names: 
            pressure_data = rsk.data['pressure']
            df = pd.DataFrame({
                'time': time_data,
                'temperature': temp_data,
                'pressure': pressure_data,
            })
        else:
            df = pd.DataFrame({
                'time': time_data,
                'temperature': temp_data,
            })

        # Filter by time
        df = df[(df['time'] >= t0) & (df['time'] <= tend)]
        
        # Add depth information
        df['depth'] = zSolo[i]
        
        # Append to list
        data_list.append(df)

# Combine all DataFrames
full_df = pd.concat(data_list)

# Compute elapsed seconds since the reference time
full_df['time'] = (pd.to_datetime(full_df['time']) - t0).dt.total_seconds()

# Create xarray dataset
ds = full_df.set_index(['time', 'depth']).to_xarray()

# Add time metadata
ds['time'].attrs['description'] = 'Timestamp in UTC'
ds['time'].attrs['units'] = f'seconds since {t0}'
ds['time'].attrs['calendar'] = 'proleptic_gregorian'

# Define encoding for time to ensure proper formatting in NetCDF
encoding = {'time': {
        'units': f'seconds since {t0}',
        'calendar': 'proleptic_gregorian'}
}

# Add metadata for variables
ds['depth'].attrs['units'] = 'm'
ds['depth'].attrs['description'] = 'Depth of the sensor'

ds['temperature'].attrs['units'] = '°C'
ds['temperature'].attrs['description'] = 'Temperature at each depth'

ds['pressure'].attrs['units'] = 'dbar'
ds['pressure'].attrs['description'] = 'Pressure at 1 meter above seafloor'

# Add global attributes with detailed description
ds.attrs['dataset_title'] = 'Temperature and Pressure Time Series on 2023-09-20'
ds.attrs['description'] = (
    "This dataset contains time series of temperature and pressure"
    "during an event where a wavetrain of internal solitary waves collided with the Grande-Anse Wharf "
    "in the Saguenay Fjord. The sensors were deployed at the western corner of the Grande-Anse Terminal wharf "
    "(48° 24.107′ N, 70° 50.001′ W), with the deepest instrument being the temperature-pressure sensor positioned "
    "at one meter above seafloor. The dataset includes pressure data only at the 1-meter depth."
)

# Save to NetCDF
output_file = 'rbr_20230920.nc'
ds.to_netcdf(dirRSK + output_file)


