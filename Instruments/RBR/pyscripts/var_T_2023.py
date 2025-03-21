#!/Users/sandy/miniconda3/bin/python

# This script load temperature field from rsk thermometers
# Temperatures are masked above sea level using pressure data
# The variance of temperature is computed along the depth using
# the depth-weighted temperature

# import modules
from pyrsktools     import RSK
import numpy        as np
import pandas       as pd
import xarray       as xr
import rsktools     as rsk
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
t0         = pd.to_datetime('2023-09-15 15:30:00')
tend       = pd.to_datetime('2023-11-01 16:30:00')


###################################
# initialize DataFrame
df_rsk      = []

# Step 1: Load pressure data from the last RSK file '230890_20231101_1849.rsk'
df_pressure = rsk.load_rsk_data(dirRSK, fileRSK[-1], 'pressure', t0, tend, -9.13)

# Step 2: Process all temperature files and mask temperature and depths above the free surface 
all_temp, all_depths = rsk.mask_above_surf(dirRSK, fileRSK, zSolo, pressure_data, t0, tend)

# Loop over time to adjust the depth at the surface with pressure_data
for tt in range(all_depths.shape[1]):
    # Find valid depths 
    valid_ind = np.where(~np.isnan(all_depths[:, tt]))[0]
    # If at least one valid depth exists 
    if valid_ind.size > 0: 
        # Index of first valid temperature 
        first_valid_idx = valid_ind[0]
        # Set to water level
        all_depths[first_valid_idx, tt] = pressure_data[tt]

# Step 3: Compute dz for the depth-weighted temperature
dz = rsk.compute_dz(all_depths)

# Step 4: Compute variance for the temperature along the depth
variance_depth = rsk.compute_variance_depth(all_temp=[], dz=dz)


