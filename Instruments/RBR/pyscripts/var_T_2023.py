#!/Users/sandy/miniconda3/bin/python


# import modules
from pyrsktools     import RSK
import numpy        as np
import pandas       as pd
import xarray       as xr
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

# ---- STEP 1: Load pressure data from last file ----
# Initialize and read RSK data for last file
# '230890_20231101_1849.rsk'
rsk_last = RSK(dirRSK + fileRSK[-1])
rsk_last.open()
rsk_last.readdata()
# Extract time
time_pressure = pd.to_datetime(rsk_last.data['timestamp'])
# Extract pressure
# remove atmopsheric pressure (10.13)
# add one as the first thermometer is at 1 m above the ground 
pressure_data = rsk_last.data['pressure'] - 9.13
# Filter time range
mask_pressure = (time_pressure >= t0) & (time_pressure <= tend)
time_pressure = time_pressure[mask_pressure]
pressure_data = pressure_data[mask_pressure]
# Create DataFrame for easy time-based lookup
df_pressure = pd.DataFrame({'time': time_pressure, 'pressure': pressure_data})
df_pressure.set_index('time', inplace=True)
# ---- END STEP 1 ----

# ---- STEP 2: Process all temperature files ----
all_temp   = []
all_depths = []
for nn, file in enumerate(fileRSK):
    # Initialize and read RSK data 
    rsk  = RSK(dirRSK + file)
    rsk.open()
    rsk.readdata()
    # Extract time and temperature 
    time_data = pd.to_datetime(rsk.data['timestamp']) 
    temp_data = rsk.data['temperature']
    # Filter time range
    mask_t    = (time_data >= t0) & (time_data <= tend)
    time_data = time_data[mask_t]
    temp_data = temp_data[mask_t]
    # Assign depth
    depth = zSolo[nn] * np.ones_like(temp_data)
    # Mask temperature and depth where depth > water level
    mask_z            = depth > pressure_data
    temp_data[mask_z] = np.nan
    depth[mask_z]     = 0 
    # Store data for later dz computation
    all_temp.append(temp_data)
    all_depths.append(depth)

# Convert lists to 2D arrays
# Shape: (depth, time)
all_temp   = np.array(all_temp) 
all_depths = np.array(all_depths)      

# Loop over time to adjust depth_mask with pressure_data
for tt in range(all_temp.shape[1]):
    # Find valid depths 
    valid_ind = np.where(~np.isnan(all_temp[:, tt]))[0]
    # If at least one valid depth exists 
    if valid_ind.size > 0: 
        # Index of first valid temperature 
        first_valid_idx = valid_ind[0]
        # Set to water level
        all_depths[first_valid_idx, tt] = pressure_data[tt]
# ---- END STEP 2 ----

# ---- STEP 3: Compute dz ----
# Initialize with NaNs
dz = np.zeros_like(all_depths) * np.nan

# Loop over time
for tt in range(len(time_data)):
    # Depths at time tt 
    depths_t = all_depths[:, tt]
    # Get valid depth indices
    valid_ind = np.where(depths_t > 0)[0] 

    if valid_ind.size > 2:
        # Compute centered differences
        dz[valid_ind[1:-1], tt] = (depths_t[valid_ind[2:]] - depths_t[valid_ind[:-2]]) / 2
        # Forward difference for the first valid depth
        dz[valid_ind[0], tt] = depths_t[valid_ind[0]] - (np.floor(depths_t[valid_ind[0]]) + depths_t[valid_ind[1]]) / 2
        # Backward difference for the last valid depth
        dz[valid_ind[-1], tt] = 2 
    elif valid_ind.size == 2:
        # Forward difference for the first valid depth
        dz[valid_ind[0], tt] = depths_t[valid_ind[0]] - 2 
        # Backward difference for the last valid depth
        dz[valid_ind[-1], tt] = 2 
    elif valid_ind.size == 1:
        # Assign a default value (e.g., the depth itself)
        dz[valid_ind[0], tt] = depths_t[valid_ind[0]]  
# ---- END STEP 3 ----

# Initialize weighted temperature array
temp_weighted_avg = np.zeros(all_temp.shape[1]) * np.nan

# Loop over time
for tt in range(all_temp.shape[1]):
    # Get valid depth indices
    valid_ind = np.where(~np.isnan(all_temp[:, tt]))[0]

    if valid_ind.size > 0:
        # Extract valid temperatures and corresponding dz values
        temp_valid = all_temp[valid_ind, tt]
        dz_valid   = dz[valid_ind, tt]
        # Compute depth-weighted average temperature
        temp_weighted_avg[tt] = np.nansum(temp_valid * dz_valid) / np.nansum(dz_valid)



