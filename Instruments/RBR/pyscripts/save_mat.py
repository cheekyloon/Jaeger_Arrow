#!/Users/sandy/miniconda3/bin/python
"""
Process RBR temperature records to identify and separate neap- and
spring-tide periods.

This script:
1. Loads pressure and temperature measurements from multiple RBR Solo/Duet
   instruments deployed during the 2023 or 2024 field campaign.
2. Applies a low-pass filter to the pressure record to estimate the
   free-surface elevation.
3. Masks temperature measurements located above the moving free surface.
4. Interpolates the temperature field onto a regular vertical grid and
   computes vertically averaged temperatures over the upper 13 m.
5. Applies a bandpass filter to isolate short-period temperature
   fluctuations associated with internal wave activity.
6. Estimates the local tidal amplitude using the Hilbert transform of the
   pressure signal.
7. Separates the record into neap- and spring-tide periods based on the
   smoothed tidal amplitude.
8. Extracts filtered temperature fields, vertical grids, pressure records,
   and timestamps corresponding to neap and spring tides.
9. Saves all processed variables into a Matlab .mat file for subsequent
   analysis and figure generation.

The resulting dataset contains:
- the full filtered temperature field,
- the corresponding moving vertical grid,
- pressure records,
- Matlab-formatted timestamps,
- and subsets corresponding to neap and spring tides.
"""

# import modules
import numpy     as np
import pandas    as pd
import rsktools  as rsk
from scipy.io    import savemat
from scipy.signal        import hilbert

###################################
# Period of measurements
year   = '2023'
# define RSK directory 
dirRSK = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/Instruments/RBR/RBR_Solo-Duet/' + year + '/'
# define RSK files name 
if year == '2023':
    fileRSK = [
        '230463_20231101_1957.rsk', '230462_20231101_1952.rsk',
        '230461_20231101_1947.rsk', '230460_20231101_1942.rsk',
        '230459_20231101_1936.rsk', '230458_20231101_1931.rsk',
        '230457_20231101_1925.rsk', '230456_20231101_1920.rsk',
        '230455_20231101_1902.rsk', '230454_20231101_1912.rsk',
        '230453_20231101_1907.rsk', '230452_20231101_1916.rsk',
        '230451_20231101_1843.rsk', '230450_20231101_1837.rsk',
        '230890_20231101_1849.rsk'
    ]
    # beginning and end time of event
    t0      = pd.to_datetime('2023-09-15 15:30:00')
    tend    = pd.to_datetime('2023-11-01 16:30:00')
else:
    fileRSK = [
        '230463_20240911_1135.rsk', '230462_20240911_1114.rsk',
        '230461_20240911_1200.rsk', '230460_20240911_1028.rsk',
        '230459_20240911_1127.rsk', '230458_20240911_1148.rsk',
        '230457_20240911_1144.rsk', '230456_20240911_1153.rsk',
        '230455_20240911_1131.rsk', '230454_20240911_1139.rsk',
        '230453_20240911_1032.rsk', '230452_20240911_1204.rsk',
        '230451_20240911_1118.rsk', '230450_20240911_1037.rsk',
        '230890_20240911_1041.rsk'
    ]
    # beginning and end time of event
    t0      = pd.to_datetime('2024-07-24 00:00:00')
    tend    = pd.to_datetime('2024-09-11 12:00:00')

# Define RSK z-axis
zSolo      = np.array([19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 7, 5, 3, 1])
# Depth for temperature averaging 
depth = 13
# order of filter 
N_but = 4

# Step 1: Load depth data from the last RSK file '230890_20231101_1849.rsk'
df_h = rsk.load_rsk_data(dirRSK, fileRSK[-1], 'depth', t0, tend) + 1
# apply a lowpass filter on water level
df_h['h'] = rsk.lp_filter(df_h['depth'], df_h.index, 1/1800, N_but)
# extract h
h = df_h['h']
# computer eta
eta   = h.values - h.mean()

# Step 2: Process all temperature files and mask temperature and depths above the free surface 
all_T, all_z = rsk.mask_above_surf(dirRSK, fileRSK, zSolo, h, t0, tend)

# Step 4: Compute the average temperature over the first 13 m when interpolated onto 
# a regular vertical grid with dz = 0.5 m 
T_mean, T_i, z_grid = rsk.interp_avg_top(all_T, all_z, h, depth=depth)

# Step 5: Apply bandpass filter 
# cutoff frequency for the bandpass
T_low   = 360
T_high  = 30
# apply filter
T_mean_but = rsk.bp_filter(T_mean, df_h.index, 1/T_low, 1/T_high, N_but)
T_i_but    = rsk.bp_filter(T_i, df_h.index, 1/T_low, 1/T_high, N_but)

# Step 3: convert the time to Maltlab format
# Convert pandas index to list of Python datetimes
time_python = df_h.index.to_pydatetime()

# Convert to MATLAB datenum
time_matlab = np.array([
    dt.toordinal() + 366 + (dt.hour*3600 + dt.minute*60 + dt.second + dt.microsecond/1e6) / 86400
    for dt in time_python
])

# Step 4: neap vs spring 
# Compute the local tidal amplitude using the Hilbert transform
# The mean water level is removed before calculating the analytic signal
amp = np.abs(hilbert(h.values - np.nanmean(h.values)))

# Smooth the amplitude over ~1 day
dt = (df_h.index[1] - df_h.index[0]).total_seconds()
win = int((24 * 3600) / dt)

amp_smooth = (
    pd.Series(amp, index=df_h.index)
    .rolling(win, center=True, min_periods=1)
    .mean()
    .values
)

# Split the time series into two equal groups based on tidal amplitude
# Low-amplitude periods correspond to neap tides
# High-amplitude periods correspond to spring tides
amp_med = np.nanmedian(amp_smooth)

# Create masks separating low- and high-amplitude tidal periods
# Low tidal amplitudes correspond to neap tides
# High tidal amplitudes correspond to spring tides
mask_neap   = amp_smooth <= amp_med
mask_spring = amp_smooth >  amp_med

# Extract filtered temperature during neap and spring tides
T_neap = T_i_but[:, mask_neap]
T_spring = T_i_but[:, mask_spring]

# Extract water level time series for neap and spring tides
h_neap = h[mask_neap]
h_spring = h[mask_spring]

# Extract corresponding vertical grid
z_neap = z_grid[:, mask_neap]
z_spring = z_grid[:, mask_spring]

# Extract corresponding timestamps
time_neap = time_matlab[mask_neap]
time_spring = time_matlab[mask_spring]

# Step 5: Prepare the data to save
data_to_save = {
  'all_T': T_i_but,                  # temperature matrix
  'all_z': z_grid,                   # depth matrix
  'time': time_matlab,               # time in Matlab format 
  'pressure': h.values,              # pressure values
  'T_neap': T_neap,                  # temperature for neap tides
  'z_neap': z_neap,                  # depth for neap tides 
  'time_neap': time_neap,            # time in Matlab format 
  'pressure_neap': h_neap.values,    # pressure values
  'T_spring': T_spring,              # temperature for spring tides
  'z_spring': z_spring,              # depth for spring tides
  'time_spring': time_spring,        # time in Matlab format 
  'pressure_spring': h_spring.values # pressure values
}

# Step 5: save in a .mat file
dir_mat = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/Instruments/RBR/mat/'
savemat(dir_mat + 'T_i_13m_' + year + '.mat', data_to_save)
