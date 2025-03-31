#!/Users/sandy/miniconda3/bin/python

# This script load temperature field from rsk thermometers
# Temperatures are masked above sea level using pressure data
# The std of temperature is computed along the depth using
# the depth-weighted temperature
# The phase-averaged sea level and depth-weighted temperature std
# are also computed and plotted

# import modules
import numpy             as np
import pandas            as pd
import rsktools          as rsk
import matplotlib.pyplot as plt

###################################
# define RSK directory 
dirRSK     = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/Instruments/RBR/RBR_Solo-Duet/2024/'
# define RSK files name 
fileRSK    = ['230463_20240911_1135.rsk', '230462_20240911_1114.rsk', \
             '230461_20240911_1200.rsk', '230460_20240911_1028.rsk', \
             '230459_20240911_1127.rsk', '230458_20240911_1148.rsk', \
             '230457_20240911_1144.rsk', '230456_20240911_1153.rsk', \
             '230455_20240911_1131.rsk', '230454_20240911_1139.rsk', \
             '230453_20240911_1032.rsk', '230452_20240911_1204.rsk', \
             '230451_20240911_1118.rsk', '230450_20240911_1037.rsk', \
             '230890_20240911_1041.rsk']
# Define RSK z-axis
zSolo      = np.array([19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 7, 5, 3, 1])
# beginning and end time of event
t0         = pd.to_datetime('2024-07-24 00:00:00')
tend       = pd.to_datetime('2024-09-11 12:00:00')

# Step 1: Load pressure data from the last RSK file '230890_20231101_1849.rsk'
df_pressure   = rsk.load_rsk_data(dirRSK, fileRSK[-1], 'pressure', t0, tend, -9.13)
pressure_data = df_pressure['pressure']

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
        all_depths[first_valid_idx, tt] = pressure_data.iloc[tt] 

# Step 3: Compute dz for the depth-weighted temperature
dz = rsk.compute_dz(all_depths)

# Step 4: Compute the std for the temperature along the depth
weighted_temp, std_temp = rsk.compute_std_temp(all_temp, dz=dz)

# Step 5: Compute phase-averaged sea level and depth-weighted temperature std
#h_phase, std_temp_phase = rsk.compute_phase_averaged(df_pressure, std_temp)
# The desired number of bins. Could be anything.
N_bin = 50
h = df_pressure['pressure'].values - df_pressure['pressure'].mean()
phase_bin, T_bin = rsk.M2_phase_avg(df_pressure, weighted_temp, N_bin);
phase_bin, std_T_bin = rsk.M2_phase_avg(df_pressure, std_temp, N_bin);
phase_bin, h_bin = rsk.M2_phase_avg(df_pressure, h, N_bin);


#### Make the figure ####

# Define phase bins
#iphi = np.arange(-np.pi, np.pi + np.pi/25, np.pi/25)

# Vizualisation
plt.ion()

# Create the figure and subplots
fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

# First subplot: Interpolated h_anomaly vs iphi
axes[0].plot(phase_bin, h_bin, linestyle='-', color='g')
axes[0].set_ylabel(r"$\eta$ (m)")

axes[1].plot(phase_bin, T_bin, linestyle='-', color='b')
axes[1].set_ylabel('Depth-Weighted Temperature')

# Second subplot: std_temp_phase vs iphi
axes[2].plot(phase_bin, std_T_bin, linestyle='-', color='b')
axes[2].set_xlabel('Tidal Phase')
axes[2].set_ylabel('Depth-Weighted Temperature STD')

# Show plot
plt.tight_layout()

# Define number of bins
num_bins = 12

# Compute bin edges from -π to π
bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)

# Compute bin centers for plotting
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Digitize iphi to find which bin each point belongs to
bin_indices = np.digitize(iphi, bin_edges) - 1  # Subtract 1 to make indices zero-based

# Compute mean std_temp_mean for each bin
hist_values = np.array([np.nanmean(std_temp_phase[bin_indices == i]) for i in range(num_bins)])

# Plot histogram as a bar chart
plt.figure(figsize=(8, 5))
plt.bar(bin_centers, hist_values, width=(2 * np.pi / num_bins), align='center', color='b', alpha=0.7, edgecolor='k')

# Labels and formatting
plt.xlabel('iphi (Tidal Phase)')
plt.ylabel('Mean std_temp_mean')
plt.title('Histogram of std_temp_mean (12 Bins)')
plt.grid(True)
# Format x-axis labels
plt.xticks(bin_centers, labels=[f"{x:.2f}" for x in bin_centers])

# Save in .dat
df = pd.DataFrame({
    "P": df_pressure['pressure'].values,
    "weighted_T": weighted_temp,
    "std_T": std_temp
}, index=df_pressure.index)
df.to_csv("RBR_2024.dat", sep="\t", index=True, header=True, index_label="Timestamp")
