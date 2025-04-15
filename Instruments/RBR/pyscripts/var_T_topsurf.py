#!/Users/sandy/miniconda3/bin/python

# This script load temperature field from rsk thermometers
# Temperatures are masked above sea level using pressure data
# The depth-weighted temperature is computed and filtered (Method 1)
# The temperature is filtered and depth-averaged (Method 2) 
# The phase-averaged sea level and filtered depth-weighted temperature 
# are also computed
# Figures display the different steps 

# import modules
import numpy             as np
import pandas            as pd
import rsktools          as rsk
import matplotlib.pyplot as plt
import matplotlib.dates  as md
from scipy               import signal

###################################
# Period of measurements
year   = '2024'
# define directory to save figure
figdir = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/Instruments/RBR/Figs/'
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
    t0      = pd.to_datetime('2024-07-24 00:00:00')
    tend    = pd.to_datetime('2024-09-11 12:00:00')

# Define RSK z-axis
zSolo      = np.array([19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 7, 5, 3, 1])

# Step 1: Load pressure data from the last RSK file '230890_20231101_1849.rsk'
df_pressure   = rsk.load_rsk_data(dirRSK, fileRSK[-1], 'pressure', t0, tend, -9.13)
pressure_data = df_pressure['pressure']

# Step 2: Process all temperature files and mask temperature and depths above the free surface 
all_T, all_z = rsk.mask_above_surf(dirRSK, fileRSK, zSolo, pressure_data, t0, tend)

# Step 4: Compute the average temperature over the first 10 m when interpolated onto 
# a regular vertical grid with dz = 0.5 m 
T_mean, _, _  = rsk.interp_avg_top(all_T, all_z, pressure_data)

# Step 5: Apply bandpass filter 
# cutoff frequency for the bandpass
T_low     = 900
T_high    = 70
Fc_low    = 1/T_low
Fc_high   = 1/T_high
# order of filter 
N_but     = 4
# apply filter
T_but     = rsk.bp_filter(T_mean, df_pressure.index, Fc_low, Fc_high, N_but)
all_T_but = rsk.bp_filter(all_T, df_pressure.index, Fc_low, Fc_high, N_but)

# Step 6: Compute the weighted average for the filtered temperature along the depth
T_but_mean, _, _ = rsk.interp_avg_top(np.abs(all_T_but), all_z, pressure_data)

# Step 6: Compute phase averages 
# computer SSH
h = df_pressure['pressure'].values - df_pressure['pressure'].mean()
# number of bins
N_bin = 25
# water level
phase_bin, h_bin = rsk.M2_phase_avg(df_pressure, h, N_bin)
# weighted-averaged temperature
phase_bin, T_bin        = rsk.M2_phase_avg(df_pressure, np.abs(T_but), N_bin)
phase_bin, T_mean_bin   = rsk.M2_phase_avg(df_pressure, T_but_mean, N_bin)
phase_bin, all_T_bin    = rsk.M2_phase_avg(df_pressure, np.abs(all_T_but), N_bin)

# Step 6: Spectral analysis
# computer  temperature anomalies
T = T_mean - np.nanmean(T_mean)
# Calculate sampling interval and frequency
dt = (df_pressure.index[1] - df_pressure.index[0]).total_seconds()
Fs = 1 / dt
# === Mimic MATLAB default ===
# segment length
perseg  = int(np.floor(len(h) / 4.5))
# 50% overlap
overlap = int(np.floor(perseg / 2))
# next power of 2 of nperseg
nfft     = max(256, 2**int(np.ceil(np.log2(perseg))))
# === End mimic MATLAB default ===
# water level
F_h, P_h = signal.welch(h, Fs, nperseg = perseg, noverlap=overlap, nfft=nfft, window = "hamming", detrend = False)
# weighted-averaged temperature 
F_T, P_T = signal.welch(T, Fs, nperseg = perseg, noverlap=overlap, nfft=nfft, window = "hamming", detrend = False)
# filtered weighted-averaged temperature 
F_Tf, P_Tf = signal.welch(T_but, Fs, nperseg = perseg, noverlap=overlap, nfft=nfft, window = "hamming", detrend = False)


#### Make the figure ####

# Create colormap from dark to light blue
cmap     = plt.cm.Blues
n_depths = np.size(all_z, 0)
colors   = [cmap(i / n_depths) for i in range(n_depths)]

# Vizualisation
plt.ion()

# Create the figure and subplots
# Phase average
f1, ax1 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# First subplot: water level 
ax1[0].plot(phase_bin, h_bin, linestyle='-', color='g')
ax1[0].set_ylabel(r"$\eta$ (m)")
ax1[0].set_title(f"Year: {int(year)}")

# Second subplot: weighted-averaged temperature 
ax1[1].plot(phase_bin, T_bin, linestyle='-', color='r', label='Method 1')
ax1[1].plot(phase_bin, T_mean_bin, linestyle='-', color='b', label='Method 2')
ax1[1].legend()
ax1[1].set_ylabel('Depth-Averaged Temperature over 10 m')
ax1[1].set_xlabel('Tidal Phase')
ax1[1].set_title(f"Bandpass Filtered Periods: {int(T_low)}s – {int(T_high)}s")

# Save figure
figname1 =f"{figdir}phase-avg-BP{int(T_low)}–{int(T_high)}-y{int(year)}-10m.png" 
f1.savefig(figname1,dpi=500,bbox_inches='tight')

# Method 2 only
f2, ax2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# First subplot: water level 
ax2[0].plot(phase_bin, h_bin, linestyle='-', color='g')
ax2[0].set_ylabel(r"$\eta$ (m)")
ax2[0].set_title(f"Year: {int(year)}")

# Second subplot: weighted-averaged temperature 
ax2[1].plot(phase_bin, T_mean_bin, linestyle='-', linewidth=1.5, color='b')
for zz in range(np.size(all_T_bin,0)):
    ax2[1].plot(phase_bin, all_T_bin[zz,:], linestyle='--', linewidth=0.8, color=colors[zz])
ax2[1].set_ylabel('Depth-Averaged Temperature over 10 m')
ax2[1].set_xlabel('Tidal Phase')
ax2[1].set_title(f"Method 2: Bandpass Filtered Periods: {int(T_low)}s – {int(T_high)}s")

# Save figure
figname2 =f"{figdir}method2-phase-avg-BP{int(T_low)}–{int(T_high)}-y{int(year)}-10m.png" 
f2.savefig(figname2,dpi=500,bbox_inches='tight')

# Timeseries
# Method 1
f3, ax3 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# First subplot: water level 
ax3[0].plot(df_pressure.index, h, linestyle='-', color='g')
ax3[0].set_ylabel(r"$\eta$ (m)")
ax3[0].set_title(f"Year: {int(year)}")

# Second subplot: weighted-averaged temperature 
ax3[1].plot(df_pressure.index, T, linestyle='-', color='k', label='Full signal')
ax3[1].plot(df_pressure.index, T_but, linestyle='-', color='r', label='Filtered signal')
ax3[1].legend()

# Date formatting on the x-axis of the second subplot
ax3[1].xaxis.set_major_locator(md.MonthLocator())
ax3[1].xaxis.set_major_formatter(md.DateFormatter('%b'))
# Tilt the labels to avoid overlap
f3.autofmt_xdate()

# Labels
ax3[1].set_xlabel("Date in 2023 [month]")
ax3[1].set_ylabel('Depth-Averaged Temperature over 10 m')
ax3[1].set_title(f"Method 1: Bandpass Filtered Periods: {int(T_low)}s – {int(T_high)}s")

# Save figure
figname3 =f"{figdir}method1-TS-BP{int(T_low)}–{int(T_high)}-y{int(year)}-10m.png" 
f3.savefig(figname3,dpi=500,bbox_inches='tight')

# Method 2
f4, ax4 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# First subplot: water level 
ax4[0].plot(df_pressure.index, h, linestyle='-', color='g')
ax4[0].set_ylabel(r"$\eta$ (m)")
ax4[0].set_title(f"Year: {int(year)}")

# Second subplot: weighted-averaged temperature 
ax4[1].plot(df_pressure.index, T_but_mean, linestyle='-', color='r')
#for zz in range(np.size(all_T_but,0)):
#     ax4[1].plot(df_pressure.index, all_T_but[zz,:], linestyle='--', linewidth=0.8, color=colors[zz])

# Date formatting on the x-axis of the second subplot
ax4[1].xaxis.set_major_locator(md.MonthLocator())
ax4[1].xaxis.set_major_formatter(md.DateFormatter('%b'))
# Tilt the labels to avoid overlap
f4.autofmt_xdate()

# Labels
ax4[1].set_xlabel("Date in 2023 [month]")
ax4[1].set_ylabel('Depth-Averaged Temperature over 10 m')
ax4[1].set_title(f"Method 2: Bandpass Filtered Periods: {int(T_low)}s – {int(T_high)}s")

# Save figure
figname4 =f"{figdir}method2-TS-BP{int(T_low)}–{int(T_high)}-y{int(year)}-10m.png" 
f4.savefig(figname4,dpi=500,bbox_inches='tight')

# PSD
f5, ax5 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
# First subplot: water level 
ax5[0].loglog(F_h,P_h,'g')
ax5[0].set_ylabel(r"PSD for $\eta$")
ax5[0].set_title(f"Year: {int(year)}")
# Second subplot: weighted-averaged temperature 
ax5[1].loglog(F_T, P_T, 'b', label='Full signal')
ax5[1].loglog(F_Tf, P_Tf, 'r', label='Filtered signal')
ax5[1].legend()
# add labels
ax5[1].set_xlabel("f [Hz]")
ax5[1].set_ylabel(r"PSD for temperature")
ax5[1].set_title(f"Method 1 ({int(year)}): Bandpass Filtered Periods: {int(T_low)}s – {int(T_high)}s")
#plt.ylabel(r"PSD [m$^2$ Hz$^{-1}$]")

# Save figure
figname5 =f"{figdir}method1-PSD-BP{int(T_low)}–{int(T_high)}-y{int(year)}-10m.png" 
f5.savefig(figname5,dpi=500,bbox_inches='tight')


