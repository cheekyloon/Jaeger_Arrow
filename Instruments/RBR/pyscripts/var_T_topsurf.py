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
from matplotlib.ticker   import LogLocator

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
zSolo       = np.array([19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 7, 5, 3, 1])

# Depth for temperature averaging 
depth = 13
# order of filter 
N_but = 4

# Step 1: Load pressure data from the last RSK file '230890_20231101_1849.rsk'
df_pressure = rsk.load_rsk_data(dirRSK, fileRSK[-1], 'pressure', t0, tend, -9.13)
# apply a lowpass filter on water level
df_pressure['h'] = rsk.lp_filter(df_pressure['pressure'], df_pressure.index, 1/1800, N_but)
# extract h
h = df_pressure['h']
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
T_mean_but = rsk.bp_filter(T_mean, df_pressure.index, 1/T_low, 1/T_high, N_but)
T_i_but    = rsk.bp_filter(T_i, df_pressure.index, 1/T_low, 1/T_high, N_but)

# Step 6: Compute the average of the filtered temperature over 13 m 
T_i_but_mean = np.nanmean(T_i_but**2, axis=0)

# save data into .mat
from scipy.io    import savemat
# Convert pandas index to list of Python datetimes
time_python = df_pressure.index.to_pydatetime()
# Convert to MATLAB datenum
time_matlab = np.array([
    dt.toordinal() + 366 + (dt.hour*3600 + dt.minute*60 + dt.second + dt.microsecond/1e6) / 86400
    for dt in time_python
])
# Prepare the data to save
data_to_save = {
  'Tprime': T_i_but_mean,
  'time'  : time_matlab,    
}
# Save in a .mat file
dir_mat = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/Instruments/RBR/mat/'
savemat(dir_mat + 'Tprime_mean_2024.mat', data_to_save)

# Step 7: Compute phase averages 
# number of bins
N_bin = 50
# water level
phase_bin, h_bin     = rsk.M2_phase_avg(h, df_pressure.index, eta, N_bin)
# weighted-averaged temperature
phase_bin, T_bin_1   = rsk.M2_phase_avg(h, df_pressure.index, T_mean_but**2, N_bin)
phase_bin, T_bin_2   = rsk.M2_phase_avg(h, df_pressure.index, T_i_but_mean, N_bin)
phase_bin, all_T_bin = rsk.M2_phase_avg(h, df_pressure.index, T_i_but**2, N_bin)

# Step 8: Spectral analysis
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
# weighted-averaged temperature 
F_T, P_T   = signal.welch(T_mean, Fs, nperseg = perseg, noverlap=overlap, nfft=nfft, window = "hamming", detrend = False)
# filtered depth-averaged temperature (Method 1) 
F_T, P_Tf1 = signal.welch(T_mean_but, Fs, nperseg = perseg, noverlap=overlap, nfft=nfft, window = "hamming", detrend = False)
# depth-averaged of filtered temperature (Method 2) 
F_T, P_Tf2 = signal.welch(T_i_but_mean, Fs, nperseg = perseg, noverlap=overlap, nfft=nfft, window = "hamming", detrend = False)

#### Make the figure ####

# define figure characteristics
plt.rcParams['font.family']         = 'Helvetica'
plt.rcParams['axes.titlepad']       = 15
plt.rcParams['font.size']           = 17
plt.rcParams['xtick.labelsize']     = 17
plt.rcParams['ytick.labelsize']     = 17
plt.rcParams['axes.labelsize' ]     = 17
plt.rcParams['legend.fontsize']     = 15
plt.rcParams['figure.titlesize']    = 20
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True

# Create colormap from dark to light blue
cmap     = plt.cm.RdYlBu_r
n_depths = np.size(z_grid, 0)
colors   = [cmap(i / n_depths) for i in range(n_depths)]

# M2, M4, M6 and M8 period in hours
T_M2 = 12.4206012025189  
T_M4 = T_M2/2 
T_M6 = T_M2/3 
T_M8 = T_M2/4
# convert tidal phase in radians to time in hours
time_bin = ((phase_bin + np.pi/2) / (2 * np.pi)) * T_M2

# Vizualisation
plt.ion()

# Phase average
f1, ax1 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
# First subplot: water level 
ax1[0].plot(time_bin, h_bin, linestyle='-', color='b')
ax1[0].set_yticks(np.arange(-2,2.5,0.5))
ax1[0].set_ylabel(r"$\eta$ (m)")
ax1[0].set_title(
    rf"Year {int(year)}: $f_c = [1/{T_low}\,\mathrm{{s}},\,1/{T_high}\,\mathrm{{s}}]$"
)
# Second subplot: weighted-averaged temperature 
ax1[1].plot(time_bin, np.sqrt(T_bin_1), linestyle='-', color='b', label=r"$\sigma \left(\overline{T}'\right)$")
ax1[1].plot(time_bin, np.sqrt(T_bin_2), linestyle='-', color='g', label=r"$\sigma \left(\overline{T'}\right)$")
ax1[1].plot(time_bin, np.nanmean(np.sqrt(all_T_bin), axis=0), linestyle='-', color='orange', label=r"$\overline{\sigma \left(T'\right)}$") 
ax1[1].legend()
ax1[1].set_xticks(np.arange(-3,9.5,1))
ax1[1].set_yticks(np.arange(0.1,0.75,0.1))
ax1[1].set_ylabel(r"$[^\circ\mathrm{C}]$")
ax1[1].set_xlabel('Time relative to high water [h]')
# Save figure
figname1 =f"{figdir}phase-avg-BP{T_low}–{T_high}-y{int(year)}-{depth}m.png" 
f1.savefig(figname1,dpi=500,bbox_inches='tight')

# Phase average: method 2 
f2, ax2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
# First subplot: water level 
ax2[0].plot(time_bin, h_bin, linestyle='-', color='b')
ax2[0].set_yticks(np.arange(-2,2.5,0.5))
ax2[0].set_ylabel(r"$\eta$ (m)")
ax2[0].set_title(
    rf"Year {int(year)}: $f_c = [1/{T_low}\,\mathrm{{s}},\,1/{T_high}\,\mathrm{{s}}]$"
)
# Second subplot: weighted-averaged temperature 
ax2[1].plot(time_bin, np.sqrt(T_bin_2), linestyle='-', linewidth=1.5, color='g')
for zz in range(np.size(all_T_bin,0)):
    ax2[1].plot(time_bin, np.sqrt(all_T_bin[zz,:]), linestyle='--', linewidth=0.8, color=colors[zz])
ax2[1].set_xticks(np.arange(-3,9.5,1))
ax2[1].set_yticks(np.arange(0.,0.95,0.1))
ax2[1].set_ylabel(r"$\sigma \left(\overline{T'}\right) [^\circ\mathrm{C}]$")
ax2[1].set_xlabel('Time relative to high water [h]')
# Save figure
figname2 =f"{figdir}method2-phase-avg-BP{T_low}–{T_high}-y{year}-{depth}m.png" 
f2.savefig(figname2,dpi=500,bbox_inches='tight')

# Timeseries
f3, ax3 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
# First subplot: water level 
ax3[0].plot(df_pressure.index, h, linestyle='-', color='b')
ax3[0].set_yticks(np.arange(14,21,1))
# Second subplot: weighted-averaged temperature 
ax3[1].plot(df_pressure.index, T_mean - np.nanmean(T_mean), linestyle='-', color='k', label=r"$\overline{T} - mean(\overline{T})$")
ax3[1].plot(df_pressure.index, T_mean_but, linestyle='-', color='b', label=r"$\overline{T}'$")
ax3[1].legend(loc="lower right", fontsize=10)
ax3[1].set_yticks(np.arange(-8,9,2))
# Date formatting on the x-axis of the second subplot
ax3[1].xaxis.set_major_locator(md.DayLocator(interval=7))  # every 7 days
ax3[1].xaxis.set_major_formatter(md.DateFormatter('%d %b'))  # e.g., 15 Jul
# Tilt the labels to avoid overlap
f3.autofmt_xdate()
# Labels
ax3[0].set_ylabel(r"$h$ (m)")
ax3[0].set_title(
    rf"Year {int(year)}: $f_c = [1/{T_low}\,\mathrm{{s}},\,1/{T_high}\,\mathrm{{s}}]$"
)
ax3[1].set_xlabel(f"Date in {year} [month]")
ax3[1].set_ylabel(r"$[^\circ\mathrm{C}]$")
# Save figure
figname3 =f"{figdir}method1-TS-BP{T_low}–{T_high}-y{int(year)}-{depth}m.png" 
f3.savefig(figname3,dpi=500,bbox_inches='tight')

# PSD
f5, ax5 = plt.subplots(figsize=(12, 6))
# Second subplot: weighted-averaged temperature 
ax5.loglog(F_T, P_T, 'k', linewidth=0.8, label=r"$\overline{T}$")
ax5.loglog(F_T, P_Tf1, 'b', linewidth=0.8, label=r"$\overline{T}'$")
ax5.loglog(F_T, P_Tf2, 'g', linewidth=0.8, label=r"$\overline{(T')^2}$")
ax5.legend()
# add vertical axes
ax5.axvline(x=1/T_low, color='r', linestyle='--', linewidth=0.5, label=fr"$f_c = 1/{T_low}$")
ax5.axvline(x=1/T_high, color='m', linestyle='--', linewidth=0.5, label=fr"$f_c = 1/{T_high}$")
# add text
ax5.text(1/(T_M2*3600)-4e-6, 4e6, "M2", color='k', fontsize=11)
ax5.text(1/(T_M4*3600)-8e-6, 4e5, "M4", color='k', fontsize=11)
ax5.text(1/(T_M6*3600)-1.2e-5, 1.5e5, "M6", color='k', fontsize=11)
ax5.text(1/(T_M8*3600)-1.3e-5, 3e4, "M8", color='k', fontsize=11)
# add labels
ax5.set_xlabel("f [Hz]")
ax5.set_ylabel(r"PSD [Hz$^{-1}$]")
ax5.set_title(
    rf"Year {int(year)}: $f_c = [1/{T_low}\,\mathrm{{s}},\,1/{T_high}\,\mathrm{{s}}]$"
)
# set major y-tick labels 
ax5.set_yticks([1e-9, 1e-7,1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5, 1e7])
ax5.set_yticklabels([r"$10^{-9}$", r"$10^{-7}$", r"$10^{-5}$", r"$10^{-3}$", r"$10^{-1}$", r"$10^1$", r"$10^3$", r"$10^5$", r"$10^7$"])
# enable minor ticks 
ax5.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto'))
ax5.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
# Show ticks inward
ax5.tick_params(axis='both', which='both', direction='in', right=True, top=True)
ax5.tick_params(axis='y', which='minor', length=4, color='gray')
# set x- and y-limits
ax5.set_xlim(1e-6, 5e-1)
ax5.set_ylim(1e-10, 1e8)

# Save figure
figname5 =f"{figdir}PSD-BP{T_low}–{T_high}-y{int(year)}-{depth}m.png" 
f5.savefig(figname5,dpi=500,bbox_inches='tight')


