#!/Users/sandy/miniconda3/bin/python
"""
Plot interpolated and bandpass-filtered temperature fields during a selected
internal wave event.

This script:
1. Extracts a user-defined time window from the interpolated temperature field.
2. Plots the interpolated temperature field (T_i) in the upper panel.
3. Plots the bandpass-filtered interpolated temperature field (T_i_but)
   in the lower panel.
4. Uses the moving vertical grid z_grid to account for free-surface variations.
5. Adds independent colorbars for the raw and filtered temperature fields.

The upper panel shows the large-scale thermal structure, while the lower
panel highlights short-period temperature fluctuations associated with
internal wave activity.
"""

# import modules
import numpy             as np
import pandas            as pd
import rsktools          as rsk
import matplotlib.pyplot as plt
import matplotlib.dates  as md
###################################
# Period of measurements
year   = '2023'
# define RSK directory 
dirRSK = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/Instruments/RBR/RBR_Solo-Duet/' + year + '/'
figdir = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/Instruments/RBR/Figs/'
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
    # selected period 
    T0 = np.array('2023-09-28T20:00', dtype='datetime64[ms]')
    T1 = np.array('2023-09-28T23:00', dtype='datetime64[ms]')
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
    # selected period 
    T0 = np.array('2024-08-05T09:00', dtype='datetime64[ms]')
    T1 = np.array('2024-08-05T12:00', dtype='datetime64[ms]')

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

# Convert to pandas timestamps
T0_pd = pd.to_datetime(str(T0))
T1_pd = pd.to_datetime(str(T1))

# Time mask
mask_event = (df_h.index >= T0_pd) & (df_h.index <= T1_pd)

# Extract time and data during event
time_event    = df_h.index[mask_event]
T_i_event     = T_i[:, mask_event]
T_i_but_event = T_i_but[:, mask_event]
z_event       = z_grid[:, mask_event]

X_event = np.tile(time_event.values, (T_i_event.shape[0], 1))

# Color levels
theta_ticks    = np.arange(6,18.2,0.2)
theta_ticks_cb = np.arange(6,20,2)

# Filtered temperature contours (bottom panel)
thetaf_ticks    = np.arange(-8, 8.2, 0.2)
thetaf_ticks_cb = np.arange(-8, 10, 2)

# Figure
plt.ion()
f, ax = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

# General title with date
date_str = T0_pd.strftime('%d-%m-%Y')
f.suptitle(date_str, fontsize=16)

# --- Top panel: interpolated temperature ---
c0 = ax[0].contourf(
    X_event, z_event, T_i_event,
    theta_ticks,
    cmap='RdYlBu_r',
    extend='both'
)

ax[0].contour(
    X_event, z_event, T_i_event,
    theta_ticks_cb,
    colors='k',
    linewidths=0.4,
    alpha=0.5
)

ax[0].set_ylabel(r"$z$ (m)")
ax[0].set_title("Interpolated temperature")
ax[0].set_facecolor("lightgrey")

# Colorbar top panel
cax0 = f.add_axes([0.91, 0.57, 0.015, 0.30])

f.colorbar(
    c0,
    cax=cax0,
    ticks=theta_ticks_cb,
    orientation='vertical',
    label=r'Temperature ($^\circ$C)'
)

# --- Bottom panel: filtered interpolated temperature ---
c1 = ax[1].contourf(
    X_event, z_event, T_i_but_event,
    thetaf_ticks,
    cmap='RdBu_r',
    extend='both'
)

ax[1].contour(
    X_event, z_event, T_i_but_event,
    thetaf_ticks_cb,
    colors='k',
    linewidths=0.4,
    alpha=0.5
)

ax[1].set_ylabel(r"$z$ (m)")
ax[1].set_xlabel(
    f"Time (UTC) on {T0_pd.strftime('%d %B %Y')}"
)
ax[1].set_title("Filtered interpolated temperature")
ax[1].set_facecolor("lightgrey")

# Colorbar bottom panel
cax1 = f.add_axes([0.91, 0.13, 0.015, 0.30])

f.colorbar(
    c1,
    cax=cax1,
    ticks=thetaf_ticks_cb,
    orientation='vertical',
    label=r"Filtered temperature ($^\circ$C)"
)

# =========================================================
# Time formatting
# =========================================================

ax[1].xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
ax[1].xaxis.set_minor_locator(md.MinuteLocator(interval=5))
plt.gcf().autofmt_xdate()

# General title
date_str = T0_pd.strftime('%d-%m-%Y')
f.suptitle(date_str, fontsize=18)

# Layout
f.subplots_adjust(right=0.88, hspace=0.30)

figname = (
    f"{figdir}T-interp-filtered-event-"
    f"{T0_pd.strftime('%Y%m%d_%H%M')}-"
    f"{T1_pd.strftime('%H%M')}.png"
)

f.savefig(figname, dpi=500, bbox_inches='tight')

