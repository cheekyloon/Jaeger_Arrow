#!/Users/sandy/miniconda3/bin/python

# This script plot hovmoller
# of temperature from Solo thermometers
# for the 27th of september with
# ISW occurence

# import modules
from rsk_tools            import winmean_rsk_data
import numpy              as np
import pandas             as pd
import xarray             as xr
import scipy.io
import matplotlib.dates   as md
import matplotlib.pyplot  as plt
import datetime

###################################
# define figure name
figname   = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/Instruments/RBR/Figs/hov-pix-T_20230927.png'
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
# define colormap
cmapT     = 'RdYlBu_r'
## define contour and colorbar ticks for temperature
Tticks    = np.arange(6,18.2,0.2)
Tticks_cb = np.arange(6,20,2)

###################################
# load file for pixel intensity
# specify path
dirPI  = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/Instruments/CamDo/'
# specify file
filePI = 'Hovmoller_27_Sept_2023_1m.mat'
# load mat file
matPI  = scipy.io.loadmat(dirPI + filePI)
# Convert to datetime using the MATLAB epoch (January 1, 0000) as the reference
timePI = pd.to_datetime(matPI['mtime'][0] - 719529, unit='D', origin='unix')
# remove mean from pixel intensity
pixel_int = matPI['pixel_int'] - np.nanmean(matPI['pixel_int'])
# Create meshgrid
X, Y = np.meshgrid(timePI, matPI['x_prime'][0])

###################################
# define RSK directory 
dirRSK     = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/Instruments/RBR/RBRsolo/'
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
t0         = pd.to_datetime('2023-09-27 17:20:00')
tend       = pd.to_datetime('2023-09-27 18:10:00')
### specify time for arrows
ta         = pd.to_datetime('2023-09-27 17:24:29')
tb         = pd.to_datetime('2023-09-27 17:37:28')
tc         = pd.to_datetime('2023-09-27 17:47:29')
td         = pd.to_datetime('2023-09-27 18:05:28')


###################################
# define length of window for
# a mean over a 5-points window
window_size = 5
# variable to extract and average
# in rsk files
var1        = 'temperature'
var2        = 'pressure'
# initialize DataFrame
df_rsk      = []
# Start loop
for nn, file in enumerate(fileRSK):
    # extract and average temperature 
    dfm = winmean_rsk_data(dirRSK, file, var1, window_size)
    if file == fileRSK[-1]:
       # extract and average pressure
       # threshold in winmean_rsk_data:
       # substract atmopsheric pressure
       # add one as the 1st thermometer is 
       # at 1 m above the ground
       dfPm = winmean_rsk_data(dirRSK, file, var2, window_size, - 9.13)
       # Extract the event period
       # create a copy to avoid warning
       # convert to xarray
       Pm              = dfPm[t0:tend].copy().to_xarray()
    # Extract the event period
    # create a copy to avoid warning
    dfe             = dfm[t0:tend].copy()
    # Add depth coordinate before converting to xarray
    dfe.loc[:, 'h'] = zSolo[nn]  
    dfe             = dfe.set_index(['h'], append=True)
    # Append
    df_rsk.append(dfe.to_xarray())

# combined temperature along the height dimension
Tmean = xr.concat(df_rsk, dim='h', join='outer')
# Extract every 5th data point along the time axis
T5th  = Tmean.isel(datetime=slice(None, None, 5))
P5th  = Pm.isel(datetime=slice(None, None, 5))
# Create a mask: 
# True where h is greater than P5th 
mask = T5th['h'] > P5th['pressure']
# Mask values where depth is greater than P5th
T5th = T5th.where(~mask) 
# Convert datetime array to 
#T5th['datetime'] = pd.to_datetime(T5th['datetime'].values)

###################################
# Parameters for the inclined arrow to draw
# on the hovmoller of pixel intensity
# incident wave
# slope
slope_i               = -0.7
# distance at t = 0
intercept_i           =  400   
#intercept_i           = 1064 
# Define the line's start and end points in terms of time
line_start_time_i     = pd.Timestamp('2023-09-27 17:23:00')
line_end_time_i       = pd.Timestamp('2023-09-27 17:27:00')
#line_start_time_i     = pd.Timestamp('2023-09-27 17:25:55')
#line_end_time_i       = pd.Timestamp('2023-09-27 17:45:20')
line_start_distance_i = slope_i * (line_start_time_i - t0) / pd.Timedelta(seconds=1) + intercept_i
line_end_distance_i   = slope_i * (line_end_time_i - t0) / pd.Timedelta(seconds=1) + intercept_i
# Add a legend aligned with the arrow
mid_time_i            = line_start_time_i + (line_end_time_i - line_start_time_i) / 2
mid_distance_i        = line_start_distance_i + (line_end_distance_i - line_start_distance_i) / 2
# reflected wave
# slope
slope_r               = 0.5    
# distance at t = 0
intercept_r           =  -1184  
# Define the line's start and end points in terms of time
line_start_time_r     = pd.Timestamp('2023-09-27 18:03:00')
line_end_time_r       = pd.Timestamp('2023-09-27 18:07:00')
line_start_distance_r = slope_r * (line_start_time_r - t0) / pd.Timedelta(seconds=1) + intercept_r
line_end_distance_r   = slope_r * (line_end_time_r - t0) / pd.Timedelta(seconds=1) + intercept_r
# Add a legend aligned with the arrow
mid_time_r            = line_start_time_r + (line_end_time_r - line_start_time_r) / 2
mid_distance_r        = line_start_distance_r + (line_end_distance_r - line_start_distance_r) / 2

###################################
# display figure 
plt.ion()
# start figure 
f1,ax1 = plt.subplots(figsize=(20,8), nrows = 2, ncols = 1, sharex=True)
# pixel intensity row
c0 = ax1[0].pcolormesh(X, Y, pixel_int, shading='gouraud', vmin=-80, vmax=70)
# Apply gray colormap
c0.set_cmap('gray')
# Plot an inclined line
#ax1[0].plot([line_start_time_i, line_end_time_i], [line_start_distance_i, line_end_distance_i], 'k--', lw=0.8)
# Add an arrow with the slope being the incident wave phasespeed 
ax1[0].annotate('', xy=(line_end_time_i, line_end_distance_i), xytext=(line_start_time_i, line_start_distance_i),
            arrowprops=dict(arrowstyle="->", color='w', lw=2.))
# Add an arrow with the slope being the reflected wave phasespeed 
ax1[0].annotate('', xy=(line_end_time_r, line_end_distance_r), xytext=(line_start_time_r, line_start_distance_r),
            arrowprops=dict(arrowstyle="->", color='w', lw=2.))
# Adjust text to be above the arrow and rotate to align with it
# incident wave
ax1[0].text(mid_time_i + pd.Timedelta(seconds=-120), mid_distance_i - 51, r'$c_i$ = 0.7 m s$^{-1}$', color='w', rotation=-41,
        fontsize=14)
# reflected wave
ax1[0].text(mid_time_r + pd.Timedelta(seconds=-173), mid_distance_r - 34, r'$c_r$ = 0.5 m s$^{-1}$', color='w', rotation=30,
        fontsize=14)
# add labels
ax1[0].set_ylabel('$x^{\prime}$ (m)')
# temperature row
# fill nan with color
ax1[1].set_facecolor('lightgrey')
# filled contour
c1 = ax1[1].contourf(T5th['datetime'], T5th['h'], T5th['temperature'].T, Tticks, cmap = cmapT, extend = 'both')
# line contour
ax1[1].contour(T5th['datetime'], T5th['h'], T5th['temperature'].T, Tticks_cb, colors='k', linewidths=0.8)
# add arrows and symbol at different location along x-axis
ax1[1].annotate('(a)', xy=(ta, 19), xytext=(ta, 20.5), horizontalalignment="center", arrowprops=dict(arrowstyle='->',lw=1.5))
ax1[1].annotate('(b)', xy=(tb, 19), xytext=(tb, 20.5), horizontalalignment="center", arrowprops=dict(arrowstyle='->',lw=1.5))
ax1[1].annotate('(c)', xy=(tc, 19), xytext=(tc, 20.5), horizontalalignment="center", arrowprops=dict(arrowstyle='->',lw=1.5))
ax1[1].annotate('(d)', xy=(td, 19), xytext=(td, 20.5), horizontalalignment="center", arrowprops=dict(arrowstyle='->',lw=1.5))
# x-axis format in HH:MM
ax1[1].xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
# Rotate x-axis labels for better readability
plt.gcf().autofmt_xdate()
# set minor tick every minute
ax1[1].xaxis.set_minor_locator(md.MinuteLocator(interval=1))
# set xlim
ax1[1].set_xlim(t0, tend)
# add triangles pointing at depth at which there are thermometers
for y in zSolo:
    ax1[1].annotate('', xy=(tend, y), xytext=(tend + pd.Timedelta(seconds=0.1), y),
                arrowprops=dict(arrowstyle="-|>", color='k', lw=1))
# add labels
ax1[1].set_ylabel('$z$ (m)')
ax1[1].set_xlabel(r'Time (UTC) on 27 September 2023')
# add colorbar
cax = f1.add_axes([0.92, 0.002, 0.015, 0.4])
f1.colorbar(c1, cax=cax, ticks = Tticks_cb, orientation='vertical', label = r'Temperature ($^{\circ}$C)')
# adjuts subplot
f1.subplots_adjust(top=1.,hspace=0.15,bottom=0.)

# save the figure 
f1.savefig(figname,dpi=500,bbox_inches='tight')


