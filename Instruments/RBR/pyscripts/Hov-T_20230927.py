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
import matplotlib.dates   as md
import matplotlib.pyplot  as plt
import datetime

###################################
# define figure name
figname = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/Instruments/RBR/Figs/solo20230927.png'
# define colormap
cmapT   = 'RdYlBu_r'
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

# combined along the height
Tmean = xr.concat(df_rsk, dim='h', join='outer')
# Extract every 5th data point along the time axis
T5th  = Tmean.isel(datetime=slice(None, None, 5))
P5th  = Pm.isel(datetime=slice(None, None, 5))
# Create a mask: 
# True where h is greater than P5th 
mask = T5th['h'] > P5th['pressure']
# Mask values where depth is greater than P5th
T5th = T5th.where(~mask) 
# Convert datetime coordinates to nanosecond precision
T5th['datetime'] = T5th['datetime']
# Convert datetime to nanosecond precision using pandas
T5th['datetime'] = pd.to_datetime(T5th['datetime'].values).astype('datetime64[ns]')
## define contour and colorbar ticks
Tticks    = np.arange(6,18.2,0.2)
Tticks_cb = np.arange(6,20,2)

# display figure 
plt.ion()

# start figure 
f1,ax1 = plt.subplots(figsize=(20,8))
ax1.set_facecolor('lightgrey')
# filled contour
c1 = ax1.contourf(T5th['datetime'], T5th['h'], T5th['temperature'].T, Tticks, cmap = cmapT, extend = 'both')
# line contour
ax1.contour(T5th['datetime'], T5th['h'], T5th['temperature'].T, Tticks_cb, colors='k', linewidths=0.8)
# add arrows and symbol at different location along x-axis
ax1.annotate('(a)', xy=(ta, 19), xytext=(ta, 20.5), horizontalalignment="center", arrowprops=dict(arrowstyle='->',lw=1.5))
ax1.annotate('(b)', xy=(tb, 19), xytext=(tb, 20.5), horizontalalignment="center", arrowprops=dict(arrowstyle='->',lw=1.5))
ax1.annotate('(c)', xy=(tc, 19), xytext=(tc, 20.5), horizontalalignment="center", arrowprops=dict(arrowstyle='->',lw=1.5))
ax1.annotate('(d)', xy=(td, 19), xytext=(td, 20.5), horizontalalignment="center", arrowprops=dict(arrowstyle='->',lw=1.5))
# x-axis format in HH:MM
ax1.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
# Rotate x-axis labels for better readability
plt.gcf().autofmt_xdate()
# set minor tick every minute
ax1.xaxis.set_minor_locator(md.MinuteLocator(interval=1))
# set xlim
ax1.set_xlim(t0, tend)
# add labels
ax1.set_ylabel('$z$ (m)')
ax1.set_xlabel(r'Time (UTC) on 27 September 2023')
# add colorbar
cax = f1.add_axes([0.91, 0.2, 0.015, 0.4])
f1.colorbar(c1, cax=cax, ticks = Tticks_cb, orientation='vertical', label = r'Temperature ($^{\circ}$C)')

# save the figure 
f1.savefig(figname,dpi=500,bbox_inches='tight')


