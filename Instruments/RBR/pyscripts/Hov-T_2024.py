#!/Users/sandy/miniconda3/bin/python


# This script plot hovmoller
# of temperature from the thermometers
# deployed at Grande-Anse wharf
# in summer 2024 

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
### define variables
# name of saved figure
fname = '2024'
if fname=='2024':
    # beginning and end time of event
    t0   = pd.to_datetime('2024-07-24 00:00:00')
    tend = pd.to_datetime('2024-09-11 12:00:00')
    # x-axis format in YYYY-mm-dd
    fmt  = '%Y-%m-%d'
    # figure title
    xlab = 'Time (UTC)'  
else:
    # beginning and end time of event
    t0   = pd.to_datetime('2024-07-24 11:30:00')
    tend = pd.to_datetime('2024-07-24 13:30:00')
    # x-axis format in HH:MM
    fmt  = '%H:%M' 
    # figure title
    xlab = 'Time (UTC) on 24 July 2024'  
### specify time for arrows
ta         = pd.to_datetime('2024-07-24 17:09:08')
tb         = pd.to_datetime('2024-07-24 17:15:29')
tc         = pd.to_datetime('2024-07-24 17:20:28')
td         = pd.to_datetime('2024-07-24 17:28:28')

###################################
# define figure name
figname   = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/Instruments/RBR/Figs/hov-T_' + fname + '.png'
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
# define contour and colorbar ticks for temperature
Tticks    = np.arange(6,18.2,0.2)
Tticks_cb = np.arange(6,20,2)
# set yticks for temperature
tticks    = np.arange(0,20,2)

###################################
# define RSK directory 
dirRSK     = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/Instruments/RBR/RBRsolo/20240911/'
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

###################################
# display figure 
plt.ion()
# start figure 
f1,ax1 = plt.subplots(figsize=(20,8), nrows = 2, ncols = 1, sharex=True)
# don't show top panel 
ax1[0].axis("off")
# fill nan with color
ax1[1].set_facecolor('lightgrey')
# filled contour
c1 = ax1[1].contourf(T5th['datetime'], T5th['h'], T5th['temperature'].T, Tticks, cmap = cmapT, extend = 'both')
if fmt=='%H:%M':
   # line contour
   ax1[1].contour(T5th['datetime'], T5th['h'], T5th['temperature'].T, Tticks_cb, colors='k', linewidths=0.8)
   # add arrows and symbol at different location along x-axis
   #ax1[1].annotate('(a)', xy=(ta, 19), xytext=(ta, 20.5), horizontalalignment="center", arrowprops=dict(arrowstyle='->',lw=1.5))
   #ax1[1].annotate('(b)', xy=(tb, 19), xytext=(tb, 20.5), horizontalalignment="center", arrowprops=dict(arrowstyle='->',lw=1.5))
   #ax1[1].annotate('(c)', xy=(tc, 19), xytext=(tc, 20.5), horizontalalignment="center", arrowprops=dict(arrowstyle='->',lw=1.5))
   #ax1[1].annotate('(d)', xy=(td, 19), xytext=(td, 20.5), horizontalalignment="center", arrowprops=dict(arrowstyle='->',lw=1.5))
# x-axis format 
ax1[1].xaxis.set_major_formatter(md.DateFormatter(fmt))
# Rotate x-axis labels for better readability
plt.gcf().autofmt_xdate()
if fmt=='%H:%M':
   # set minor tick every minute
   ax1[1].xaxis.set_minor_locator(md.MinuteLocator(interval=1))
# set xlim
ax1[1].set_xlim(t0, tend)
# add triangles pointing at depth at which there are thermometers
for y in zSolo:
    ax1[1].annotate('', xy=(tend, y), xytext=(tend + pd.Timedelta(seconds=0.1), y),
                arrowprops=dict(arrowstyle="-|>", color='k', lw=1))
# set y-ticks
ax1[1].set_yticks(tticks)
# add labels
ax1[1].set_ylabel('$z$ (m)')
ax1[1].set_xlabel(xlab)
# add colorbar
cax = f1.add_axes([0.92, 0.002, 0.015, 0.4])
f1.colorbar(c1, cax=cax, ticks = Tticks_cb, orientation='vertical', label = r'Temperature ($^{\circ}$C)')
# adjuts subplot
f1.subplots_adjust(top=1.,hspace=0.15,bottom=0.)

# save the figure 
f1.savefig(figname,dpi=500,bbox_inches='tight')


