#!/Users/sandy/miniconda3/bin/python

### plot T, S and rho vs depth
### and T vs rho with polynomial fit 
### for the CTD profile
### done on the 11 September 2024

# import modules
from pyrsktools          import RSK
import gsw
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

#=========================================
### path to root directory, and rsk file name
dirF    = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/Instruments/RBR/RBRconcerto/'
fileCTD = '207669_20240911_0919.rsk'

#=========================================
### figure
# path to save the figures
figdir  = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/Instruments/RBR/Figs/'
figname = figdir + 'CTD20240911.png'
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
# display figures
plt.ion()

#=========================================
### Open the RSK file and load the variables 
# Initiate an RSK class object, passing the path to an RSK file
rskCTD = RSK(dirF+fileCTD)
# open class object and read data
rskCTD.open()
rskCTD.readdata()
# exemple to print informations
#rskCTD.printchannels()
# load data
P     = rskCTD.data["sea_pressure"]
z     = rskCTD.data["depth"]
S     = rskCTD.data["salinity"]
theta = rskCTD.data["temperature"]
time  = rskCTD.data["timestamp"]
# close class object
rskCTD.close()

#=========================================
# Compute density
rho  = gsw.density.rho_t_exact(S, theta, P)

#=========================================
# Create a DataFrame
df = pd.DataFrame({'S':S, 'T':theta, 'P':P, 'rho':rho, 'depth':z})
# Add datetime with units (milliseconds)
df['datetime'] = pd.to_datetime(time, unit='ms')
# Set the datetime column as the index 
df.set_index('datetime', inplace=True)

#=========================================
# Mask values when the CTD was out of water
# i.e. values where P < 0
df  = df.mask(df['P']<0)
# Drop rows where 'depth' is NaN
df1 = df.dropna(subset=['depth']).copy()

#=========================================
# display depth to see the descent and ascent of CTD
#plt.figure()
#plt.plot(df1['depth'])

# times around the start of the first descent
tstart = pd.to_datetime('2024-09-11 13:11:00')
tend   = pd.to_datetime('2024-09-11 13:12:00')
td1    = df1[tstart:tend]['depth'].idxmin()
# times around the start of the first ascent
tstart = pd.to_datetime('2024-09-11 13:12:00')
tend   = pd.to_datetime('2024-09-11 13:13:00')
ta1    = df1[tstart:tend]['depth'].idxmax()
# times around the end of first ascent
tstart = pd.to_datetime('2024-09-11 13:13:00')
tend   = pd.to_datetime('2024-09-11 13:14:00')
ta1end = df1[tstart:tend]['depth'].idxmin()
# times around the start of the second descent
tstart = pd.to_datetime('2024-09-11 13:13:30')
tend   = pd.to_datetime('2024-09-11 13:14:00')
td2    = df1[tstart:tend]['depth'].idxmin()
# times around the start of the second ascent
tstart = pd.to_datetime('2024-09-11 13:14:00')
tend   = pd.to_datetime('2024-09-11 13:15:00')
ta2    = df1[tstart:tend]['depth'].idxmax()
# times around the end of second ascent
tstart = pd.to_datetime('2024-09-11 13:15:00')
tend   = pd.to_datetime('2024-09-11 13:15:05')
ta2end = df1[tstart:tend]['depth'].idxmin()
# times around the start of the third descent
tstart = pd.to_datetime('2024-09-11 13:15:05')
tend   = pd.to_datetime('2024-09-11 13:16:00')
td3    = df1[tstart:tend]['depth'].idxmin()
# times around the start of the third ascent
tstart = pd.to_datetime('2024-09-11 13:15:00')
tend   = pd.to_datetime('2024-09-11 13:17:00')
ta3    = df1[tstart:tend]['depth'].idxmax()
# times around the end of second ascent
tstart = pd.to_datetime('2024-09-11 13:16:00')
tend   = pd.to_datetime('2024-09-11 13:16:35')
ta3end = df1[tstart:tend]['depth'].idxmin()

# Start figure
f1, f1ax = plt.subplots(figsize=(18,12), ncols = 3, nrows = 1, sharey = True)
# first descent and ascent
f1ax[0].plot(df1[td1:ta1]['T'], df1[td1:ta1]['depth'], 'b')
f1ax[0].plot(df1[ta1:ta1end]['T'], df1[ta1:ta1end]['depth'], 'r')
f1ax[0].legend(['descente','montée'])
f1ax[0].set_title('Première descente/montée')
# second descent and ascent
f1ax[1].plot(df1[td2:ta2]['T'], df1[td2:ta2]['depth'], 'b')
f1ax[1].plot(df1[ta2:ta2end]['T'], df1[ta2:ta2end]['depth'], 'r')
f1ax[1].set_title('Deuxième descente/montée')
# third descent and ascent
f1ax[2].plot(df1[td3:ta3]['T'], df1[td3:ta3]['depth'], 'b')
f1ax[2].plot(df1[ta3:ta3end]['T'], df1[ta3:ta3end]['depth'], 'r')
f1ax[2].set_title('Troisième descente/montée')
# xlabels
f1ax[0].set_xlabel(r"Temperature ($^{\circ}$C)")
f1ax[1].set_xlabel(r"Temperature ($^{\circ}$C)")
f1ax[2].set_xlabel(r"Temperature ($^{\circ}$C)")
# ylabels
f1ax[0].set_ylabel(r"Profondeur (m)")
f1.suptitle('CTD du 11 septembre 2024')
# x-limit
f1ax[0].set_xlim([2, 18])
# y-limit
f1ax[0].set_ylim([0, 18])
# save the plot as a file
#f1.savefig(figdir + 'CTD20240911descent-ascent.png',dpi=500,bbox_inches='tight')

# remove ascent from dataframe
df2 = pd.concat([df1[td1:ta1], df1[td2:ta2], df1[td3:ta3]])
# compute the polynomial fit of 2nd order
p2  = np.poly1d(np.polyfit(df2['T'], df2['rho'], 2))
ip2 = np.argsort(p2(df2['T']))


### 

# Start figure
f2, ax = plt.subplots(figsize=(18,12), ncols = 2, nrows = 1)
# Add some extra space for the second and third axes at the bottom
f2.subplots_adjust(bottom=0.3, wspace=0.4)

# create twin axis to have shared yaxis
ax3 = ax[0].twiny()
ax4 = ax[0].twiny()
# Move twinned axis ticks and label from top to bottom
ax3.xaxis.set_ticks_position("bottom")
ax3.xaxis.set_label_position("bottom")
ax4.xaxis.set_ticks_position("bottom")
ax4.xaxis.set_label_position("bottom")
# Offset the twin axis below the host
ax3.spines["bottom"].set_position(("axes", -0.22))
ax4.spines["bottom"].set_position(("axes", -0.43))

# subplot 1 
# T
ax[0].plot(df1[td1:ta1]['T'], df1[td1:ta1]['depth'], "b-", linewidth=0.5)
ax[0].plot(df1[td2:ta2]['T'], df1[td2:ta2]['depth'], "b-", linewidth=0.5)
ax[0].plot(df1[td3:ta3]['T'], df1[td3:ta3]['depth'], "b-", linewidth=0.5)
ax[0].set_xlim([2,18])
ax[0].set_ylim([0,18])
ax[0].set_xlabel("Temperature ($^\circ$C)")
ax[0].xaxis.label.set_color('b')
ax[0].set_ylabel("Profondeur (m)")
ax[0].spines['bottom'].set_color('b')
ax[0].tick_params(axis='x', colors='b')
# S
ax3.plot(df1[td1:ta1]['S'], df1[td1:ta1]['depth'], "r-", linewidth=0.5)
ax3.plot(df1[td2:ta2]['S'], df1[td2:ta2]['depth'], "r-", linewidth=0.5)
ax3.plot(df1[td3:ta3]['S'], df1[td3:ta3]['depth'], "r-", linewidth=0.5)
ax3.set_xlim([4,30])
ax3.set_xlabel("Salinité (PSU)")
ax3.xaxis.label.set_color('r')
ax3.spines['bottom'].set_color('r')
ax3.tick_params(axis='x', colors='r')
# rho
ax4.plot(df1[td1:ta1]['rho'], df1[td1:ta1]['depth'], "g-", linewidth=0.5)
ax4.plot(df1[td2:ta2]['rho'], df1[td2:ta2]['depth'], "g-", linewidth=0.5)
ax4.plot(df1[td3:ta3]['rho'], df1[td3:ta3]['depth'], "g-", linewidth=0.5)
ax4.set_xlim([1002,1024])
ax4.set_xlabel("Densité (kg m$^{-3}$)")
ax4.xaxis.label.set_color('g')
ax4.spines['bottom'].set_color('g')
ax4.tick_params(axis='x', colors='g')
# square plot
ax[0].set_box_aspect(1)

# subplot 2
ax[1].plot(df2['T'].iloc[ip2], p2(df2['T'].iloc[ip2]), color='red', linewidth=0.5)
ax[1].plot(df2['T'], df2['rho'], '.', markersize=1)
ax[1].set_xlim([2,18])
ax[1].set_ylim([1002,1024])
ax[1].set_xlabel("Temperature ($^\circ$C)")
ax[1].set_ylabel("Densité (kg m$^{-3}$)")
# Manually format the polynomial equation with spaces after signs
poly_eq = r'$\rho$ = %.2f T$^2$  %s%.2f T  %s%.2f' % (
    p2[0],
    '+ ' if p2[1] >= 0 else '- ', abs(p2[1]),
    '+ ' if p2[2] >= 0 else '- ', abs(p2[2])
)
# Add the legend
ax[1].legend([poly_eq], fontsize=12.0, bbox_to_anchor=(1, 1), loc=1)
# square plot
ax[1].set_box_aspect(1)

# save the plot as a file
f2.savefig(figname,dpi=500,bbox_inches='tight')


