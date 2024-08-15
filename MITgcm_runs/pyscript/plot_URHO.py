#!/usr/bin/env python 


# plot contour of U and rho at different time
# for simulation of an underwater wavetrain 
# reflecting against the Grande-Anse Wharf 
 
#=========================================
### import modules
import numpy                as np
import xarray               as xr
import matplotlib.pyplot    as plt
from matplotlib.patches import Rectangle

#=========================================
# name of experiment 
exp_name = 'APE5e5'
# path to directory
dirF   = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/MITgcm_runs/ISW4-CTDF14/' + exp_name +'/'
# name of grid file
gfile  = 'grid.glob.nc'
### grid path
gf     = dirF + gfile 
### variable path
df1    = dirF + 'dynDiag.' + exp_name + '.nc' 
# name and path of figure to save 
fgname = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/pictures/CTDF14-' + exp_name + '.png'

#=========================================
# define density surface
rho2   = 1005
# define density bottom
rho1   = 1025
# x- and z-axis limit 
Lmax   = 1000  # [m]
zmax   = 20    # [m]
# convert kn to m/s
kn2ms = 1852/3600
# convert m/s to kn
ms2kn = 1/kn2ms

#=========================================
# load grid 
grid  = xr.open_dataset(gf) 
# depth
depth = grid.Depth.max().values
# length of domain
L     = grid.Xp1.isel(Xp1=-1).values
# create mask
mask  = grid.HFacC.isel(Y=0).where(grid.HFacC.isel(Y=0)>0)
masku = grid.HFacW.isel(Y=0).where(grid.HFacW.isel(Y=0)>0)
# time array 
T     = xr.open_dataset(df1)['T'] 

#=========================================
# define figure characteristics
plt.rcParams['font.family']         = 'Helvetica'
plt.rcParams['axes.titlepad']       = 6
plt.rcParams['xtick.labelsize']     = 12
plt.rcParams['ytick.labelsize']     = 12
plt.rcParams['axes.labelsize' ]     = 12
plt.rcParams['legend.fontsize']     = 12
plt.rcParams['figure.titlesize']    = 14
plt.rcParams['font.size']           = 14
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['savefig.bbox']        = 'standard' 
# define colormap
cmap1   = 'seismic'
# define meshgrids
xx,zz   = np.meshgrid(grid.X, grid.RC)
xxu,zzu = np.meshgrid(grid.Xp1, grid.RC)
# define xticks
xticks  = np.arange(0,1100,100)
# x and z values to patch the topography
xpatch  = grid.Depth.X.isel(X=slice(0,400)).values
zpatch  = grid.Depth.isel(X=slice(0,400)).values.flatten()
# define contour levels for rho
vrho0   = rho2
vrho1   = rho1
rrho    = (vrho1-vrho0)/50
rrhocb  = 2
cbrho   = np.arange(vrho0,vrho1+rrhocb,rrhocb)

#=========================================
# choose time to initialize the start of the 
# simulation find time index we want to display and
# create array for u contours 
# 1e4
if exp_name == 'APE1e4':
    t0   = 700
    t1   = 1075
    t2   = 1155
    t3   = 1250
    t4   = 1340
    t5   = 1440
    #t5   = 2270
    ulev = np.arange(-0.22,0.24,0.02) 
# 1e5
elif exp_name == 'APE1e5':
    t0   = 900
    t1   = 1190
    t2   = 1270
    t3   = 1300
    t4   = 1360
    t5   = 1970
    ulev = np.arange(-1.0,1.1,0.1) 
# 5e5
elif exp_name == 'APE5e5':
    t0   = 700
    t1   = 1015 #2nd i
    t2   = 1110 #1rst r 
    t3   = 1215 #3rd i
    t4   = 1320 #2nd r
    #t5   = 1420 #4rth i
    t5   = 1620 #3rd r
    t6   = 1800 #last peak in timeserie of Fig. 18
    ulev = np.arange(-1.0,1.1,0.1) 
# array of time index
TT     = np.array([t0,t1,t2,t3,t4,t5])
# u level for contours
ulev_kn  = ms2kn * ulev 

#=========================================
# Round small values close to zero to zero for labeling
def format_tick(tick):
    return f'{tick:.2f}' if not np.isclose(tick, 0) else '0.00'

#=========================================
# display figure
plt.ion()
plt.show()

### Start figure 
f1,f1ax = plt.subplots(figsize=(18,12),nrows=len(TT),ncols=1,sharex=True)
f1ax    = f1ax.ravel()
# axis label
f1ax[0].set_ylabel(r'Depth (m)')
f1ax[1].set_ylabel(r'Depth (m)')
f1ax[2].set_ylabel(r'Depth (m)')
f1ax[3].set_ylabel(r'Depth (m)')
f1ax[4].set_ylabel(r'Depth (m)')
f1ax[5].set_ylabel(r'Depth (m)')
f1ax[5].set_xlabel(r'Distance from the wharf (m)')
for ii in range(len(TT)):
    if ii == 0:
        cont1 = f1ax[ii].contourf(xxu,-zzu, masku * xr.open_dataset(df1)['UVELMASS'].sel(T=TT[ii],method='nearest').isel(Y=0).rename({'Zmd000100':'Z'}), levels = ulev, cmap = cmap1, extend = "both")
        f1ax[ii].contour(xx,-zz, mask * xr.open_dataset(df1)['RHOAnoma'].sel(T=TT[ii],method='nearest').isel(Y=0).rename({'Zmd000100':'Z'})+1000, cbrho, colors = 'k', linewidths = 0.8)
        #add a rectangle which scales the Jaeger Arrow
        f1ax[ii].add_patch(Rectangle((150-25/2, 0), 25, 10, color = 'grey', fill = True, alpha = 0.5))
    else:
        f1ax[ii].contourf(xxu,-zzu, masku * xr.open_dataset(df1)['UVELMASS'].sel(T=TT[ii],method='nearest').isel(Y=0).rename({'Zmd000100':'Z'}), levels = ulev, cmap = cmap1, extend = "both")
        f1ax[ii].contour(xx,-zz, mask * xr.open_dataset(df1)['RHOAnoma'].sel(T=TT[ii],method='nearest').isel(Y=0).rename({'Zmd000100':'Z'})+1000, cbrho, colors = 'k', linewidths = 0.8)
    #patch topography
    f1ax[ii].fill_between(xpatch, zpatch, zmax, color = '#ffeabc')
    #add time
    f1ax[ii].text(0.9, 0.09,'$t$ = %0.1f min' %((T.sel(T=TT[ii],method='nearest')-T.sel(T=t0,method='nearest'))/60), transform=f1ax[ii].transAxes, color='k', fontsize=15)
    #set x limit
    f1ax[ii].set_xlim(0, Lmax)
    #set y limit
    f1ax[ii].set_ylim(0, zmax)
    #set x tickts
    f1ax[ii].set_xticks(xticks)
    #invert y axis
    f1ax[ii].invert_yaxis()

# colorbar box
cbax1 = f1.add_axes([0.12, 0.93, 0.35, 0.015])
# colorbar with unit m/s
cbar1 = f1.colorbar(cont1,ax=f1ax[0], cax=cbax1, label='$u$ (m s$^{-1}$)', orientation="horizontal", ticks=ulev)
# set the tick labels, keeping one label over two
cbar1.set_ticklabels([format_tick(tick) if i % 2 == 0 else '' for i, tick in enumerate(ulev)]) 
# ticks size
cbar1.ax.tick_params(labelsize=10)

# create a second axis for the second unit
ax2 = cbar1.ax.twiny()
# set second label
ax2.set_xlabel('$u$ (kn)')
# set ticks for the second axis
ax2.set_xticks(ulev_kn)
# set the tick labels, keeping one label over two
ax2.set_xticklabels([format_tick(tick) if i % 2 == 0 else '' for i, tick in enumerate(ulev_kn)])
# Set the limits of the twin axis to match the primary axis limits
ax2.set_xlim(ulev_kn[0], ulev_kn[-1])
# remove minor ticks
ax2.minorticks_off()
# ticks size
ax2.tick_params(labelsize=10)

# save figure
f1.savefig(fgname,bbox_inches='tight')



