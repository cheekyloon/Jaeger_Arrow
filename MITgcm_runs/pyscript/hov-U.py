#!/usr/bin/env python 


# plot hovmoller of U along the first 200 m
# for the several simulations of an underwater 
# wavetrain reflecting against the Grande-Anse Wharf 
 
#=========================================
### import modules
import numpy                as np
import xarray               as xr
import matplotlib.pyplot    as plt

#=========================================
# name of experiment 
exp_name = ['APE1e5','APE5e5']
# path to directory
dirF   = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/MITgcm_runs/ISW4-CTDF14/' 
# name of grid file
gfile  = '/grid.glob.nc'
### grid path
gf     = dirF + exp_name[0] + gfile 
# name and path of figure to save 
fgname = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/pictures/hov-U-CTDF14.png'

#=========================================
# x-limit 
Lmax   = 200  
# convert kn to m/s
kn2ms = 1852/3600
# convert m/s to kn
ms2kn = 1/kn2ms

#=========================================
# initiate values for array for u contours
vmin = np.array([-0.8, -1.0])
vmax = np.array([1.0, 1.1])
dv   = np.array([0.1, 0.1])
# create array for initial and end times (s)
# for hovmöller period
t0   = np.array([900, 700])
tend = np.array([2100, 1900])
# set title for the figure
title = ['Medium-amplitude wave','Large-amplitude wave']

#=========================================
# load grid 
grid  = xr.open_dataset(gf)

#=========================================
# define figure characteristics
plt.rcParams['font.family']         = 'Helvetica'
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
plt.rcParams['savefig.bbox']        = 'standard' 
# define colormap
cmap1 = 'seismic'
# define yticks
yticks  = np.arange(0,22,2)


#=========================================
# Round small values close to zero to zero for labeling
def format_tick(tick):
    return f'{tick:.2f}' if not np.isclose(tick, 0) else '0.00'

#=========================================
# display figure
plt.ion()
plt.show()

### Start figure 
f,ax = plt.subplots(figsize=(18,12), nrows = len(exp_name), ncols = 1, sharex=True)
# axis label
ax[0].set_ylabel(r'$t$ (min)')
ax[1].set_ylabel(r'$t$ (min)')
ax[1].set_xlabel(r'Distance from the wharf (m)')
# create array for u contours
ulev = np.arange(vmin[1], vmax[1], dv[1])
# u level for contours
ulev_kn = ms2kn * ulev 
for nn, file in enumerate(exp_name):
    ### variable path
    df1    = dirF + file + '/dynDiag.' + file + '.nc'
    # extract U and compute Umax
    U   = xr.open_dataset(df1)['UVELMASS'].isel(Y=0).sel(T=slice(t0[nn],tend[nn])).sel(Xp1=slice(0,Lmax))\
       .rename({'Zmd000100':'Z'}).assign_coords(Z=grid.RC).sel(Z=slice(0,-10)).mean('Z')
    # extract time
    T   = xr.open_dataset(df1)['T'].sel(T=slice(t0[nn],tend[nn]))
    T   = (T - T.isel(T=0))/60 
    # define meshgrids
    xx,yy = np.meshgrid(grid.Xp1.sel(Xp1=slice(0,Lmax)), T)
    # filled contour
    cU = ax[nn].contourf(xx, yy, U, levels = ulev, cmap = cmap1, extend = "both") 
    #set y tickts
    ax[nn].set_yticks(yticks)
    # set title
    ax[nn].set_title(title[nn], loc='right') 

# colorbar box
cbax = f.add_axes([0.12, 0.93, 0.35, 0.015])
# colorbar with unit m/s
cbar = f.colorbar(cU, ax=ax[1], cax=cbax, label='$U_{max}$ (m s$^{-1}$)', orientation="horizontal", ticks=ulev)
# set the tick labels, keeping one label over two
cbar.set_ticklabels([format_tick(tick) if i % 2 == 0 else '' for i, tick in enumerate(ulev)]) 
# ticks size
cbar.ax.tick_params(labelsize=11)

# create a second axis for the second unit
ax2 = cbar.ax.twiny()
# set second label
ax2.set_xlabel('$U_{max}$ (kn)')
# set ticks for the second axis
ax2.set_xticks(ulev_kn)
# set the tick labels, keeping one label over two
ax2.set_xticklabels([format_tick(tick) if i % 2 == 0 else '' for i, tick in enumerate(ulev_kn)])
# Set the limits of the twin axis to match the primary axis limits
ax2.set_xlim(ulev_kn[0], ulev_kn[-1])
# remove minor ticks
ax2.minorticks_off()
# ticks size
ax2.tick_params(labelsize=11)

# adjuts subplot
f.subplots_adjust(hspace=0.15)

# save figure
f.savefig(fgname,bbox_inches='tight')



