#!/usr/bin/env python 


# plot contour of U and rho at different time
# zoomed over a larger depth for the large-amplitude 
# underwater wave simulation
# to see more details of the breaking waves.
 
#=========================================
### import modules
import numpy                as np
import xarray               as xr
import matplotlib.pyplot    as plt
import colormaps            as cm

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
fgname = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/pictures/CTDF14-BreakingWaves.png'

#=========================================
# define density surface
rho2   = 1005
# define density bottom
rho1   = 1025
# x- and z-axis limit 
Lmax   = 100 # [m]
zmax   = 40  # [m]
# convert kn to m/s
kn2ms = 1852/3600
# convert m/s to kn
ms2kn = 1/kn2ms

#=========================================
# load grid 
grid  = xr.open_dataset(gf) 
# create mask
mask  = grid.HFacC.isel(Y=0).where(grid.HFacC.isel(Y=0)>0)
masku = grid.HFacW.isel(Y=0).where(grid.HFacW.isel(Y=0)>0)

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
#cmap2   = cm.blue
cmap2   = 'RdYlBu' 
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
rrhocb  = 2
rholev  = np.arange(vrho0,vrho1+rrhocb,rrhocb)
# define contour for u
ulev = np.arange(-1.0,1.1,0.1)
# u level for contours
ulev_kn  = ms2kn * ulev 
# set title for the figure
title = ['Medium-amplitude wave','Large-amplitude wave']

#=========================================
# choose time to initialize the start of the 
# simulation and find time index we want to display 
# 1e5
tini1 = 900
TT1   = np.array([1145, 1205, 1235, 1255])
# 5e5
tini2 = 700
TT2   = np.array([905, 930, 960, 1040])

#=========================================
# Round small values close to zero to zero for labeling
def format_tick(tick):
    return f'{tick:.2f}' if not np.isclose(tick, 0) else '0.00'

#=========================================
# display figure
plt.ion()
plt.show()

### Start figure 
f1,f1ax = plt.subplots(figsize=(18,12),nrows=4,ncols=len(TT1),sharex=True,sharey=True)
# y-axis label
f1ax[0,0].set_ylabel(r'Depth (m)')
f1ax[1,0].set_ylabel(r'Depth (m)')
f1ax[2,0].set_ylabel(r'Depth (m)')
f1ax[3,0].set_ylabel(r'Depth (m)')

for nn, file in enumerate(exp_name):
    ### variable path
    df1 = dirF + file + '/dynDiag.' + file + '.nc' 
    # time array 
    T   = xr.open_dataset(df1)['T'] 
    # set title
    f1ax[0,0].set_title(title[0], fontdict={'fontweight': 'bold', 'family': 'serif'}, loc='left')
    f1ax[2,0].set_title(title[1], fontdict={'fontweight': 'bold', 'family': 'serif'}, loc='left')

    for ii in range(len(TT1)):
        if file==exp_name[0]:
           f1ax[0,ii].contourf(xx,-zz, mask * xr.open_dataset(df1)['RHOAnoma'].sel(T=TT1[ii],method='nearest').isel(Y=0).rename({'Zmd000100':'Z'})+1000, rholev, cmap = cmap2, extend = "both") 
           f1ax[1,ii].contourf(xxu,-zzu, masku * xr.open_dataset(df1)['UVELMASS'].sel(T=TT1[ii],method='nearest').isel(Y=0).rename({'Zmd000100':'Z'}), levels = ulev, cmap = cmap1, extend = "both")
           #add time
           f1ax[0,ii].text(0.05, 0.04,'$t$ = %0.1f min' %((T.sel(T=TT1[ii],method='nearest')-T.sel(T=tini1,method='nearest'))/60), transform=f1ax[0,ii].transAxes, color='k', fontsize=14)
        elif file==exp_name[1] and ii==0:
           cont1 = f1ax[2,ii].contourf(xx,-zz, mask * xr.open_dataset(df1)['RHOAnoma'].sel(T=TT2[ii],method='nearest').isel(Y=0).rename({'Zmd000100':'Z'})+1000, rholev, cmap = cmap2, extend = "both")
           cont2 = f1ax[3,ii].contourf(xxu,-zzu, masku * xr.open_dataset(df1)['UVELMASS'].sel(T=TT2[ii],method='nearest').isel(Y=0).rename({'Zmd000100':'Z'}), levels = ulev, cmap = cmap1, extend = "both")
           f1ax[2,ii].text(0.05, 0.04,'$t$ = %0.1f min' %((T.sel(T=TT2[ii],method='nearest')-T.sel(T=tini2,method='nearest'))/60), transform=f1ax[2,ii].transAxes, color='k', fontsize=14)
        else:
           f1ax[2,ii].contourf(xx,-zz, mask * xr.open_dataset(df1)['RHOAnoma'].sel(T=TT2[ii],method='nearest').isel(Y=0).rename({'Zmd000100':'Z'})+1000, rholev, cmap = cmap2, extend = "both") 
           f1ax[3,ii].contourf(xxu,-zzu, masku * xr.open_dataset(df1)['UVELMASS'].sel(T=TT2[ii],method='nearest').isel(Y=0).rename({'Zmd000100':'Z'}), levels = ulev, cmap = cmap1, extend = "both")
           f1ax[2,ii].text(0.05, 0.04,'$t$ = %0.1f min' %((T.sel(T=TT2[ii],method='nearest')-T.sel(T=tini2,method='nearest'))/60), transform=f1ax[2,ii].transAxes, color='k', fontsize=14)
        #patch topography
        f1ax[0,ii].fill_between(xpatch, zpatch, zmax, color = '#ffeabc')
        f1ax[1,ii].fill_between(xpatch, zpatch, zmax, color = '#ffeabc')
        f1ax[2,ii].fill_between(xpatch, zpatch, zmax, color = '#ffeabc')
        f1ax[3,ii].fill_between(xpatch, zpatch, zmax, color = '#ffeabc')
        #set x limit
        f1ax[0,ii].set_xlim(0, Lmax)
        f1ax[1,ii].set_xlim(0, Lmax)
        f1ax[2,ii].set_xlim(0, Lmax)
        f1ax[3,ii].set_xlim(0, Lmax)
        #set x tickts
        #f1ax[ii].set_xticks(xticks)
        #invert y axis
        f1ax[0,ii].invert_yaxis()
        f1ax[1,ii].invert_yaxis()
        f1ax[2,ii].invert_yaxis()
        f1ax[3,ii].invert_yaxis()
        #set y limit
        f1ax[0,ii].set_ylim(zmax, 0)
        f1ax[1,ii].set_ylim(zmax, 0)
        f1ax[2,ii].set_ylim(zmax, 0)
        f1ax[3,ii].set_ylim(zmax, 0)
        # x-axis label
        f1ax[3,ii].set_xlabel(r'Distance from the wharf (m)')

# colorbar box
cbax1 = f1.add_axes([0.91, 0.63, 0.015, 0.25])
cbax2 = f1.add_axes([0.95, 0.105, 0.015, 0.25])
# colorbar with unit m/s
cbar1 = f1.colorbar(cont1,ax=f1ax[0,0], cax=cbax1, label=r'$\rho$ (kg m$^{-3}$)', orientation="vertical", ticks=rholev)
cbar2 = f1.colorbar(cont2,ax=f1ax[1,0], cax=cbax2, label=r'$u$ (m s$^{-1}$)', orientation="vertical", ticks=ulev)
# set label to the left
cbar2.ax.yaxis.set_label_position('left')
# set the tick labels, keeping one label over two
cbar2.set_ticklabels([format_tick(tick) if i % 2 == 0 else '' for i, tick in enumerate(ulev)]) 
# ticks size
cbar1.ax.tick_params(labelsize=10)
cbar2.ax.tick_params(labelsize=10)

# create a second axis for the second unit
ax2 = cbar2.ax.twinx()
# set second label
ax2.set_ylabel('$u$ (kn)')
# set ticks for the second axis
ax2.set_yticks(ulev_kn)
# set the tick labels, keeping one label over two
ax2.set_yticklabels([format_tick(tick) if i % 2 == 0 else '' for i, tick in enumerate(ulev_kn)])
# Set the limits of the twin axis to match the primary axis limits
ax2.set_ylim(ulev_kn[0], ulev_kn[-1])
# remove minor ticks
ax2.minorticks_off()
# ticks size
ax2.tick_params(labelsize=10)

# adjuts subplot
f1.subplots_adjust(hspace=0.15, wspace=0.1)

# save figure
f1.savefig(fgname,bbox_inches='tight')



