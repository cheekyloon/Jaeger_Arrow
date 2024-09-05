#!/usr/bin/env python 


# plot contour of rho at different time
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
rho1   = 1022
# x- and z-axis limit 
Lmax   = 100 # [m]
zmax   = 40  # [m]

#=========================================
# load grid 
grid  = xr.open_dataset(gf) 
# create mask
mask  = grid.HFacC.isel(Y=0).where(grid.HFacC.isel(Y=0)>0)

#=========================================
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
plt.rcParams['savefig.bbox']        = 'standard'
# define colormap
cmap2   = 'RdYlBu' 
# define meshgrids
xx,zz   = np.meshgrid(grid.X, grid.RC)
# define xticks
xticks  = np.arange(0,1100,100)
# x and z values to patch the topography
xpatch  = grid.Depth.X.isel(X=slice(0,400)).values
zpatch  = grid.Depth.isel(X=slice(0,400)).values.flatten()
# define contour levels for rho
vrho2    = rho2
vrho1    = rho1
rrho     = 0.2
rrhocb   = 2
rholev   = np.arange(vrho2,vrho1+rrho,rrho)
rholevcb = np.arange(vrho2,vrho1+rrhocb,rrhocb)
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
# display figure
plt.ion()
plt.show()

### Start figure 
f1,f1ax = plt.subplots(figsize=(18,12),ncols=len(exp_name),nrows=len(TT1),sharex=True,sharey=True)

for nn, file in enumerate(exp_name):
    ### variable path
    df1 = dirF + file + '/dynDiag.' + file + '.nc' 
    # time array 
    T   = xr.open_dataset(df1)['T'] 
    # set title
    f1ax[0,nn].set_title(title[nn], fontdict={'fontweight': 'bold', 'family': 'serif'}, loc='left')
    f1ax[0,nn].set_title(title[nn], fontdict={'fontweight': 'bold', 'family': 'serif'}, loc='left')

    for ii in range(len(TT1)):
        if file==exp_name[0]:
           f1ax[ii,nn].contourf(xx,-zz, mask * xr.open_dataset(df1)['RHOAnoma'].sel(T=TT1[ii],method='nearest').isel(Y=0).rename({'Zmd000100':'Z'})+1000, rholev[:-1], cmap = cmap2, extend = "both") 
           #add time
           f1ax[ii,nn].text(0.05, 0.04,'$t$ = %0.1f min' %((T.sel(T=TT1[ii],method='nearest')-T.sel(T=tini1,method='nearest'))/60), transform=f1ax[ii,nn].transAxes, color='k', fontsize=14)
        elif file==exp_name[1] and ii==0:
           cont1 = f1ax[ii,1].contourf(xx,-zz, mask * xr.open_dataset(df1)['RHOAnoma'].sel(T=TT2[ii],method='nearest').isel(Y=0).rename({'Zmd000100':'Z'})+1000, rholev[:-1], cmap = cmap2, extend = "both")
           f1ax[ii,nn].text(0.05, 0.04,'$t$ = %0.1f min' %((T.sel(T=TT2[ii],method='nearest')-T.sel(T=tini2,method='nearest'))/60), transform=f1ax[ii,nn].transAxes, color='k', fontsize=14)
        else:
           f1ax[ii,nn].contourf(xx,-zz, mask * xr.open_dataset(df1)['RHOAnoma'].sel(T=TT2[ii],method='nearest').isel(Y=0).rename({'Zmd000100':'Z'})+1000, rholev[:-1], cmap = cmap2, extend = "both") 
           f1ax[ii,nn].text(0.05, 0.04,'$t$ = %0.1f min' %((T.sel(T=TT2[ii],method='nearest')-T.sel(T=tini2,method='nearest'))/60), transform=f1ax[ii,nn].transAxes, color='k', fontsize=14)
        #patch topography
        f1ax[ii,nn].fill_between(xpatch, zpatch, zmax, color = '#ffeabc')
        #set x limit
        f1ax[ii,nn].set_xlim(0, Lmax)
        #invert y axis
        f1ax[ii,nn].invert_yaxis()
        #set y limit
        f1ax[ii,nn].set_ylim(zmax, 0)
        if ii == len(TT1)-1:
           # x-axis label
           f1ax[ii,nn].set_xlabel(r'Distance from the wharf (m)')
        # y-axis label
        f1ax[ii,0].set_ylabel(r'Depth (m)')

# colorbar box
cbax1 = f1.add_axes([0.91, 0.63, 0.015, 0.25])
# colorbar with unit m/s
cbar1 = f1.colorbar(cont1,ax=f1ax[0,1], cax=cbax1, label=r'$\rho$ (kg m$^{-3}$)', orientation="vertical", ticks=rholevcb[:-1])
# ticks size
cbar1.ax.tick_params(labelsize=10)


# adjuts subplot
f1.subplots_adjust(hspace=0.2, wspace=0.1)

# save figure
f1.savefig(fgname,bbox_inches='tight')



