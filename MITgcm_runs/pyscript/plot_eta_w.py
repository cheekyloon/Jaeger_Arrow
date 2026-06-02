#!/usr/bin/env python

"""
Generate 4-panel figures of MITgcm free-surface elevation (eta) or velocity
fields overlaid on a georeferenced Google Earth image of the Grande-Anse terminal
(Saguenay Fjord, Québec).

This script:
1. Loads the MITgcm model grid and diagnostics.
2. Loads a georeferenced Google Earth image of the Grande-Anse wharf.
3. Converts the image coordinates from geographic coordinates (WGS84)
   to MTM Zone 7 projected coordinates.
4. Applies the same rotation used during bathymetry generation so the
   Google Earth image aligns with the rotated MITgcm model grid.
5. Displays the satellite image beneath the model fields.
6. Plots selected timesteps in a 2×2 panel layout.
7. Uses a symmetric colormap centered on zero to highlight positive
   and negative free-surface anomalies.
8. Masks land areas so the satellite image remains visible over the wharf
   and shoreline.
9. Produces publication-style figures suitable for scientific visualization
   of internal wave propagation near the Grande-Anse terminal.

Input files:
- MITgcm grid file (grid.glob.nc)
- MITgcm surface diagnostics (SurfDiag.glob.nc)
- MITgcm dynamical diagnostics (dynDiag.glob.nc)
- Wharf reference coordinates (.dat)
- Georeferenced Google Earth image (.mat)

Output:
- 4-panel PNG figures of eta or velocity fields overlaid on the
  Grande-Anse satellite image.

Author: Sandy Gregorio
Date: May 2026
"""

import os
import numpy             as np
import pandas            as pd
import xarray            as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors   import LinearSegmentedColormap
from scipy.io            import loadmat
from pyproj              import Transformer
#=========================================
exp_name  = '45Deg_4ISW_sponge_extent_right'
dirF      = '/Volumes/LaCie/JaegerArrow/MITgcm_runs/'
GA_dir   = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/topo_GA/Data/'



gfile     = os.path.join(dirF, exp_name, 'mnc_glob/grid.glob.nc')
surf_file = os.path.join(dirF, exp_name, 'mnc_glob/SurfDiag.0000000000.glob.nc')
dyn_file  = os.path.join(dirF, exp_name, 'mnc_glob/dynDiag.glob.nc')

fgname_eta = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/pictures/eta_' + exp_name + '.png'
fgname_w   = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/pictures/w_' + exp_name + '.png'

#=========================================
# Load Grande-Anse wharf

def convert_latlon_to_mtm(lon, lat):
    """
    Convert WGS84 coordinates (longitude, latitude) to MTM zone 7 (x, y in meters).

    Parameters:
    - lon: float or array-like (longitude)
    - lat: float or array-like (latitude)

    Returns:
    - x: MTM Easting in meters
    - y: MTM Northing in meters
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2949", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y

def compute_rotation_angle(x1, y1, x2, y2):
    """
    Compute the angle (in radians) between the line (x1, y1) to (x2, y2)
    and the horizontal x-axis.
    """
    dx = x2 - x1
    dy = y2 - y1
    return np.arctan2(dy, dx)

def rotate_coordinates(points, origin, theta_rad):
    """
    Rotate a set of 2D points (Nx2) around an origin by a given angle in radians.
    """
    R = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad),  np.cos(theta_rad)]
    ])
    return (points - origin) @ R.T


# Wharf endpoints
df_w = pd.read_csv(
    GA_dir + 'Wharf_Grande-Anse.dat',
    sep='\s+',
    header=None,
    names=['x', 'y', 'val']
)

x1, y1 = df_w['x'][0], df_w['y'][0]
x2, y2 = df_w['x'][1], df_w['y'][1]

# Rotation angle
theta = compute_rotation_angle(x1, y1, x2, y2)

# Rotation origin = middle of wharf
origin = np.array([
    (x1 + x2) / 2,
    (y1 + y2) / 2
])

mat_img = loadmat(GA_dir + 'Grande_Anse_Google_map.mat')
img = mat_img['img_GA']
lat = mat_img['lat_GA'].squeeze()
lon = mat_img['lon_GA'].squeeze()

lon2d, lat2d = np.meshgrid(lon, lat)

x_img, y_img = convert_latlon_to_mtm(lon2d, lat2d)

coords_img = np.column_stack((x_img.ravel(), y_img.ravel()))

coords_img_rot = rotate_coordinates(
    coords_img,
    origin,
    -theta
)

x_img_rot = coords_img_rot[:, 0].reshape(x_img.shape)
y_img_rot = coords_img_rot[:, 1].reshape(y_img.shape)

img = img / 255.0

dx_img = -25.0
dy_img = -65.0 

# RGB image
rgb = img[..., :3]

# Convert to grayscale
img_gray = rgb.mean(axis=2)

# Detect very bright pixels
mask_bad = img_gray > 0.72

# Make them transparent
img_gray = np.where(mask_bad, np.nan, img_gray)

#=========================================
# Load grid
grid = xr.open_dataset(gfile)

# Mask topo / land
mask_surf = grid.HFacC.isel(Z=0)
land = mask_surf.where(mask_surf == 0)

# Load eta and w
eta = xr.open_dataset(surf_file).ETAN.isel(Zd000001=0)
#w   = xr.open_dataset(dyn_file).WVELMASS.isel(Zld000089=0)

# Coordinates
xx, yy = np.meshgrid(grid.X.values, grid.Y.values)

# Times to plot
itimes = [1, 120, 240, 326]

# If T is in seconds in your file:
times_sec = eta.coords['T'].isel(T=itimes).values 
times_min = times_sec/60

#=========================================
# Figure style
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titlepad'] = 10
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True

isw_colors = [
    "#1f2a44",  # deep blue
    "#4b5678",  # blue-gray
    "#7483a5",  # muted steel blue
    "#a9bccd",  # pale blue
    "#e4edf2"   # soft white
]

isw_cmap = LinearSegmentedColormap.from_list(
    "isw_satellite",
    isw_colors,
    N=256
)

# Colormap
plt.ion()

#=========================================
def plot_4panels(var, fgname, cbar_label, cmap='bone', land_color='#ffeabc'):

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Symmetric limits around zero
    vmax = 0.4 * np.nanmax(np.abs(var.isel(T=itimes).values))
    vmin = -vmax
    levels = np.linspace(vmin, vmax, 31)
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    #bg_color = cmap(norm(0.0))
    #bg = np.array(cmap(norm(0.0))[:3])
    #bg_dark = tuple(0.85 * bg)
    #bg_light = tuple(0.75 * bg + 0.25 * np.ones(3))
    #bg_color = "#262626"

    # Extract a small portion of the image to use as a background texture 
    texture = img[900:1100, 200:400].copy()
    # Convert RGB image to grayscale
    texture_gray = texture.mean(axis=2)
    # Reduce brightness to obtain a subtle background texture
    texture_gray *= 0.25
    # Add Gaussian noise to mimic paper grain
    noise = 0.10 * np.random.randn(*texture_gray.shape)
    texture_gray = np.clip(texture_gray + noise, 0, 1)

    fig, ax = plt.subplots(
        figsize=(8, 9.3),
        ncols=2,
        nrows=2,
        sharex=True,
        sharey=True
    )

    axes = ax.flatten()

    for a, it, tmin in zip(axes, itimes, times_min):

        # Google Earth image
        a.pcolormesh(
            x_img_rot + dx_img,
            y_img_rot + dy_img,
            img_gray,
            shading='auto',
            cmap='gray',
            zorder=0
        )

        # eta seulement là où il y a de l'eau
        var_plot = var.isel(T=it).where(mask_surf > 0)

        cf = a.contourf(
            xx, yy,
            var_plot,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend='both',
            zorder=5
        )

        #a.contour(
        #    xx, yy,
        #    mask_surf,
        #    levels=[0.5],
        #    colors='red',
        #    linewidths=0.8,
        #    zorder=20
        #)


        # Topo
        #a.contourf(
        #    xx, yy,
        #    land,
        #    levels=[-0.5, 0.5],
        #    colors=[land_color],
        #    alpha=0.4,
        #    zorder=10
        #)
 

        a.imshow(
            texture_gray,
            extent=[
                grid.X.min(),
                grid.X.max(),
                grid.Y.min(),
                grid.Y.max()
            ],
            cmap='gray',
            aspect='equal',
            alpha=0.5,
            zorder=-10
        )
        #a.set_facecolor(bg_color)
        #a.set_facecolor(bg_dark)
        #a.set_facecolor(bg_light)
        a.text(
            0.02, 0.03,
            f'time = {float(tmin-times_min[0]):.0f} min',
            transform=a.transAxes,
            ha='left',
            va='bottom',
            fontsize=11,
            bbox=dict(
                facecolor='white',
                edgecolor='0.4',
                boxstyle='round,pad=0.25',
                alpha=0.85
            ),
            zorder=20
        )

        a.tick_params(
            which='both',
            direction='out',
            top=True,
            right=True,
            labelbottom=False,
            labelleft=False
        )

        a.set_xlim(grid.X.min().values, grid.X.max().values)
        a.set_ylim(grid.Y.min().values, grid.Y.max().values)
        a.set_aspect('equal', adjustable='box')

#    ax[1, 0].set_xlabel('Longitude')
#    ax[1, 1].set_xlabel('Longitude')
#    ax[0, 0].set_ylabel('Latitude')
#    ax[1, 0].set_ylabel('Latitude')

    # Colorbar
#    cbax = fig.add_axes([0.91, 0.08, 0.012, 0.40])

    # nice ticks
#    tickmax = np.ceil(vmax * 100) / 100
#    ticks = np.arange(-tickmax, tickmax + 0.001, 0.01)

#    cbar = fig.colorbar(
#        cf,
#        cax=cbax,
#        orientation='vertical',
#        ticks=ticks
#    )

#    cbar.set_label(cbar_label)
#    cbar.ax.set_yticklabels([f'{t:.2f}' for t in ticks])
#    cbar.ax.tick_params(direction='out', labelsize=11)

    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.08,
        top=0.95,
        hspace=0.08,
        wspace=0.01
    )

    fig.savefig(fgname, dpi=300, bbox_inches='tight')



#=========================================

plot_4panels(
    eta,
    fgname_eta,
    r'$\eta$ [m]',
    cmap=isw_cmap
)

#plot_4panels(
#    w,
#    fgname_w,
#    r'$w$ [m s$^{-1}$]',
#    cmap=isw_cmap
#)

