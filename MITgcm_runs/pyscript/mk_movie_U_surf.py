#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate an animation of surface speed (0–10 m)
Optimized version: no redraw, lazy loading, fast render.
"""

import mitgcm_tools
import os
import numpy                as np
import matplotlib.pyplot    as plt
import matplotlib.animation as animation
import xarray               as xr
import pandas               as pd
from scipy.io               import loadmat
from matplotlib             import colors
from pyproj                 import Transformer
# =========================================
# --- USER CONFIG ---
GA_dir   = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/topo_GA/Data/'
exp      = '45Deg_4ISW_sponge_extent_right'
outpath  = '/Volumes/LaCie/JaegerArrow/movie/'
datapath = f'/Volumes/LaCie/JaegerArrow/MITgcm_runs/{exp}/'
framepath = f'{datapath}speed_frames/'
movie    = f'{outpath}{exp}_speed_surf.mov'

fgrid  = f'{datapath}mnc_glob/grid.glob.nc'

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

# =========================================
# --- LOAD GRID & STATE ---
grid, xgrid = mitgcm_tools.loadgrid(fgrid, basin_masks=False)
grid.close()

n_files = len([f for f in os.listdir(framepath) if f.startswith("speed_")])
nmax = n_files

print(f"Detected {nmax} frames total")


# --- Mask and grid ---
mask = grid.HFacC.isel(ZC=0).where(grid.HFacC.isel(ZC=0) > 0)
land = np.isnan(mask)
x = grid.XC
y = grid.YC

# =========================================
# --- VISUAL SETTINGS ---
topo_color = '#ffeabc'
cmap_name  = 'RdYlBu_r'
ulev = np.arange(0, 1.35, 0.05)

# Extract a small portion of the image to use as a background texture 
texture = img[900:1100, 200:400].copy()
# Convert RGB image to grayscale
texture_gray = texture.mean(axis=2)
# Reduce brightness to obtain a subtle background texture
texture_gray *= 0.25
# Add Gaussian noise to mimic paper grain
noise = 0.10 * np.random.randn(*texture_gray.shape)
texture_gray = np.clip(texture_gray + noise, 0, 1)

plt.rcParams.update({
    'axes.titlepad': 6,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'font.size': 10
})

# convert kn to m/s
kn2ms = 1852/3600
# convert m/s to kn
ms2kn = 1/kn2ms

# --- Initial speed field ---
raw_speed = np.load(f'{framepath}speed_0000.npy')
speed = ms2kn * mask * raw_speed

# =========================================
# --- FIGURE SETUP ---
fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_aspect('equal')

# Google Earth image
ax1.pcolormesh(
    x_img_rot + dx_img,
    y_img_rot + dy_img,
    img_gray,
    shading='auto',
    cmap='gray',
    zorder=0
)

ax1.imshow(
    texture_gray,
    extent=[
        grid.XC.min(),
        grid.XC.max(),
        grid.YC.min(),
        grid.YC.max()
    ],
    cmap='gray',
    aspect='equal',
    alpha=0.5,
    zorder=-10
)

#norm = colors.Normalize(vmin=ulev[0], vmax=ulev[-1], clip=True)

# Land and initial contour
#ax1.contourf(grid.XC, grid.YC, land, levels=[0.5, 1.5],
#             colors=[topo_color], zorder=4)
cont = ax1.contourf(grid.XC, grid.YC, speed, cmap=cmap_name, levels=ulev, extend='both', zorder=5)

# Fixed colorbar
cbax = fig.add_axes([0.14, 0.9, 0.35, 0.015])
cbar = fig.colorbar(cont, cax=cbax, orientation="horizontal", ticks=ulev[::2])

# Force ticks and labels on top
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
#cbar.ax.tick_params(axis='x', direction='out', labeltop=True, labelbottom=False, top=True, bottom=False)

# Ensure top spine is visible (and bottom hidden)
#for spine in cbar.ax.spines.values():
#    spine.set_visible(False)
#cbar.ax.spines['top'].set_visible(True)

# Label
cbar.set_label(r'$|U|$ (kn)', fontsize=9)
cbar.ax.tick_params(labelsize=7)


# =========================================
cont_obj = [cont]
# --- ANIMATION FUNCTION ---
def animate(i):
    ufile = f'{framepath}speed_{i:04d}.npy'
    raw_speed = np.load(ufile)
    speed = ms2kn * mask * raw_speed
    speed = np.clip(speed, ulev[0], ulev[-1])
    # Supprime le contour précédent s’il existe encore
    old = cont_obj[0]
    try:
        old.remove()  # depuis Matplotlib 3.8, cette méthode est sûre
    except Exception:
        pass

    # Crée un nouveau contour et le sauvegarde
    new_cont = ax1.contourf(
        grid.XC, grid.YC,
        speed,
        cmap=cmap_name, levels=ulev, extend='neither', zorder=5 
    )
    cont_obj[0] = new_cont

#    # Terre (fixe)
#    ax1.contourf(grid.XC, grid.YC, land, levels=[0.5, 1.5],
#                 colors=[topo_color], zorder=4)

    # Google Earth image
    ax1.pcolormesh(
        x_img_rot + dx_img,
        y_img_rot + dy_img,
        img_gray,
        shading='auto',
        cmap='gray',
        zorder=0
    )

    ax1.imshow(
        texture_gray,
        extent=[
            grid.XC.min(),
            grid.XC.max(),
            grid.YC.min(),
            grid.YC.max()
        ],
        cmap='gray',
        aspect='equal',
        alpha=0.5,
        zorder=-10
    )

    return new_cont.collections

# =========================================
# --- CREATE MOVIE ---
writer = animation.FFMpegWriter(fps=15)
anim = animation.FuncAnimation(fig, animate, frames=nmax, blit=False)
anim.save(movie, dpi=150, writer=writer)

print(f"\nAnimation saved to {movie}")

