#!/Users/sandy/miniconda3/bin/python

"""
Identify and separate neap- and spring-tide periods from RBR pressure records.

This script:
1. Loads pressure and time series previously saved in a .mat file.
2. Converts Matlab datenums into Python datetime format.
3. Separates low- and high-amplitude tidal periods corresponding to
   neap and spring tides.
4. Plots the complete pressure record together with:
      - neap-tide periods in the upper panel,
      - spring-tide periods in the lower panel.

The figure is used to visually validate the tidal classification method
based on the local tidal amplitude estimated from the Hilbert transform.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
from scipy.io import loadmat
plt.ion()

dir_mat = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/Instruments/RBR/mat/'
year = '2023'

data = loadmat(dir_mat + 'T_i_13m_' + year + '.mat')

time        = data['time'].squeeze()
time_neap   = data['time_neap'].squeeze()
time_spring = data['time_spring'].squeeze()

h        = data['pressure'].squeeze()
h_neap   = data['pressure_neap'].squeeze()
h_spring = data['pressure_spring'].squeeze()


# Matlab datenum -> pandas datetime
time_py = pd.to_datetime(time.squeeze() - 719529, unit='D')
time_neap_py = pd.to_datetime(time_neap.squeeze() - 719529, unit='D')
time_spring_py = pd.to_datetime(time_spring.squeeze() - 719529, unit='D')

# Figure
fig, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

# --- Neap tides ---
ax[0].plot(time_py, h, color='k', linewidth=0.8, label='Full series')
ax[0].plot(time_neap_py, h_neap, color='b', linewidth=1.0, label='Neap')

ax[0].set_ylabel("Pressure / water level (m)")
ax[0].set_title("Neap tides")
ax[0].legend()

# --- Spring tides ---
ax[1].plot(time_py, h, color='k', linewidth=0.8, label='Full series')
ax[1].plot(time_spring_py, h_spring, color='r', linewidth=1.0, label='Spring')

ax[1].set_ylabel("Pressure / water level (m)")
ax[1].set_xlabel("Time")
ax[1].set_title("Spring tides")
ax[1].legend()

# Time formatting
ax[1].xaxis.set_major_formatter(md.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()

fig.suptitle(f"Year {year}", fontsize=16)

plt.tight_layout()
