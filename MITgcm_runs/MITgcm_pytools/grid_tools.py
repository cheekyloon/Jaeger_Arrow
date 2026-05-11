#!/usr/bin/env python

#import modules
import time
import sys
import scipy.io
import numpy             as np
import matplotlib.pyplot as plt
from scipy               import interpolate
from PyDJL               import DJL, Diagnostic

def write_to_binary(data, fileout, precision='double'):
    ''' write variable from np.array to fileout with precision '''
    # write data to binary files
    fid   = open(fileout, "wb")
    flatdata = data.flatten()
    if precision == 'single':
        if sys.byteorder == 'little':
            tmp = flatdata.astype(np.dtype('f')).byteswap(True).tobytes()
        else:
            tmp = flatdata.astype(np.dtype('f')).tobytes()
    elif precision == 'double':
        if sys.byteorder == 'little':
            tmp = flatdata.astype(np.dtype('d')).byteswap(True).tobytes()
        else:
            tmp = flatdata.astype(np.dtype('d')).tobytes()
    fid.write(tmp)
    fid.close()
    return None

def write_mitgcm_binary(data, fileout, precision='double'):
    """
    Write a NumPy array to a binary file compatible with MITgcm.

    Parameters
    ----------
    data : np.ndarray
        The array to write. Must be ordered as (Nz, Ny, Nx) for 3D data.
    fileout : str
        Output file path.
    precision : str, optional
        Precision of the output data ('single' or 'double').
        Defaults to 'double'.

    Notes
    -----
    MITgcm requires big-endian binary files, and data should be stored in Fortran order.
    """
    flatdata = data.flatten(order='F')  # Fortran-order flattening
    if precision == 'single':
        dtype = np.dtype('>f4')  # big-endian float32
    elif precision == 'double':
        dtype = np.dtype('>f8')  # big-endian float64
    else:
        raise ValueError("Precision must be 'single' or 'double'")

    flatdata.astype(dtype).tofile(fileout)

def trim_grid_to_fit_mpi(dx, dy, nPx=6, nPy=8):
    """
    Trim the horizontal grid (dx, dy) so that its dimensions are multiples
    of the MPI decomposition (nPx, nPy). The trimming is:
      - symmetrical in x (split west/east)
      - from the south in y

    Parameters
    ----------
    dx : ndarray
        1D array of x-direction grid spacing.
    dy : ndarray
        1D array of y-direction grid spacing.
    nPx : int
        Number of MPI processes/tiles in x-direction.
    nPy : int
        Number of MPI processes/tiles in y-direction.

    Returns
    -------
    dx_new : ndarray
        Trimmed x-direction spacing (length divisible by nPx).

    dy_new : ndarray
        Trimmed y-direction spacing (length divisible by nPy).

    nxW : int
        Number of points trimmed from the west.

    nxE : int
        Number of points trimmed from the east.

    diff_ny : int
        Number of points trimmed from the south.
    """

    nx = len(dx)
    ny = len(dy)

    # How many points to remove to make divisible
    diff_nx = nx % nPx
    diff_ny = ny % nPy

    # Split trimming in x-direction: west/east
    nxW = diff_nx // 2
    nxE = diff_nx - nxW

    # Apply trimming
    dx_new = dx[nxW:] if nxE == 0 else dx[nxW:-nxE]
    dy_new = dy[diff_ny:]

    return dx_new, dy_new, nxW, nxE, diff_ny

def generate_vertical_grid(dz0=0.5, max_dz=2.0, growth=1.10, z0=20.0, target_depth=100.0, verbose=True):
    """
    Generate a 1D vertical grid spacing (layer thicknesses) from surface to bottom with:
    - A uniform fine resolution near the surface (up to depth z0),
    - Followed by a gradual geometric increase in layer thickness,
    - Capped at a maximum layer thickness max_dz,
    - Extended with constant max_dz layers until the total depth reaches or exceeds target_depth.

    Parameters
    ----------
    dz0 : float
        Initial layer thickness in meters (fine resolution step near surface).
    max_dz : float
        Maximum allowed thickness of any vertical layer in meters.
    growth : float
        Multiplicative growth factor controlling how quickly layer thickness grows.
    z0 : float
        Depth (in meters) over which layers have uniform thickness dz0.
    target_depth : float
        Minimum total depth (sum of all layer thicknesses) to reach or exceed.
    verbose : bool
        If True, print summary information about generated vertical grid.

    Returns
    -------
    dz : list of float
        List of vertical layer thicknesses (meters) from surface downward.
    """

    # Step 1: Create uniform fine resolution layers of thickness dz0 down to depth z0
    dz = [dz0] * int(z0 / dz0)

    # Step 2: Gradually increase layer thickness geometrically, capped by max_dz
    dz_last = dz[-1]  # start growth from last fine resolution layer thickness
    while True:
        dz_next = min(dz_last * growth, max_dz)  # grow layer thickness, capped at max_dz
        dz.append(dz_next)  # add new layer thickness
        if dz_next >= max_dz:
            break  # stop increasing once maximum thickness is reached
        dz_last = dz_next  # update for next iteration

    # Step 3: Add layers of constant max_dz thickness until total depth reaches target_depth
    while sum(dz) < target_depth:
        dz.append(max_dz)

    # Step 4: Ensure last layer thickness is exactly max_dz (adjust if needed)
    if dz[-1] != max_dz:
        dz[-1] = max_dz

    # Optional verbose output of grid characteristics
    if verbose:
        print(f"Vertical grid generated:")
        print(f" - Total depth: {sum(dz):.2f} m")
        print(f" - Number of layers: {len(dz)}")
        print(f" - Last dz: {dz[-1]} m")

    return dz

def generate_djl_wave_from_CTDF14(
    H,
    L,
    profile_path,
    profile_z_key='zb_F14',
    profile_rho_key='rhob_F14',
    APE=1e5,
    NX_init=20,
    NZ_init=10,
    epsilon_init=1e-3,
    resolutions=None,
    plot_wave=False
):
    """
    Generate a DJL internal solitary wave using the background CTD F14 profile.

    Parameters
    ----------
    H : float
        Total depth of the domain (in meters).
    L : float
        Domain length (in meters).
    profile_path : str
        Path to the .mat file containing the background profile.
    profile_z_key : str
        Variable name for the vertical coordinate in the .mat file.
    profile_rho_key : str
        Variable name for the background density in the .mat file.
    APE : float
        Target available potential energy for the wave (in kg m/s²).
    NX_init : int
        Initial horizontal resolution for DJL solver.
    NZ_init : int
        Initial vertical resolution for DJL solver.
    epsilon_init : float
        Initial solver tolerance.
    resolutions : list of tuple(int, int, float), optional
        List of (NX, NZ, epsilon) triplets for refinement stages.
    plot_wave : bool, optional
        If True, plot diagnostic figures (density, u, w).

    Returns
    -------
    djl : PyDJL.DJL
        Final DJL wave solution.
    diag : PyDJL.Diagnostic
        Diagnostics associated with the wave (density, velocity, etc.).
    """

    if resolutions is None:
        resolutions = [
            (50, 10, 1e-3),
            (100, 20, 1e-4),
            (150, 50, 1e-5),
            (200, 100, 1e-6),
            (300, 150, 1e-7),
            (400, 200, 1e-7)
        ]

    # === Load CTD F14 profile ===
    mat = scipy.io.loadmat(profile_path)
    zF14   = mat[profile_z_key].squeeze()
    rhoF14 = mat[profile_rho_key].squeeze()

    # Keep only depths shallower than H
    mask = zF14 >= -H
    zdata = zF14[mask]
    rhodata = rhoF14[mask]

    # Replace shallow NaNs by the first valid value
    rhodata[0:5] = rhoF14[5]

    # Background density scale (maximum value)
    # Used in DJL module to compute N2(z)
    rho0 = np.max(rhodata)

    # Compute drho/dz numerically
    rhozdata = np.gradient(rhodata, zdata)

    # Create interpolating functions
    rho  = lambda z: interpolate.interp1d(zdata, rhodata, bounds_error=False, fill_value="extrapolate")(z)
    rhoz = lambda z: interpolate.interp1d(zdata, rhozdata, bounds_error=False, fill_value="extrapolate")(z)

    # Background velocity profile (zero everywhere)
    Ubg   = lambda z: 0 * z
    Ubgz  = lambda z: 0 * z
    Ubgzz = lambda z: 0 * z

    # === DJL solver: continuation on APE ===
    print("Building DJL wave...")
    start_time = time.time()

    for i, A in enumerate(np.linspace(1.0e4, APE, 10)):
        if i == 0:
            djl = DJL(A, L, H, NX_init, NZ_init, rho, rhoz, rho0=rho0, epsilon=epsilon_init, verbose=1)
        else:
            djl = DJL(A, L, H, NX_init, NZ_init, rho, rhoz, rho0=rho0, epsilon=epsilon_init, initial_guess=djl, verbose=1)

    # === Refinement stages ===
    for NX, NZ, eps in resolutions:
        djl = DJL(APE, L, H, NX, NZ, rho, rhoz, rho0=rho0, epsilon=eps, initial_guess=djl, verbose=1)

    end_time = time.time()
    print(f"DJL solver completed in {end_time - start_time:.2f} seconds.")

    # === Compute diagnostics ===
    diag = Diagnostic(djl)

    # === Optional plotting ===
    if plot_wave:
        plt.ion()
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        cs1 = axs[0].contourf(djl.XC, djl.ZC, diag.density, levels=50)
        axs[0].set_title("Density")
        axs[0].set_xlabel("x (m)")
        axs[0].set_ylabel("z (m)")
        fig.colorbar(cs1, ax=axs[0])

        cs2 = axs[1].contourf(djl.XC, djl.ZC, diag.u, levels=50)
        axs[1].set_title("Zonal velocity (u)")
        axs[1].set_xlabel("x (m)")
        axs[1].set_ylabel("z (m)")
        fig.colorbar(cs2, ax=axs[1])

        cs3 = axs[2].contourf(djl.XC, djl.ZC, diag.w, levels=50)
        axs[2].set_title("Vertical velocity (w)")
        axs[2].set_xlabel("x (m)")
        axs[2].set_ylabel("z (m)")
        fig.colorbar(cs3, ax=axs[2])

        plt.tight_layout()
        plt.show()

    return djl, diag

