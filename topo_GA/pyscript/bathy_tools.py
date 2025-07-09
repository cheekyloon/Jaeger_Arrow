import numpy             as np
import re
import pandas            as pd
import matplotlib.pyplot as plt
from scipy.spatial       import cKDTree
from matplotlib.path     import Path
from pyproj              import Transformer
from scipy.interpolate   import griddata

# interactive mode
plt.ion()

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

def convert_mtm_to_latlon(x, y):
    """
    Convert MTM zone 7 (x, y in meters) to WGS84 coordinates (longitude, latitude).

    Parameters:
    - x: float or array-like (MTM Easting)
    - y: float or array-like (MTM Northing)

    Returns:
    - lon: longitude in decimal degrees
    - lat: latitude in decimal degrees
    """
    transformer = Transformer.from_crs("EPSG:2949", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lon, lat

def dms_to_decimal(dms_str):
    """
    Convert a DMS (Degrees-Minutes-Seconds) coordinate string to decimal degrees.

    Parameters
    ----------
    dms_str : str
        Coordinate string in the format 'DD-MM-SS.SSS[N|S|E|W]'

    Returns
    -------
    float
        Decimal degree representation of the coordinate.
    """
    match = re.match(r"(\d+)-(\d+)-([\d.]+)([NSEW])", dms_str.strip())
    if not match:
        raise ValueError(f"Invalid DMS format: {dms_str}")
    deg, min_, sec, direction = match.groups()
    decimal = int(deg) + int(min_) / 60 + float(sec) / 3600
    if direction in ('S', 'W'):
        decimal *= -1
    return decimal

def load_lr_bathy_from_txt(filepath, convert_latlon_to_mtm_func):
    """
    Load low-resolution bathymetry from a text file with DMS coordinates.

    Parameters
    ----------
    filepath : str
        Path to the text file containing the bathymetry.
    convert_latlon_to_mtm_func : function
        Function that converts (lon, lat) in decimal degrees to (X, Y) in meters
        in the MTM Zone 7 projection.

    Returns
    -------
    X : ndarray
        Projected x-coordinates (MTM, in meters).
    Y : ndarray
        Projected y-coordinates (MTM, in meters).
    Z : ndarray
        Depth values (in meters).
    """
    # Load the file (tab-separated)
    df = pd.read_csv(filepath, sep='\t')

    # Convert DMS strings to decimal degrees
    df['lat'] = df['Lat (DMS)'].apply(dms_to_decimal)
    df['lon'] = df['Long (DMS)'].apply(dms_to_decimal)

    # Extract variables
    LAT = df['lat'].values
    LON = df['lon'].values
    Z   = df['Depth (m)'].values

    # Convert to projected coordinates
    X, Y = convert_latlon_to_mtm_func(LON, LAT)

    return X, Y, Z

def build_variable_resolution(center_start, center_end, full_start, full_end,
                                        d_fine=0.5, d_max=2.0, growth=1.1):
    """
    Create a 1D axis from full_start to full_end with:
    - uniform fine resolution between center_start and center_end
    - geometrically increasing resolution toward the domain edges until d_max

    Parameters:
    - center_start : float, start of fine-resolution zone
    - center_end   : float, end of fine-resolution zone
    - full_start   : float, full domain start (min x or y)
    - full_end     : float, full domain end (max x or y)
    - d_fine       : float, resolution in the fine zone (default = 0.5 m)
    - d_max        : float, max resolution beyond the fine zone (default = 2.0 m)
    - growth       : float, geometric growth factor (default = 1.1)

    Returns:
    - axis : 1D NumPy array of coordinates
    """

    # Central zone
    fine_zone = np.arange(center_start, center_end + d_fine, d_fine)

    # Right side
    right = [center_end]
    dx = d_fine
    while right[-1] + dx < full_end:
        dx = min(dx * growth, d_max)
        right.append(right[-1] + dx)

    # Left side
    left = [center_start]
    dx = d_fine
    while left[-1] - dx > full_start:
        dx = min(dx * growth, d_max)
        left.append(left[-1] - dx)

    # Remove duplicate if needed
    left_rev = left[::-1]
    if np.isclose(left_rev[-1], center_start):
        left_rev = left_rev[:-1]

    # Combine
    axis = np.array(left_rev + list(fine_zone) + right[1:])  # skip duplicate at center_end 
    return axis

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

def interpolate_to_grid(points, values, grid_x, grid_y, method='linear', fill_value=np.nan):
    """
    Interpolate scattered data to a regular grid.
    """
    return griddata(points, values, (grid_x, grid_y), method=method, fill_value=fill_value)


def apply_mask(grid_x, grid_y, polygon_coords):
    """
    Create a boolean mask over a grid where True = inside polygon.
    """
    path   = Path(polygon_coords)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    return path.contains_points(points).reshape(grid_x.shape)

def rotate_topography(df_xyz, x1, y1, x2, y2, reso=1.0, method='linear', fill_value=np.nan, origin=None):
    """
    Rotate a bathymetry DataFrame so that a reference line (e.g., a wharf)
    becomes aligned with the x-axis (horizontal), and interpolate the rotated 
    bathymetry onto a regular 2D grid.

    Parameters:
    - df_xyz : DataFrame with columns ['x', 'y', 'z'] in projected coordinates (e.g., MTM)
    - x1, y1 : Coordinates of the first reference point (e.g., west corner of the wharf)
    - x2, y2 : Coordinates of the second reference point (e.g., east corner of the wharf)
    - reso : Grid resolution in meters (default = 1.0)
    - method : Interpolation method used in `griddata` ('linear', 'nearest', or 'cubic')
    - fill_value : Value used to fill grid points without valid data
    - origin : Optional (x, y) tuple or array specifying the center of rotation.
               If None, defaults to the first point (x1, y1)

    Returns:
    - grid_x_rot, grid_y_rot : 2D meshgrid arrays in the rotated coordinate system
    - grid_depth_rot         : Bathymetry interpolated onto the rotated grid
    - theta_rad              : Rotation angle in radians (negative, to align the line horizontally)
    """
    # Compute rotation angle
    theta_rad = compute_rotation_angle(x1, y1, x2, y2) 

    # Define rotation origin
    if origin is None:
        origin = np.array([x1, y1])

    coords     = df_xyz[['x', 'y']].values
    coords_rot = rotate_coordinates(coords, origin, -theta_rad)

    df_xyz['x_rot'] = coords_rot[:, 0]
    df_xyz['y_rot'] = coords_rot[:, 1]

    # Define output grid limits based on rotated coordinates
    xg_rot = np.arange(df_xyz['x_rot'].min(), df_xyz['x_rot'].max() + reso, reso)
    yg_rot = np.arange(df_xyz['y_rot'].min(), df_xyz['y_rot'].max() + reso, reso)
    grid_x_rot, grid_y_rot = np.meshgrid(xg_rot, yg_rot)

    # Interpolate depths on rotated grid
    depth_rot = interpolate_to_grid(
        df_xyz[['x_rot', 'y_rot']].values,
        df_xyz['z'].values,
        grid_x_rot, grid_y_rot,
        method=method, fill_value=fill_value
    )

    return grid_x_rot, grid_y_rot, depth_rot, -theta_rad

def plot_bathy_difference(grid_x, grid_y, z_hr_ga, z_hr_fjord, show_figure=False):
    """
    Calculates the difference between high-resolution (HR) and low-resolution (LR) bathymetry
    within the HR area, then plots the difference map using a diverging colormap centered at zero.

    Parameters
    ----------
    grid_x, grid_y : 2D arrays
        Coordinate grids (meshgrid) used for interpolation.
    z_hr_ga : 2D array
        High-resolution bathymetry interpolated onto the grid.
    z_hr_fjord : 2D array
        Low-resolution bathymetry interpolated onto the grid.
    show_figure : bool, optional
        If True (default), displays the figure.

    Returns
    -------
    diff_topo : 2D array
        Matrix of HR - LR differences within the HR area; NaN elsewhere.
    """

    # Mask of valid HR area
    mask_hr = ~np.isnan(z_hr_ga)

    # Limit LR data to HR area (NaN elsewhere)
    z_lr_masked = np.where(mask_hr, z_hr_fjord, np.nan)

    # Compute difference
    diff_topo = z_hr_ga - z_lr_masked

    if show_figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        vmax = np.nanmax(np.abs(diff_topo))
        c = ax.pcolormesh(grid_x, grid_y, diff_topo, shading='auto',
                          cmap='seismic', vmin=-vmax, vmax=vmax)
        fig.colorbar(c, label="Difference HR - LR (m)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Bathymetry Difference HR - LR (HR zone)")
        ax.set_aspect('equal')
        ax.set_xlim(np.nanmin(grid_x), np.nanmax(grid_x))
        ax.set_ylim(np.nanmin(grid_y), np.nanmax(grid_y))
        fig.tight_layout()

    return diff_topo

def compute_hr_lr_offset_from_contour(
    grid_x, grid_y,
    z_hr_ga, z_hr_fjord,
    x_hr, y_hr,
    distance_tol=1.0,
    show_figure=False):
    """
    Computes the average offset between high-resolution (HR) and low-resolution (LR)
    bathymetries along the HR contour line, to correct vertical shifts between datasets.

    Parameters
    ----------
    grid_x, grid_y : 2D arrays
        Coordinate grids (meshgrid) used for interpolation.
    z_hr_ga : 2D array
        High-resolution bathymetry interpolated on the grid.
    z_hr_fjord : 2D array
        Low-resolution bathymetry interpolated on the grid.
    x_hr, y_hr : 1D arrays
        Grid axis coordinates (order: x and y).
    distance_tol : float, optional
        Maximum distance (in meters) to consider points "close to the contour".
    show_figure : bool, optional
        If True, displays a plot of the HR contour.

    Returns
    -------
    offset : float
        The average HR - LR offset computed along the contour.
    """

    # 1. Mask valid HR points
    mask_hr = ~np.isnan(z_hr_ga)

    # 2. Extract the contour line of the HR valid area
    contours = plt.contour(grid_x, grid_y, mask_hr.astype(float), levels=[0.5])
    paths = contours.collections[0].get_paths()
    contour_coords = max(paths, key=lambda p: p.vertices.shape[0]).vertices

    if show_figure:
        fig, ax = plt.subplots(figsize=(6, 6))
        c = ax.contourf(grid_x, grid_y, z_hr_ga, levels=50, cmap='viridis')
        fig.colorbar(c, label="Depth (m)")
        ax.plot(contour_coords[:, 0], contour_coords[:, 1], 'r-', lw=2, label='HR Contour')
        ax.set_aspect('equal')
        ax.set_title('HR Zone Contour')
        ax.legend()
        fig.tight_layout()

    # 3. Extract HR grid points
    points_hr = np.column_stack((grid_x[mask_hr], grid_y[mask_hr]))

    # 4. Compute distance from HR points to contour line
    tree = cKDTree(contour_coords)
    distances, _ = tree.query(points_hr)

    # 5. Select points close to the contour within tolerance
    mask_close = distances <= distance_tol
    points_near_contour = points_hr[mask_close]

    # 6. Find corresponding grid indices for these points
    final_mask = np.zeros_like(z_hr_ga, dtype=bool)
    for x, y in points_near_contour:
        ix = np.argmin(np.abs(x_hr - x))
        iy = np.argmin(np.abs(y_hr - y))
        final_mask[iy, ix] = True

    # 7. Calculate offset between HR and LR at selected points
    valid = final_mask & ~np.isnan(z_hr_ga) & ~np.isnan(z_hr_fjord)
    if np.count_nonzero(valid) == 0:
        raise ValueError("No valid points found within the contour band.")

    offset = np.nanmean(z_hr_ga[valid] - z_hr_fjord[valid])
    print(f"Average HR - LR offset along contour: {offset:.3f} m")

    return offset


