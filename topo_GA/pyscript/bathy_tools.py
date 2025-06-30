import numpy           as np
import pandas          as pd
from matplotlib.path   import Path
from pyproj            import Transformer
from scipy.interpolate import griddata

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

