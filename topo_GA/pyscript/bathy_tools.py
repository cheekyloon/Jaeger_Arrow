import numpy           as np
import pandas          as pd
from matplotlib.path   import Path
from pyproj            import Transformer
from scipy.interpolate import griddata

def rotate_coordinates(points, origin, theta_deg):
    """
    Rotate a set of 2D points (Nx2) around an origin by a given angle in degrees.
    """
    theta_rad = np.radians(theta_deg)
    R = np.array([
        [np.cos(-theta_rad), -np.sin(-theta_rad)],
        [np.sin(-theta_rad),  np.cos(-theta_rad)]
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
    path = Path(polygon_coords)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    return path.contains_points(points).reshape(grid_x.shape)

def rotate_topography(df_xyz, x1, y1, x2, y2, reso=1.0, method='linear', fill_value=np.nan, origin=None):
    """
    Rotate a bathymetry DataFrame so that the line (x1, y1)-(x2, y2) becomes horizontal.
    Allows custom rotation origin (e.g., center of the wharf).

    Parameters:
    - df_xyz : DataFrame with columns ['x', 'y', 'z']
    - x1, y1, x2, y2 : points defining the dock line
    - reso : resolution of the output grid (default 1.0 m)
    - method : interpolation method for griddata
    - fill_value : value assigned to missing points in interpolation
    - origin : optional (x, y) point used as rotation center. If None, defaults to (x1, y1)

    Returns:
    - grid_x_rot, grid_y_rot : 2D meshgrid of rotated coordinates
    - grid_depth_rot : interpolated depth field
    - theta_deg : applied rotation angle (degrees)
    - df_xyz[['x_rot', 'y_rot']] : DataFrame with rotated coordinates
    """
    dx = x2 - x1
    dy = y2 - y1
    theta_rad = np.arctan2(dy, dx)
    theta_deg = np.degrees(theta_rad)

    # Define rotation origin
    if origin is None:
        origin = np.array([x1, y1])

    coords = df_xyz[['x', 'y']].values
    coords_rot = rotate_coordinates(coords, origin, theta_deg)

    df_xyz['x_rot'] = coords_rot[:, 0]
    df_xyz['y_rot'] = coords_rot[:, 1]

    # Define output grid limits based on rotated coordinates
    xg_rot = np.arange(df_xyz['x_rot'].min(), df_xyz['x_rot'].max() + reso, reso)
    yg_rot = np.arange(df_xyz['y_rot'].min(), df_xyz['y_rot'].max() + reso, reso)
    grid_x_rot, grid_y_rot = np.meshgrid(xg_rot, yg_rot)

    # Interpolate depths on rotated grid
    grid_depth_rot = interpolate_to_grid(
        df_xyz[['x_rot', 'y_rot']].values,
        df_xyz['z'].values,
        grid_x_rot, grid_y_rot,
        method=method, fill_value=fill_value
    )

    return grid_x_rot, grid_y_rot, grid_depth_rot, theta_deg, df_xyz[['x_rot', 'y_rot']]

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

