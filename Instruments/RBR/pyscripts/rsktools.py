#!/Users/sandy/miniconda3/bin/python

# functions for RSK

###################################
# import modules
import pandas     as pd
from   pyrsktools import RSK

def winmean_rsk_data(dirRSK, file, variable, window_size=5, threshold=0):
    """
    Open RSK file data, converts it into a DataFrame, and applies a rolling mean.

    Parameters:
    -----------
    dirRSK      : str
        The directory path where the RSK file is located.
    file        : str
        The name of the RSK file to be processed.
    variable    : str
        The key to extract the variabel data from the RSK object. 
    window_size : int, optional
        The size of the window for calculating the rolling mean. Default is 5.
    threshold   : int, optional
        The threshold to add to the variable, if needed ( for pressure ). 

    Returns:
    --------
    pd.DataFrame
        A DataFrame with the variable and datetime as the index, with the rolling mean applied.

    Example:
    --------
    dirRSK      = "/path/to/directory/"
    file        = "your_rsk_file.rsk"
    variable    = "temperature"
    window_size = 5
    output_df   = winmean_rsk_data(dirRSK, file, variable, window_size)
    """
    # Initialize the RSK object with the file path
    rsk  = RSK(dirRSK + file)
    
    # Open class object
    rsk.open()
    
    # Read data
    rsk.readdata()
    
    # Load specified variable data
    var  = rsk.data[variable] + threshold
    
    # Load timestamp
    time = rsk.data["timestamp"]
    
    # Close RSK class object
    rsk.close()
    
    # Convert array into a DataFrame
    df   = pd.DataFrame({variable: var})
    
    # Add datetime and convert to nanosecond precision
    df['datetime'] = pd.to_datetime(time).astype('datetime64[ns]')
    
    # Set the datetime column as the index
    df.set_index('datetime', inplace=True)
    
    # Get the mean over a rolling window
    dfm  = df.rolling(window_size, center=True).mean()
    
    return dfm

def load_rsk_data(dirRSK, file, variable, t0, tend, threshold=0):
    """
    Load and filter RSK data for a given variable (e.g., pressure or temperature).

    Parameters:
    -----------
    dirRSK      : str
        The directory path where the RSK file is located.
    file        : str
        The name of the RSK file to be processed.
    variable    : str
        The key to extract the variable data from the RSK object.
    t0          : pd.Timestamp
        The start time for filtering the data.
    tend        : pd.Timestamp
        The end time for filtering the data.
    threshold   : float, optional (default=0)
        A value to be added to the extracted data (useful for pressure correction).

    Returns:
    --------
    pd.DataFrame
        A DataFrame with a datetime index and a column for the requested variable.
    """
    # Initialize and read RSK data
    rsk = RSK(dirRSK + file)
    rsk.open()
    rsk.readdata()

    # Create DataFrame with timestamp as index
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(rsk.data['timestamp']),
        variable: rsk.data[variable] + threshold
    }).set_index('timestamp')

    # Apply time filtering
    df_filtered = df.loc[t0:tend]

    return df_filtered


def mask_above_surf(dirRSK, fileRSK, zSolo, pressure_data, t0, tend):
    """
    Masked temperature and depths above the free surface based on pressure data.

    Parameters:
    -----------
    fileRSK      : list of str
        List of RSK file names to be processed.
    dirRSK       : str
        Directory path where the RSK files are located.
    zSolo        : list or np.ndarray
        Array of reference depths corresponding to each file in `fileRSK`.
    pressure_data: np.ndarray
        Array of pressure values used to mask temperature and depth values where depth exceeds pressure.
    t0           : datetime-like
        Start time for filtering the RSK data.
    tend         : datetime-like
        End time for filtering the RSK data.

    Returns:
    --------
    np.ndarray for temperature and depth with shape (depth, time)
    """
    # Initialize
    all_temp   = [] 
    all_depths = []
    for nn, file in enumerate(fileRSK):
        # Load temperature data as a DataFrame
        df = load_rsk_data(dirRSK, file, 'temperature', t0, tend)

        # Extract temperature values
        temp_data = df['temperature'].values
        # Initialize depth array at the reference depth for the current file
        depth = zSolo[nn] * np.ones_like(temp_data)
        
        # Mask temperature where depth > pressure level
        mask_z = depth > pressure_data
        # Assign NaN to masked temperature values
        temp_data[mask_z] = np.nan
        # Assign Nan to masked depth values
        depth[mask_z] = np.nan
        # Append data 
        all_temp.append(temp_data) 
        all_depths.append(depth)
    
    # Convert to 2D arrays
    # Shape: (depth, time)
    return np.array(all_temp), np.array(all_depths)


def compute_dz(all_depths):
    """
    Compute dz (depth differences) along the time series.

    Parameters:
    -----------
    all_depths: np.ndarray 
        Array of depth values with shape (depth, time)

    Returns:
    --------
    np.ndarray for depth difference with shape (depth, time)
    """
    # Initialize with NaNs
    dz = np.ones_like(all_depths) * np.nan
    for tt in range(all_depths.shape[1]):
        # Depths at time tt
        depths_t = all_depths[:, tt]
        # Find valid depths 
        valid_ind = np.where(~np.isnan(depths_t))[0] 
        
        if valid_ind.size > 2:
            # Forward difference for the first valid depth
            dz[valid_ind[0], tt] = depths_t[valid_ind[0]] - (np.floor(depths_t[valid_ind[0]]) + depths_t[valid_ind[1]]) / 2
            # Backward difference for the last valid depth
            dz[valid_ind[-1], tt] = 2
            # Compute centered differences
            dz[valid_ind[1], tt] = (np.floor(depths_t[valid_ind[0]]) - depths_t[valid_ind[2]]) / 2
            if valid_ind.size > 3:
                dz[valid_ind[2:-1], tt] = (depths_t[valid_ind[1:-2]] - depths_t[valid_ind[3:]]) / 2
        elif valid_ind.size == 2:
            # Forward difference for the first valid depth
            dz[valid_ind[0], tt] = depths_t[valid_ind[0]] - 2
            # Backward difference for the last valid depth
            dz[valid_ind[1], tt] = 2
        elif valid_ind.size == 1:
            # Assign a default value, i.e., the depth itself
            dz[valid_ind[0], tt] = depths_t[valid_ind[0]]
     
    return dz

def compute_variance_depth(all_temp, dz):
    """
    Compute the variance of the temperature along the depth for each time.
    Parameters:
    -----------
    all_temp: np.ndarray 
        Array of temperature with shape (depth, time)
    dz      : np.ndarray
        Array of depth difference with shape (depth, time) used for the weighted average 

    Returns:
    --------
    np.ndarray for the temperature variance 
    """
    variance_temp = np.ones(all_temp.shape[1]) * np.nan
    for tt in range(all_temp.shape[1]):
        # Temperature at time tt
        temp_t = all_temp[:, tt]
        # Depth difference at time tt
        dz_t = dz[:, tt]
        # Compute depth-weighted average temperature
        weighted_avg_temp = np.nansum(temp_t * dz_t) / np.nansum(dz_t)
        # Compute temperature variance
        variance_temp[tt] = np.nansum(((temp_t - weighted_avg_temp) ** 2) * dz_t) / np.nansum(dz_t)

    return variance_temp

