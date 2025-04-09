#!/Users/sandy/miniconda3/bin/python

# functions for RSK

###################################
# import modules
import pandas          as pd
import numpy           as np
import rsktools        as rsk
from scipy             import signal
from pyrsktools        import RSK
from sklearn.neighbors import NearestNeighbors

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


def compute_dz(all_depths, pressure_data):
    """
    Compute dz (depth differences) along the time series.

    Parameters:
    -----------
    all_depths: np.ndarray 
        Array of depth values with shape (depth, time)
    pressure_data: np.ndarray
        Array of pressure values used to mask temperature and depth values where depth exceeds pressure.

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
            dz[valid_ind[0], tt] = pressure_data.iloc[tt] - (depths_t[valid_ind[0]] + depths_t[valid_ind[1]]) / 2
            # Backward difference for the last valid depth
            dz[valid_ind[-1], tt] = 2
            # Compute centered differences
            dz[valid_ind[1:-1], tt] = (depths_t[valid_ind[:-2]] - depths_t[valid_ind[2:]]) / 2
        elif valid_ind.size == 2:
            # Forward difference for the first valid depth
            dz[valid_ind[0], tt] = pressure_data.iloc[tt] - 2
            # Backward difference for the last valid depth
            dz[valid_ind[1], tt] = 2
        elif valid_ind.size == 1:
            # Assign the depth itself
            dz[valid_ind[0], tt] = pressure_data.iloc[tt] 
     
    return dz

def weight_avg_z(var, dz):
    """
    Compute the weighted average of a variable, such as the temperature, along the depth for each time.
    Parameters:
    -----------
    var: np.ndarray 
        Array with shape (depth, time)
    dz : np.ndarray
        Array of depth difference with shape (depth, time) used for the weighted average 

    Returns:
    --------
    np.ndarray for the weighted-depth variable
    """
    avg_z = np.ones(var.shape[1]) * np.nan
    for tt in range(var.shape[1]):
        # Variable at time tt
        var_t = var[:, tt]
        # Depth difference at time tt
        dz_t  = dz[:, tt]
        # Compute depth-weighted average 
        avg_z[tt] = np.nansum(var_t * dz_t) / np.nansum(dz_t)

    return avg_z

def bp_filter(var, timestamps, Fc_low, Fc_high, N_but):
    """
    Applies a bandpass Butterworth filter to an array (time-dependent or depth x time).
    
    Parameters:
    - var        : np.ndarray, shape (time,) or (depth, time)
    - timestamps : pd.DatetimeIndex corresponding to the time axis
    - Fc_low     : Lower cutoff frequency in Hz 
    - Fc_high    : Higher cutoff frequency in Hz 
    - N_but      : Order of the Butterworth filter 

    Returns:
    - var_fbp    : np.ndarray of filtered data (same shape as input)
    """
    # Calculate sampling interval and frequency
    dt = (timestamps[1] - timestamps[0]).total_seconds()
    Fs = 1 / dt
    # Nyquist frequency
    Fn = Fs / 2 

    # Normalize cutoff frequencies
    Wn = [Fc_low / Fn, Fc_high / Fn]

    # Design Butterworth bandpass filter
    b, a = signal.butter(N_but, Wn, btype='band')

    # If 1D: apply filter directly
    if var.ndim == 1:
        return signal.filtfilt(b, a, var)
    # If 2D: apply filter along time axis
    elif var.ndim == 2:
        return signal.filtfilt(b, a, var, axis=1)
    else:
        raise ValueError("Input var array must be 1D or 2D (depth x time).")

def M2_phase_avg(df_pressure, y, N_bin):
    """
    This function computes phase averages over an M2 semi-diurnal tidal cycle.

    INPUT:
        df_pressure: A DataFrame with a 'pressure' column and a datetime index.
        y          : Variable to phase-average. Can be 1D (time,) or 2D (depth, time)
        N_bin      : The number of averaging bins

    OUTPUT:
        phi_bin: Phase of each bin center (length = N_bin)
        y_bin  : Phase-averaged values (shape = (N_bin,) if 1D input, or (depth, N_bin) if 2D)
    """
    # Tidal frequency (M2 in radians/second)
    w = 2 * np.pi / (3600 * 12.4206012025189)

    # Remove mean from pressure
    h = df_pressure['pressure'].values - df_pressure['pressure'].mean()

    # Time in seconds
    t = (df_pressure.index - df_pressure.index[0]).total_seconds().to_numpy()
    # Get the number of data points
    N = len(t)

    # Harmonic fit to get tidal phase
    # Matrices for harmonic analysis for a signal of a single frequency (w)
    A = np.vstack([np.ones(N), np.sin(w * t), np.cos(w * t)]).T
    # Solve the system by least squares (coef provides mean and amplitudes)
    coef = np.linalg.lstsq(A, h, rcond=None)[0]
    # The phase shift (phi0) is what we need from the coefficients
    phi0 = np.arctan2(coef[2], coef[1])

    # Wrapped phase from -pi to pi
    phi_wrap = np.arctan2(np.sin(w * t + phi0), np.cos(w * t + phi0))

    # Define bins
    dphi = 2 * np.pi / N_bin
    phi_bin_edges = np.arange(-np.pi, np.pi + dphi, dphi)
    phi_bin = phi_bin_edges[:-1] + dphi / 2

    # Determine if y is 1D (time) or 2D (depth, time)
    is_1D = y.ndim == 1

    # Initialize output
    if is_1D:
        y_bin = np.full(N_bin, np.nan)
    else:
        depth = y.shape[0]
        y_bin = np.full((depth, N_bin), np.nan)

    # Compute mean per bin
    for i in range(N_bin):
        indices = np.where((phi_wrap >= phi_bin_edges[i]) & (phi_wrap < phi_bin_edges[i+1]))[0]
        if len(indices) > 0:
            if is_1D:
                y_bin[i] = np.nanmean(y[indices])
            else:
                y_bin[:, i] = np.nanmean(y[:, indices], axis=1)

    return phi_bin, y_bin

def compute_phitime(df_pressure):
    """
    Computes the phase-time matrix (phitime) based on pressure variations.
    
    Parameters:
    df_pressure (pd.DataFrame): A DataFrame with a 'pressure' column and a datetime index.

    Returns:
    np.ndarray: A 2D array containing the interpolated phase times.
    """
    
    # Remove mean from pressure
    h = df_pressure['pressure'].values - df_pressure['pressure'].mean()

    # Extract timestamps and convert into seconds relative to the first timestamp, then into days
    timestamps = (df_pressure.index - df_pressure.index[0]).total_seconds().to_numpy() / (24 * 3600)

    # M2 frequency (in days)
    M2 = 12.4206012025189 / 24

    # Compute the phase using the equation (1) from Richards_et_al_2013
    # doi:10.1029/2012JC008154
    # numerator
    numerator = (2 * np.pi / M2) * h
    # denominator
    # the first index in t_padded is -timestamps[1] instead of
    # zero to avoid a division zero by zero. The minus sign
    # is consistent with having a positive value in np.diff(t_padded) 
    t_padded    = np.insert(timestamps, 0, -timestamps[1])
    # extrapolate a value for the first index of h_padded to avoid
    # a jump in np.diff(h_padded) 
    h_padded    = np.insert(h, 0, h[0] - (h[1] - h[0]))
    denominator = np.diff(h_padded) / np.diff(t_padded)
    # phi 
    phi         = np.arctan2(numerator, denominator)

    # Detect ebb transitions (large jumps in phi > 3)
    ebb = np.where(np.abs(np.diff(phi)) > 3)[0]

    # Define phase grid 
    pphi = np.arange(-np.pi, np.pi + np.pi/2, np.pi/2)
    iphi = np.arange(-np.pi, np.pi + np.pi/25, np.pi/25)

    # Initialize phitime matrix of shape:
    # nb of detected ebb segments x nb phase bins 
    phitime = np.full((len(ebb) - 1, len(iphi)), np.nan)
    # Loop through each tidal cycle

    for i in range(len(ebb) - 1):
        # Extracts time and phase data between two consecutive ebb points 
        mcut = timestamps[ebb[i]:ebb[i + 1]]
        pcut = phi[ebb[i]:ebb[i + 1]]
        # Finds the indices where phase values are closest to -pi/2, 0, and pi/2 
        ilw = np.argmin(np.abs(pcut - (-np.pi/2)))
        ifl = np.argmin(np.abs(pcut - 0))
        ihw = np.argmin(np.abs(pcut - (np.pi/2)))
        # Extracts the corresponding timestamps for these phases 
        ptime = [mcut[0], mcut[ilw], mcut[ifl], mcut[ihw], mcut[-1]]
        # Interpolates timestamps onto the finer phase grid iphi 
        phitime[i, :] = np.interp(iphi, pphi, ptime)

    return phitime

def compute_phase_averaged(df_pressure, std_temp):
    """
    Computes phase-averaged sea level and depth-weighted temperature std 
    using interpolation for more accurate matching. 

    Parameters:
    df_pressure (pd.DataFrame): A DataFrame with a 'pressure' column and a datetime index.
    std_temp (np.ndarray)     : An array representing a standard temperature-related variable.

    Returns:
    tuple: (std_temp_mean, h_interp), both as 1D arrays of length len(iphi).
    """
    
    # Compute the phase-time matrix
    phitime = rsk.compute_phitime(df_pressure)
    
    # Define phase bins
    iphi = np.arange(-np.pi, np.pi + np.pi/25, np.pi/25)

    # Convert timestamps to days
    timestamps = (df_pressure.index - df_pressure.index[0]).total_seconds().to_numpy() / (24 * 3600)
    # Ensure it's a 1D array
    timestamps = np.asarray(timestamps).flatten() 
    
    # Remove mean from pressure
    h = df_pressure['pressure'].values - df_pressure['pressure'].mean()

    # Initialize output arrays
    h_phase        = np.full(len(iphi), np.nan)
    std_temp_phase = np.full(len(iphi), np.nan)

    # Initialize NearestNeighbors model
    knn = NearestNeighbors(n_neighbors=1, algorithm='auto')
    # Fit on timestamps (ensure it's a 2D array)
    knn.fit(timestamps.reshape(-1, 1)) 

    for i in range(len(iphi)):

        # Find the closest index in timestamps to phitime[:, i]
        idk = knn.kneighbors(phitime[:, i].reshape(-1, 1), return_distance=False)

        # Store the mean at the index found by knnsearch (adjust indexing to 0-based)
        # idk.flatten() ensures we get a 1D array of indices
        h_phase[i]        = np.nanmean(h[idk.flatten()])  
        std_temp_phase[i] = np.nanmean(std_temp[idk.flatten()])  

    return h_phase, std_temp_phase
