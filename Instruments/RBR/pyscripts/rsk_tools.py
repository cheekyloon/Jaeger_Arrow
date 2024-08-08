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
