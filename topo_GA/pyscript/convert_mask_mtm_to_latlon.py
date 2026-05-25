import pandas as pd
from bathy_tools import convert_mtm_to_latlon

rdir     = '/Users/sandy/Documents/ISW_projects/Jaeger_Arrow/topo_GA/Data/'

# Input / output
fmask_in  = rdir + "Grande-Anse_mask.dat"
fmask_out = rdir + "Grande-Anse_mask_latlon.dat"

# Lire le fichier MTM
mask_df = pd.read_csv(
    fmask_in,
    sep=r"\s+",
    header=None,
    names=["x", "y", "val"]
)

# Conversion MTM zone 7 -> lon/lat
lon, lat = convert_mtm_to_latlon(
    mask_df["x"].values,
    mask_df["y"].values
)

# Nouveau dataframe en lon/lat
mask_ll = pd.DataFrame({
    "lon": lon,
    "lat": lat,
    "val": mask_df["val"].values
})

# Écriture du nouveau .dat
mask_ll.to_csv(
    fmask_out,
    sep=" ",
    header=False,
    index=False,
    float_format="%.8f"
)
