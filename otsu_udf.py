import numpy as np
import xarray as xr
from skimage.filters import threshold_otsu
from openeo.udf import XarrayDataCube

def apply_datacube(cube: XarrayDataCube, context: dict = None) -> XarrayDataCube:
    data_array = cube.array

    # Flatten data and remove nans
    values = data_array.values.flatten()
    values = values[~np.isnan(values)]

    if len(values) == 0:
        threshold = 0.9 
    else:
        threshold = threshold_otsu(values)

    # Binary classification nodata = nodata, 1 = ice, 2 = snow
    mask = xr.where(data_array.isnull(), np.nan, xr.where(data_array < threshold, 1.0, 2.0)).astype(np.float32)

    return XarrayDataCube(mask)

