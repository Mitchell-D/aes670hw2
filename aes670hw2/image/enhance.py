import numpy as np
from scipy.interpolate import interp1d

def linear_gamma_stretch(X:np.ndarray, lower:float=None, upper:float=None,
                  gamma:float=1):
    """
    Linear-normalize pixel values between lower and upper bound, then apply
    gamma stretching if an argument is provided.

    Returns an array of float values in range [0, 1]
    """
    lower = lower if not lower is None else np.amin(X)
    upper = upper if not upper is None else np.amax(X)
    return np.clip(((X-lower)/(upper-lower))**(1/gamma), 0, 1)

def norm_to_uint(X:np.ndarray, resolution:int, cast_type:type=np.uint):
    """
    Linearally normalizes the provided float array to bins between 0 and
    resolution-1, and returns the new integer array as np.uint.
    """
    return (np.floor(linear_gamma_stretch(X)*(resolution-1))).astype(cast_type)

def vertical_nearest_neighbor(X:np.ma.MaskedArray, debug=False):
    """
    Linearly interpolates masked values of a 2d array along axis 0,
    independently for each column using nearest-neighbor interpolation.
    If you need horizontal interpolation, transpose the array.

    This method is intended for VIIRS bowtie correction as a cosmetic
    correction, but may be used for other purposes.
    """
    print(f"Masked values: {np.count_nonzero(X.mask)}")
    if len(X.shape) != 2:
        raise ValueError(f"Array must be 2d; provided array shape: {X.shape}")

    if debug:
        print("Vertical-NN interpolating " + \
                f"{X.size-np.count_nonzero(X.mask)} points.")
    for i in range(X.shape[1]):
        col = X[:,i]
        valid = np.where(np.logical_not(col.mask))[0]
        f = interp1d(valid, col[valid], fill_value="extrapolate")
        X[:,i] = np.array(f(range(X.shape[0])))
    return X
