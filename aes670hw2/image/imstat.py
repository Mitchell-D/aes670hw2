
import numpy as np
from . import enhance

def array_stat(X:np.ndarray):
    """
    Returns a dict of useful info about an array.
    """
    return {
            "shape":X.shape,
            "stddev":np.std(X),
            "mean":np.average(X),
            "min":np.amin(X),
            "max":np.amax(X),
            "range":np.ptp(X),
            }

def get_pixel_counts(X:np.ndarray, nbins, debug=False):
    """
    Returns an integer array with length nbins which depicts the number of
    pixels with values in the index bin.

    Index zero corresponds to the bin from Xmin to Xmin + bin_size

    :@param X: any numpy ndarray with values that can be uniformly binned
    :@param nbins: Number data values ('resolution') to bin X into.
    :@return: tuple like (counts, bin_size, Xmin) containing an array with
            size nbins with the pixel count in each bin in X, a float value
            bin_size indicating the bin size in data coordinates, and a float
            value Xmin indicating the floor of the fist bin in data coords.
    """
    X = X.compressed() # Get all unmasked values
    Xmin = np.amin(X)
    Xmax = np.amax(X)
    bin_size = (Xmax-Xmin)/nbins
    if debug:
        print(f"Binning {X.size} unmasked data points")
        print(f"Original array range: ({Xmin}, {Xmax})")
    counts = np.zeros(nbins)
    X = enhance.norm_to_uint(X, nbins)
    for px in X:
        counts[px] += 1
    return counts, bin_size, Xmin

def get_cumulative_hist(X:np.ndarray, nbins:int, debug=False):
    """
    Get a cumulative array of binned pixel values for equalization

    :@param X: any numpy ndarray with values that can be uniformly binned
    :@param nbins: Number data values ('resolution') to bin X into.
    :@return: tuple like (counts, bin_size, Xmin) containing an array with
            size nbins with the cumulative pixel count up to that brightness
            level for each bin in X, a float value bin_size indicating the bin
            size in data coordinates, and a float value Xmin indicating the
            floor of the fist bin in data coords.
    """
    total = 0
    counts, bin_size, Xmin = get_pixel_counts(X, nbins)
    if debug: print("Accumulating histogram counts")
    for i in range(counts.size):
        total += counts[i]
        counts[i] = total
    return counts, bin_size, Xmin

def histogram_equalize(X:np.ndarray, nbins:int,
                       cumulative_histogram:np.array=None, debug=False):
    """
    Get a histogram-equalized version of X

    Y = (N-1)/S * C(X)
    Where X is the corrected array with N brightness bins and S pixels, C(X)
    is the cumulative number of pixels up to the brightness bin of pixel X.

    :@param X: any numpy ndarray with values that can be uniformly binned
    :@param nbins: Number data values ('resolution') to bin X into.
    :@param cumulative_histogram: 1-d array describing a custom cumulative
            histogram curve for correcting X. Array size must be nbins. If a
            histogram is provided, bin_size and Xmin are unknowable and will be
            returned as None.
    :@return: tuple like (counts, bin_size, Xmin) containing an array with
            size nbins with the cumulative pixel count up to that brightness
            level for each bin in X, a float value bin_size indicating the bin
            size in data coordinates, and a float value Xmin indicating the
            floor of the fist bin in data coords.
    """
    if cumulative_histogram is None:
        c_hist, bin_size, Xmin = get_cumulative_hist(X, nbins)
    else:
        if cumulative_histogram.size != nbins:
            raise ValueError(
                    f"Provided histogram must have {nbins} members, not " +
                    str(cumulative_histogram.size))
        c_hist = cumulative_histogram
        bin_size = None
        Xmin = None

    hist_constant = (nbins-1)/X.size
    hist_scale = hist_constant*c_hist
    if debug: print(f"Equalizing histogram with scale {hist_constant}")
    Y = np.vectorize(lambda px: c_hist[px])(enhance.norm_to_uint(X, nbins))
    return Y, bin_size, Xmin

def do_histogram_analysis(X:np.ndarray, nbins:int, equalize:bool=False,
                          debug=False):
    """
    High-level helper method aggregating the frequency and cumulative
    frequency histograms of a provided array using n normalized value bins
    with a histogram-equalized array

    :@param X: any numpy ndarray with values that can be uniformly binned
    :@param nbins: Number data values ('resolution') to bin X into.
    :@param equalize: if True, the histogram equalization algorithm is
            applied and returned dictionary
    """
    freq, bin_size, Xmin = get_pixel_counts(X, nbins, debug=debug)
    cumulative_freq, _, _ = get_cumulative_hist(X, nbins, debug=debug)
    hist_dict= {
            "px_count":X.size,
            "hist":freq/X.size,
            "c_hist":cumulative_freq/X.size,
            "domain":np.linspace(Xmin, Xmin+nbins*bin_size, nbins),
            "bin_size":bin_size,
            "Xmin":Xmin,
            "stddev":np.std(X),
            "mean":np.average(X),
            "equalized":None,
            }

    if equalize:
        hist_dict["equalized"], _, _ = histogram_equalize(
                X, nbins, cumulative_freq, debug=debug)

    return hist_dict
