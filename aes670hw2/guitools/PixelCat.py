
from dataclasses import dataclass
import pickle as pkl
from pathlib import Path
import numpy as np
import math as m

from aes670hw2 import enhance
from aes670hw2 import guitools as gt

class PixelCat:
    """
    Wrapper class for supervised and unsupervised classification of arrays.

    Initialized with an immutable set of arrays and labels. Any enhancements
    must be done as pre or post processing, but show_rgb() allows for
    temporary visualization of provided arrays with a rgb recipe.
    """
    def __init__(self, arrays:list, bands:list):
        """
        :@param arrays: List of uniformly-shaped arrays of data
        :@param bands: List of unique labels corresponding to each array
        """
        # Arrays must be uniformly-shaped, and each one must have a unique
        # band label.
        assert all([ X.shape == arrays[0].shape for X in arrays ])
        assert len(set(bands)) == len(arrays) and len(arrays)
        self._arrays = arrays
        self._bands = bands
        self._shape = arrays[0].shape

    def band(self, band:str):
        """ Returns the array associated with the provided band label """
        assert band in self._bands
        return self._arrays[self._bands.index(band)]

    def pick_linear_contrast(self, band:str, offset:float=0, debug=False):
        assert band in self._bands
        slope_scale = 50
        return m.e**((gt.trackbar_select(
                X=self.band(band),
                func=lambda X, v: enhance.linear_contrast(
                        X, m.e**((v-100)/slope_scale), offset),
                label=f"Band {band} gamma: ",
                debug=debug
                )-100)/slope_scale)

    '''
    def get_cats(bands:list, cat_count:int cat_names:list=None,
                 show_pool=False, debug=False):
        """
        """
        if type(bands)==str:
            bands = [ bands for i in range(3) ]
        assert len(bands) == 3

        return gt.get_category_series(X=np.dstack(bands), cat_count=cat_count,
                category_names=cat_names, show_pool=show_pool, debug=debug)
    '''

    def pick_gamma(self, band:str, debug=False):
        assert band in self._bands
        gamma_scale = 8
        return gt.trackbar_select(
                X=self.band(band),
                func=lambda X, v: enhance.gamma(X, (1+v)*(gamma_scale/255)),
                label=f"Band {band} gamma: ",
                debug=debug
                )

    def get_rgb(self, bands:list, recipes:list=None, show=True, debug=False):
        """
        Compile a (M,N,3)-shaped RGB numpy array of bands corresponding to
        the provided list of labels.
        By design, this class doesn't make any in-place modifications to its
        arrays, so unless a recipe is provided
        """
        # Verify 3 valid band labels were provided in the list
        assert len(bands) == 3 and all([ b in self._bands for b in bands ])
        # Verify that if recipes were provided, they are 3 functions.
        assert not recipes or len(recipes) == 3 and \
                all([ type(r)==type(lambda m:1) for r in recipes  ])
        recipes = recipes if recipes else [ lambda m:m for i in range(3) ]
        if debug: print(f"Generating RGB using bands {bands}")
        RGB = np.dstack([ recipes[i](self.band(bands[i]))
                          for i in range(len(bands))])
        if show:
            gt.quick_render(RGB)
        return RGB

    def set_band(self, array:np.ndarray, band:str, replace=False):
        """
        Add a new band array, which must have the same shape as the others,
        or set an existing band array to a new one.

        :@param array: Numpy ndarray with same shape as other arrays.
        :@param band: Unique string label for the new array. If replace=False
                and the band key already exists, and error is raised.
                Otherwise, the new array will replace the old one.
        """
        assert replace or band not in self._bands
        assert array.shape == self._shape

        if band in self._bands:
            if replace:
                self._arrays[self._bands.index(band)] = array
            else:
                raise ValueError(f"Band key {band} already exists")
            return None
        self._arrays.append(array)
        self._bands.append(band)
        return None

if __name__=="__main__":
    debug=True
    nbins=1024

    region_pkl = Path("/home/krttd/uah/23.s/aes670/aes670hw2/data/pkls/" + \
            "my_region.pkl")
    if not region_pkl.exists():
        from restore_pkl import restore_pkl
        restore_pkl(region_pkl, debug=debug)
    region, info, _, _ = pkl.load(region_pkl.open("rb"))

    pc = PixelCat(arrays=region, bands=[ b for b in info["bands"] ])

    truecolor_bands = ["M05", "M04", "M03"]
    tc_recipe = lambda X: enhance.norm_to_uint(
            enhance.histogram_equalize(X, nbins)[0], 256, np.uint8)
    noeq_recipe = lambda X: enhance.norm_to_uint(X, 256, np.uint8)

    # Show unequalized RGB
    pc.get_rgb(bands=truecolor_bands, recipes=[noeq_recipe for i in range(3)])
    # Show histogram-equalized RGB
    pc.get_rgb(bands=truecolor_bands, recipes=[tc_recipe for i in range(3)])

    pc.set_band(
        array=(pc.band("M07")-pc.band("M05"))/(pc.band("M07")+pc.band("M05")),
        band="NDVI"
        )

    pc.set_band(
        array=(pc.band("M10")-pc.band("M04"))/(pc.band("M10")+pc.band("M04")),
        band="NDSI"
        )

    custom_bands = ["M15", "NDVI", "NDSI"]
    pc.get_rgb(bands=custom_bands, recipes=[tc_recipe for i in range(3)])
