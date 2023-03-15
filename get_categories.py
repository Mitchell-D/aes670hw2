"""
Use the moderate-resolution bands covering my domain to select samples in
5 categories of surface types for supervised classification.
"""
import pickle as pkl
import numpy as np
from pprint import pprint as ppt
from pathlib import Path
import json

from aes670hw2 import geo_plot as gp
from aes670hw2 import enhance
from aes670hw2 import laads
from aes670hw2 import viirs
from aes670hw2 import guitools as gt
from aes670hw2 import geo_helpers
from aes670hw2 import imstat
from aes670hw2 import PixelCat

"""
Get a list of N arrays and N band labels from the stored pkl if it exists,
or use the restore_pkl helper script to re-download and subset it.
"""
# out-of-package helper script. Must be in same directory.
region_pkl = Path("/home/krttd/uah/23.s/aes670/aes670hw2/data/pkls/" + \
        "my_region.pkl")
if not region_pkl.exists():
    from restore_pkl import restore_pkl
    restore_pkl(region_pkl, debug=debug)
region, info, _, _ = pkl.load(region_pkl.open("rb"))
bands = info["bands"]

""" Settings """
debug = True
categories = ("cloud", "water", "mtn_veg", "clay", "grassland", "crops")
nbins=256

""" Initialize a PixelCat and add NDVI/NDSI arrays as new bands. """
pc = PixelCat(arrays=region, bands=bands)
NDVI = (pc.band("M07")-pc.band("M05"))/(pc.band("M07")+pc.band("M05"))
NDSI = (pc.band("M10")-pc.band("M04"))/(pc.band("M10")+pc.band("M04"))
pc.set_band(array=NDVI, band="NDVI")
pc.set_band(array=NDSI, band="NDSI")

""" Show an unequalized and a histogram-equalized truecolor """
truecolor_bands = ["M05", "M04", "M03"]
tc_recipe = lambda X: enhance.norm_to_uint(
        imstat.histogram_equalize(X, nbins)[0], 256, np.uint8)
noeq_recipe = lambda X: enhance.norm_to_uint(X, 256, np.uint8)
#pc.get_rgb(bands=truecolor_bands, recipes=[noeq_recipe for i in range(3)])
rgb = pc.get_rgb(bands=truecolor_bands, recipes=[tc_recipe for i in range(3)])

print(json.dumps((gt.get_category_series(
    X=rgb,
    cat_count=len(categories),
    category_names=categories,
    show_pool=True,
    debug=debug
    ))))

#gp.generate_raw_image(rgb, out_image)


""" Show a custom RGB using the LWIR band and vegetation/ice indeces """
custom_bands = ["M15", "NDVI", "NDSI"]
rgb = pc.get_rgb(
        bands=custom_bands,
        recipes=[tc_recipe for i in range(3)],
        show=False,
        )

print(json.dumps((gt.get_category_series(
    X=rgb,
    cat_count=len(categories),
    category_names=categories,
    show_pool=True,
    debug=debug
    ))))

exit(0)

#print(pc.pick_gamma("NDVI"))
print(pc.pick_linear_contrast("NDVI"))

""" Demonstrate saturation enhancement """
saturation_recipe = lambda X: enhance.norm_to_uint(
            enhance.saturated_linear_contrast(
                tc_recipe(X), nbins, lower_sat_pct=0, upper_sat_pct=1),
            resolution=256,
            cast_type=np.uint8
            )
rgb = pc.get_rgb(
        bands=truecolor_bands,
        recipes=[saturation_recipe for i in range(3)],
        show=True
        )

