"""
Use the moderate-resolution bands covering my domain to select samples in
5 categories of surface types for supervised classification using 3 different
RGBs.

Merge pixel selections for each category and prints a JSON to stdout
containing pixel selections, category names, and unique mask colors.
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
#from aes670hw2 import imstat
from aes670hw2 import PixelCat

""" Settings """

debug = True
# Category labels for pixelwise selection.
categories = ("cloud", "water", "mtn_veg", "clay", "grassland", "crops")
# Brightness bins for histogram equalization and linear stretching.
nbins=256
generate_rgbs = True
figure_dir = Path("figures/pixel_selection")
figure_template = "selectionRGB_{bands}_{label}.png"

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

# If you're adapting this script for data other than the preloaded region_pkl,
# `region` is just a list of identically-shaped 2d ndarrays for a number of
# bands, and `bands` is a list of the same number of unique strings labeling the
# bands for the PixelCat object.

region, info, _, _ = pkl.load(region_pkl.open("rb"))
bands = info["bands"]

""" Initialize a PixelCat and add NDVI/NDSI arrays as new bands. """
pc = PixelCat(arrays=region, bands=bands)
NDVI = (pc.band("M07")-pc.band("M05"))/(pc.band("M07")+pc.band("M05"))
NDSI = (pc.band("M10")-pc.band("M04"))/(pc.band("M10")+pc.band("M04"))
# Add difference composites as bands
pc.set_band(array=NDVI, band="NDVI")
pc.set_band(array=NDSI, band="NDSI")

my_cats = [ {"pixels":set([]), "name":c, "color":[]} for c in categories ]
def update_my_cats(new_cats:list):
    """
    Helper method for updating the overall categories dictionary inside the
    scope of this script. This dictionary holds all unique pixels selected
    within each category for any RGB.
    """
    for i in range(len(categories)):
        assert my_cats[i]["name"] == new_cats[i]["name"]
        my_cats[i]["pixels"].update(new_cats[i]["pixels"])
        #my_cats[i]["pixels"] = list(new_cats[i]["pixels"])

"""
Show a histogram-equalized truecolor, and prompt the user to select pixels
for each category.
"""
truecolor_bands = ["M05", "M04", "M03"]
print(f"\033[93mEqualized truecolor composite of bands \033[0m\033[1m " + \
        "{truecolor_bands}\033[0m")
eq_and_norm = lambda X: enhance.norm_to_uint(
        enhance.histogram_equalize(X, nbins)[0], 256, np.uint8)
noeq_recipe = lambda X: enhance.norm_to_uint(X, 256, np.uint8)

#pc.get_rgb(bands=truecolor_bands, recipes=[noeq_recipe for i in range(3)])

# Equalize each truecolor band
rgb = pc.get_rgb(
        bands=truecolor_bands,
        recipes=[eq_and_norm for i in range(3)],
        show=False
        )
# Prompt the user for categories.
tc_cats = gt.get_category_series(
    X=rgb,
    cat_count=len(categories),
    category_names=categories,
    show_pool=True,
    debug=True
    )
update_my_cats(tc_cats)
if generate_rgbs:
    gp.generate_raw_image(
            rgb,
            figure_dir.joinpath(figure_template.format(
                bands="-".join(truecolor_bands),
                label="tcEQ"
                )))

"""
Show a custom RGB using the LWIR band and vegetation/ice indeces, and prompt
the user to select pixels for each category.
"""
custom_bands = ["M15", "NDVI", "NDSI"]
print(f"\033[93mCustom composite of bands \033[0m\033[1m " + \
        "{custom_bands}\033[0m")
# Equalize each truecolor band
rgb = pc.get_rgb(
        bands=custom_bands,
        recipes=[eq_and_norm for i in range(3)],
        show=False,
        )
# Prompt the user for categories
custom_cats = gt.get_category_series(
    X=rgb,
    cat_count=len(categories),
    category_names=categories,
    show_pool=True,
    debug=debug
    )
update_my_cats(custom_cats)
if generate_rgbs:
    gp.generate_raw_image(
            rgb,
            figure_dir.joinpath(figure_template.format(
                bands="-".join(custom_bands),
                label="custom"
                )))

"""
Show a day cloud phase RGB using the CIRA ABI quick guide recipe, and prompt
the user to select pixels for each category.
"""
dcp_bands = ["M09", "M05", "M10"]
dcp_recipes = (
    lambda X: enhance.norm_to_uint(enhance.linear_gamma_stretch(
        X, gamma=.66), 256, np.uint8),
    lambda X: enhance.norm_to_uint(enhance.linear_gamma_stretch(
        X), 256, np.uint8),
    lambda X: enhance.norm_to_uint(enhance.linear_gamma_stretch(
        X), 256, np.uint8),
    )

rgb = pc.get_rgb(bands=dcp_bands, recipes=dcp_recipes, show=False)
dcp_cats = gt.get_category_series(
        X=rgb,
        cat_count=len(categories),
        category_names=categories,
        show_pool=True,
        debug=debug
        )
update_my_cats(dcp_cats)
if generate_rgbs:
    gp.generate_raw_image(
            rgb,
            figure_dir.joinpath(figure_template.format(
                bands="-".join(dcp_bands),
                label="dcp"
                )))

for i in range(len(my_cats)):
    my_cats[i]["pixels"] = list(my_cats[i]["pixels"])

print(json.dumps(my_cats))


#print(pc.pick_gamma("NDVI"))
#print(pc.pick_linear_contrast("NDVI"))
'''
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
'''
