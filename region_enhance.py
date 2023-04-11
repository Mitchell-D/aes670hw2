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
fig_dir = Path("figures/histogram_analysis")
#figure_template = "selectionRGB_{bands}_{label}.png"

"""
Get a list of N arrays and N band labels from the stored pkl if it exists,
or use the restore_pkl helper script to re-download and subset it.
"""
# out-of-package helper script. Must be in same directory.
#region_pkl = Path("/home/krttd/uah/23.s/aes670/aes670hw2/data/pkls/" + \
#        "my_region.pkl")
region_pkl = Path("/home/krttd/uah/23.s/aes670/aes670hw2/data/pkls/" + \
        "NP-IMG-west-region_20160712_2006.pkl")
if not region_pkl.exists():
    from restore_pkl import restore_pkl
    restore_pkl(region_pkl, debug=debug)

# If you're adapting this script for data other than the preloaded region_pkl,
# `region` is just a list of identically-shaped 2d ndarrays for a number of
# bands, and `bands` is a list of the same number of unique strings labeling the
# bands for the PixelCat object.

region, info, _, _ = pkl.load(region_pkl.open("rb"))
bands = info["bands"]

""" Initialize a PixelCat and generate a truecolor """
#tc_bands = ["M05", "M04", "M03"]
tc_bands = ["I03", "I02", "I01"]
pc = PixelCat(arrays=region, bands=bands)

rgb_path = fig_dir.joinpath(Path("truecolor_original.png"))
rgb = pc.get_rgb(tc_bands, normalize=False)
gp.generate_raw_image(rgb, rgb_path)

""" Prompt the user to select contrast for each channel """
for b in tc_bands:
    pc.pick_linear_contrast(b, set_band=True)

""" Generate the RGB the user selected """
contrast_rgb_path = fig_dir.joinpath(Path("truecolor_contrast.png"))
rgb = pc.get_rgb(tc_bands, normalize=False)
gp.generate_raw_image(rgb, contrast_rgb_path)

""" Do histogram analysis on the RGB """
nbins = 512
histograms = []
for i in range(len(tc_bands)):
    histograms.append(enhance.do_histogram_analysis(
        pc.band(tc_bands[i]), nbins, equalize=False, debug=debug))
    histograms[i].update({"band":tc_bands[i]})

gp.plot_lines(
        domain=range(nbins),
        ylines=[h["hist"] for h in histograms],
        labels=tc_bands,
        image_path=fig_dir.joinpath(Path("img_histograms_contrast.png")),
        plot_spec={
            "title": "VIIRS Imagery band frequency histograms",
            "xlabel": "Brightness level bin",
            "ylabel": "Brightness level frequency",
            "line_width":1,
            },
        show=False,
        )
