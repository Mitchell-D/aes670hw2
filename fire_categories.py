"""
Loads the region pkl generated by fire_look.py, generates RGBs, and
performs the Flasse and Ceccato fire detection algorithm with a FireDetector.
"""

import pickle as pkl
import numpy as np
import json
from pathlib import Path
from pprint import pprint as ppt

from aes670hw2 import FireDetector
from aes670hw2 import PixelCat
from aes670hw2 import enhance
from aes670hw2 import guitools as gt
from aes670hw2 import geo_plot as gp

region_pkl = Path("data/pkls/20180904_amazon_fire.pkl")
json_path = Path("data/fire_classes.json")
with region_pkl.open('rb') as pklfp:
    data, info, geo = pkl.load(pklfp)

debug = True
image_path = Path("figures/fire_look/amazon_fire_rgb.png")

pc = PixelCat(
        [ X[::-1,::-1] for X in data],
        ("LWIR", "SWIR", "VIS"))

pc.set_band(pc.band("SWIR")-pc.band("LWIR"), "IRdiff")

""" Show red, green, and blue combinations independently """
'''
print(f"SWIR-LWIR difference (Red)")
print("Selected SWIR-LWIR gamma",
      pc.pick_linear_gamma("IRdiff", set_band=True, hist_equalize=True))
#gt.quick_render(enhance.linear_gamma_stretch(pc.band("IRdiff")))
print(f"LWIR (Green)")
print("Selected LWIR gamma",
      pc.pick_linear_gamma("LWIR", set_band=True, hist_equalize=True))
#gt.quick_render(enhance.linear_gamma_stretch(pc.band("LWIR")))
print(f"Average visible (Blue)")
print(f"Selected VIS gamma ",
      pc.pick_linear_gamma("VIS", set_band=True, hist_equalize=True))
print(f"LWIR (Green) Brightness Scale")
print(f"Selected green scale",
      pc.pick_band_scale("LWIR", set_band=True))
#gt.quick_render(enhance.linear_gamma_stretch(pc.band("VIS")))
'''

rgb_bands =["IRdiff", "LWIR", "VIS"]
rgb = pc.get_rgb(
        bands=rgb_bands,
        recipes=[lambda X: enhance.linear_gamma_stretch(X),
                 lambda X: enhance.linear_gamma_stretch(X)*.3,
                 lambda X: enhance.linear_gamma_stretch(X)],
        show=False, debug=debug)

'''
cats = gt.get_category_series(rgb, cat_count=4,
                       category_names=("fire", "smoke", "cloud", "ground"))
    print(json.dumps(cats))
'''

cats = json.load(json_path.open("rb"))
for c in range(len(cats)):
    print(cats[c]["name"])
    for k in range(3):
        cats[c][rgb_bands[k]] = np.asarray(
                [ pc.band(rgb_bands[k])[i,j] for i,j in cats[c]["pixels"] ])
        print(f"    {rgb_bands[k]}: mean: {np.mean(cats[c][rgb_bands[k]])}")

#gt.quick_render(rgb)
gp.generate_raw_image(enhance.norm_to_uint(rgb, 256, np.uint8), image_path)
