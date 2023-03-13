import pickle as pkl
import numpy as np
from pprint import pprint as ppt
from pathlib import Path

from aes670hw2 import geo_plot as gp
from aes670hw2 import enhance
from aes670hw2 import guitools as gt

debug = True
region_pkl = Path("data/pkls/NP-IMG-west-region_20160712_2006.pkl")
out_image = Path("figures/test/categories.png")
category_count = 2
category_names = ["cloud", "land"]

with region_pkl.open("rb") as pklfp:
    data, info, geo, sunsat = pkl.load(pklfp)

rgb = np.dstack([ enhance.norm_to_uint(X, 256, np.uint8) for X in data ])

ppt(gt.get_category_series(
    X=rgb,
    cat_count=category_count,
    category_names=category_names,
    show_pool=True,
    debug=debug
    ))


gp.generate_raw_image(rgb, out_image)
