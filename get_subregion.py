import numpy as np
from pprint import pprint as ppt
import pickle as pkl
from pathlib import Path
from datetime import datetime as dt

from aes670hw2 import guitools as gt
from aes670hw2 import geo_helpers
from aes670hw2 import geo_plot as gp
from aes670hw2 import enhance
from aes670hw2 import imstat

""" General settings """
debug = True
# Generate an RGB of the full domain showing the sub-domain.
generate_region_indicator = True
# If True, histogram-equalizes each channel before generating images
hist_equalize = False
nbins = 256 # Bins for optional histogram equalization
# pkl created with viirs.generate_viirs_pkl().
#pkl_path =  Path("data/pkls/NP-MOD-philmont_20160712_2006.pkl")
pkl_path =  Path("data/pkls/NP-IMG-philmont_20160712_2006.pkl")
region_pkl = Path(f"data/pkls/NP-IMG-west-region_20160712_2006.pkl")

""" Sub-region selection settings """
# Ideal center lat/lon of the region
target_latlon = (37.4, -107)
# Width/height in px of rectangle describing the ergion
region_width = 640*2
region_height = 512*2

""" RGB settings """
# Bands to construct RGB, which must've been loaded into the provided pkl:
# For moderate-res, choose 3 of: ['M15', 'M11', 'M10', 'M05', 'M04', 'M03']
#rgb_bands = ["M15", "M10", "M03"]
#rgb_bands = ["M05", "M04", "M03"]
rgb_bands = ["I03", "I02", "I01"]
# Image path for region subset RGB
#region_rgb_path =  Path("figures/pixel_selection/region_naturalcolor_equalized.png")
region_rgb_path =  Path("figures/pixel_selection/region_naturalcolor.png")
#region_rgb_path =  Path("figures/pixel_selection/region_truecolor.png")
# Image path for region location indicator RGB
#indicator_rgb_path =  Path("figures/pixel_selection/indicator_naturalcolor_equalized.png")
indicator_rgb_path =  Path("figures/pixel_selection/indicator_naturalcolor.png")
#indicator_rgb_path =  Path("figures/pixel_selection/indicator_truecolor.png")

# Load the pkl generated by viirs.generate_viirs_pkl()
with pkl_path.open("rb") as pklfp:
    data, info, geo, sunsat = pkl.load(pklfp)

# Coordinate axes are flipped in the pkls.
T = lambda X: X[::-1,::-1]
data = tuple(map(T, data))
geo = tuple(map(T, geo))
sunsat = tuple(map(T, sunsat))
info["qflags"] = { k:T(info["qflags"][k]) for k in info["qflags"].keys() }

# Check that RGB bands are provided and available
assert len(rgb_bands) == 3 and all((b in info["bands"] for b in rgb_bands))
# Get a bowtie-corrected RGB from the requested bands.
band_indeces = [ info["bands"].index(b) for b in rgb_bands ]
if debug: print(f"RGB bands loaded: {rgb_bands}")
print(band_indeces)

# Find a geographic range corresponding to the image size and center point.
latlon = np.dstack(geo[0:2])
ctr_lat_idx, ctr_lon_idx = geo_helpers.get_closest_pixel(
        latlon, target_latlon)
lat_range, lon_range = geo_helpers.get_geo_range(
        latlon=latlon,
        target_latlon=target_latlon,
        dx_px=region_width,
        dy_px=region_height,
        from_center=True,
        debug=debug,
        )

# Subset the arrays to the desired region
region = [ X[lat_range[0]:lat_range[1],lon_range[0]:lon_range[1]]
          for X in data ]
geo = [ G[lat_range[0]:lat_range[1],lon_range[0]:lon_range[1]] for G in geo ]

with region_pkl.open("wb") as pklfp:
    if debug:
        print(f"Dumping bands for selected region to {region_pkl.as_posix()}")
    pkl.dump((region, info, geo, sunsat), pklfp)

# If requested, generate a bowtie-corrected image showing the sub-region
if generate_region_indicator:
    indicator_rgb = []
    # Enhance each of the bands
    for i in band_indeces:
        # Having some issues with the mask, just mask out-of-range values.
        #X = np.ma.masked_where(info["qflags"][info["bands"][i]]!=0, data[i])
        X = np.ma.masked_where(data[i]>400, data[i])
        X = enhance.vertical_nearest_neighbor(X, debug=debug)
        if hist_equalize:
            X, _, _ = imstat.histogram_equalize(X, nbins=nbins, debug=debug)
        X = enhance.norm_to_uint(X, 256, cast_type=np.uint8)
        indicator_rgb.append(X)

    # Rasterize a rectangle and centerpoint marker on the image
    indicator_rgb = np.dstack(indicator_rgb)
    indicator_rgb = gt.rect_on_rgb(
            indicator_rgb, lat_range, lon_range, thickness=4)
    indicator_rgb = gt.label_at_index(
            indicator_rgb, (ctr_lat_idx, ctr_lon_idx), size=25,
            color=(256, 0, 0), thickness=4)
    gp.generate_raw_image(
            RGB=indicator_rgb,
            image_path=indicator_rgb_path,
            )

# Histogram-normalize the subsetted RGB bands if requested.
region_rgb = []
for i in range(len(band_indeces)):
    X = region[band_indeces[i]]
    if hist_equalize:
        X, _, _ = imstat.histogram_equalize(X, nbins=nbins, debug=debug)
    region_rgb.append(enhance.norm_to_uint(X, 256, np.uint8))

# Generate an RGB of the selected sub-region
gp.generate_raw_image(
        RGB=np.dstack(region_rgb),
        image_path=region_rgb_path,
        )

