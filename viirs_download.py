"""
A quick-and-dirty script for generating pkls of VIIRS imagery bands 3, 2, and
1, and moderate-res bands 16, 11, 10, 5, 4, and 3, as well as IMG natural
color, and MOD truecolor/cloud phase.
"""
from pprint import pprint as ppt
import numpy as np
from pathlib import Path
from datetime import datetime as dt
import pickle as pkl

from aes670hw2 import viirs
from aes670hw2 import enhance
from aes670hw2 import geo_plot as gp

""" Configuration options """

# Directory where pickled data will be generated
pkl_dir = Path("./data/pkls")
# Directory where figures will be generated
figure_dir = Path("./figures/viirs-full")
# Directory to buffer downloaded netCDFs (deleted afterwards by default)
nc_dir = Path("./data/pkls")
# LAADS DAAC token file
token_file = Path("./laads_token")
# Prints debug information if True
debug = True
# Target acquisition time of the image. Most recent if None.
target_time = dt(year=2016, month=7, day=9, hour=12, minute=36)
# If True, NN-interpolates along the vertical axis for bowtie correction.
bowtie_correct = True
# Only accept swaths acquired during the day
day_only = True
# Select a satellite to query
satellite = "NP" # NP or J1
# Short string to identify this domain
domain_desc = "europe"
# Pixel location that must be present in the image
# latlon = (34.43, -86.35) # Huntsville
# latlon = (.23, -50.07) # Amazon river delta
# latlon = (36.39, 25.44) # Aegean sea
#latlon = (36.44, -104.97) # Philmont (The West)
# latlon = (-5.77, 141) # Papua
# latlon = (24.65, 89.79) # Bangladesh
latlon = (45.93, 8.32) # Alps

"""
Get a pkl for the pseudo truecolor in imagery resolution and generate an image.
"""

# Select imagery resolution and bands for natural color
resolution = "IMG"
bands = (3,2,1)
product_key = f"V{satellite}02{resolution}"
label = f"{satellite}-{resolution}-{domain_desc}"
# Download and generate a pkl of the requested bands for the granule.
img_pkl_path = viirs.generate_viirs_pkl(
       product_key=product_key, target_time=target_time, label=label,
       dest_dir=pkl_dir, token_file=token_file, bands=bands, day_only=day_only,
       replace=True, latlon=latlon, include_sunsat=True, include_geo=True,
       mask=True, debug=debug)

# Re-load the data and generate a naturalcolor image.
with img_pkl_path.open("rb") as pklfp:
    data, info, geo, sunsat = pkl.load(pklfp)

if bowtie_correct:
    data = [
        enhance.norm_to_uint(
            enhance.vertical_nearest_neighbor(A), 256).transpose()
        for A in data ]

data = np.dstack(data)

naturalcolor_img_path = figure_dir.joinpath(Path(
        info['file_time'].strftime('%Y%m%d-%H%M') + \
                f"_375m_{label}-naturalcolor.png"))
if debug:
    lat, lon, _ = geo
    print(f"lat range: ({np.amin(lat)}, {np.amax(lat)})",)
    print(f"lon range: ({np.amin(lon)}, {np.amax(lon)})",)
    print(f"geo avg:   ({np.average(lat)}, {np.average(lon)})",)
rgb = np.ma.masked_where(data > 400, data)
rgb = enhance.linear_gamma_stretch(data, lower=0, upper=1, gamma=2)
gp.generate_raw_image(rgb[::-1,::-1,:], naturalcolor_img_path)

"""
Get a pkl for the moderate-resolution bands 11, 10, 5, 4, and 3, and generate
a truecolor and cloud-phase image to use as a sample.
"""
# Select moderate resolution and bands for truecolor and cloud phase RGBs
resolution = "MOD"
bands = (15, 11, 10, 5, 4, 3)
product_key = f"V{satellite}02{resolution}"
label = f"{satellite}-{resolution}-{domain_desc}"
# Download and generate a pkl fof the requested bands for the granule
mod_pkl_path = viirs.generate_viirs_pkl(
       product_key=product_key, target_time=target_time, label=label,
       dest_dir=pkl_dir, token_file=token_file, bands=bands, day_only=day_only,
       replace=True, latlon=latlon, include_sunsat=True, include_geo=True,
       mask=True, debug=debug)

# Re-load the data and generate a truecolor image.
with mod_pkl_path.open("rb") as pklfp:
    data, info, geo, sunsat = pkl.load(pklfp)

if bowtie_correct:
    data = [
        enhance.norm_to_uint(
            enhance.vertical_nearest_neighbor(A), 256).transpose()
        for A in data ]

data = np.dstack(data)

truecolor_img_path = figure_dir.joinpath(Path(
    info['file_time'].strftime('%Y%m%d-%H%M') + \
            f"_750m_{label}-truecolor.png"))
rgb = np.ma.masked_where(data > 400, data)
rgb = enhance.linear_gamma_stretch(rgb[:,:,3:6], lower=0, upper=1, gamma=2)
gp.generate_raw_image(rgb[::-1,::-1,:], truecolor_img_path)

# Generate cloud phase image from eumetsat recipe:
# https://www.eumetsat.int/media/42197
cloudphase_img_path = figure_dir.joinpath(Path(
    info['file_time'].strftime('%Y%m%d-%H%M') + \
            f"_750m_{label}-cloudphase.png"))
rgb = np.dstack((
    enhance.linear_gamma_stretch(data[:,:,2], lower=0, upper=.5),
    enhance.linear_gamma_stretch(data[:,:,1], lower=0, upper=.5),
    enhance.linear_gamma_stretch(data[:,:,5], lower=0, upper=1.)))

gp.generate_raw_image(rgb[::-1,::-1,:], cloudphase_img_path)

mod_pkl_size = round(mod_pkl_path.stat().st_size*100e-6)/100
img_pkl_size = round(img_pkl_path.stat().st_size*100e-6)/100

print("\033[1mSuccessfully Downloaded Granule.\n" + \
        f"\033[96m{img_pkl_size}MB Image resolution pkl generated at" + \
        f"\n\t\033[92m{img_pkl_path.as_posix()}\n" + \
        f"\033[96m{mod_pkl_size}MB Moderate resolution pkl generated at" + \
        f"\n\t\033[92m{mod_pkl_path.as_posix()}\n" + \
        "\033[96mNatural color image generated at" + \
        f"\n\t\033[92m{naturalcolor_img_path.as_posix()}\n" + \
        "\033[96mTrue color image generated at" + \
        f"\n\t\033[92m{truecolor_img_path.as_posix()}\n"
        "\033[96mCloud phase image generated at" + \
        f"\n\t\033[92m{cloudphase_img_path.as_posix()}\033[0m"
      )
