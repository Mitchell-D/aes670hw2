"""
Helper script to re-download and subgrid the image for this assignment.
Re-downloads and regrids region pickle in the standard format.
"""
import pickle as pkl
import numpy as np
from pprint import pprint as ppt
from pathlib import Path

from aes670hw2 import geo_plot as gp
from aes670hw2 import enhance
from aes670hw2 import laads
from aes670hw2 import viirs
from aes670hw2 import guitools as gt
from aes670hw2 import geo_helpers
#from aes670hw2 import imstat

# VIIRS
image_url = "https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/archives/"+\
        "VNP02MOD.A2016194.2006.002.2021095194801.nc"
geo_url = "https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/archives/"+\
        "VNP03MOD.A2016194.2006.002.2021056080948.nc"
token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUFMgT0F1dGgyIEF1dGhlbnRpY2F0b3IiLCJpYXQiOjE2Nzg0MDEyNzgsIm5iZiI6MTY3ODQwMTI3OCwiZXhwIjoxNjkzOTUzMjc4LCJ1aWQiOiJtZG9kc29uIiwiZW1haWxfYWRkcmVzcyI6Im10ZDAwMTJAdWFoLmVkdSIsInRva2VuQ3JlYXRvciI6Im1kb2Rzb24ifQ.gwlWtdrGZ1CNqeGuNvj841SjnC1TkUkjxb6r-w4SOmk"
target_latlon = (37.4, -107)
region_width = 640
region_height = 512
bands = (
        15, # 10.76um LWIR; longwave window channel
        14, # 8.55um MIR; surface emissivity, 8.5-10.7 for cloud phase
        12, # 3.70um SWIR; small ice crystals more reflective, fire detection
        10, # 1.61um NIR; Snow/ice absorb, liquid water reflects.
        9,  # 1.38um NIR; Cirrus detection (atmospheric abs band)
        7,  # 0.865um NIR; Strong chloryphyl reflection.
        5,  # 0.672 VIS; Red
        4,  # 0.555 VIS; Green
        3,  # 0.488 VIS; Blue
        )

def restore_pkl(region_pkl_path:Path, debug=False):
    assert not region_pkl_path.exists()
    nc_file = laads.download(image_url, region_pkl_path.parent,
                             raw_token=token, replace=True, debug=debug)
    geo_file = laads.download(geo_url, region_pkl_path.parent,
                             raw_token=token, replace=True, debug=debug)
    data, info = viirs.get_viirs_data(l1b_file=nc_file, bands=bands,
                                      mask=False, debug=debug)
    geo = viirs.get_viirs_geoloc(geoloc_file=geo_file, debug=debug)
    sunsat = viirs.get_viirs_sunsat(geoloc_file=geo_file, debug=debug)

    nc_file.unlink()
    geo_file.unlink()

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
    geo = [ G[lat_range[0]:lat_range[1],
              lon_range[0]:lon_range[1]] for G in geo ]
    sunsat = [ S[lat_range[0]:lat_range[1],
              lon_range[0]:lon_range[1]] for S in sunsat ]
    info["qflags"] = { k:info["qflags"][k][lat_range[0]:lat_range[1],
              lon_range[0]:lon_range[1]] for k in info["qflags"].keys() }

    pkl.dump((region, info, geo, sunsat), region_pkl_path.open("wb"))
    return (region, info, geo, sunsat)
