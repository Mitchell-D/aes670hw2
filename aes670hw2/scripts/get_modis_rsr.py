"""
A really cheeky way of getting RSR lookup tables for all 36 MODIS bands.
"""
import requests
import pickle as pkl
import json
from pprint import pprint as ppt

url = "https://nwp-saf.eumetsat.int/downloads/rtcoef_rttov13/ir_srf/rtcoef_eos_1_modis_srf/"
file = "rtcoef_eos_1_modis_srf_ch{i:02}.txt"
RSRs = {}
for i in range(1, 37):
    # Parse the valid CSV lines
    lines = requests.get(url + file.format(i=i)).text.split("\n")[4:-1]
    # Extract wave number and rsr values
    wn, rsr = zip(*map(lambda L: tuple(map(float, L.split())), lines))
    # Convert to wavelength in um
    wl = list(map(lambda v: (1e4)/v, wn))
    RSRs.update({i:{"wavelength":wl, "rsr":rsr}})

#with open("modis_rsrs.pkl", "wb") as pklfp:
#    pkl.dump(RSRs, pklfp)
json.dump(RSRs, Path("modis_rsrs.json").open("w"))
