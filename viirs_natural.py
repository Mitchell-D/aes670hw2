"""
eZ viirs naturalcolor RGB script to serve as a demo, and as a kind of
quick "sanity check" on the many functions in the package.
"""
from pprint import pprint as ppt
import numpy as np
from pathlib import Path
from datetime import datetime as dt
import pickle as pkl

from aes670hw2 import viirs
from aes670hw2 import enhance
from aes670hw2 import geo_plot as gp

token_file = Path("./laads_token")

def ez_naturalcolor(
        figure_path:Path=None, target_time:dt=None, buffer_dir:Path=None,
        bowtie_correct:bool=False, satellite:str="NP", pkl_path:Path=None,
        center_latlon:tuple=None, px_width:int=None, px_height:int=None,
        debug:bool=False):
    """
    """
    buffer_dir = buffer_dir if buffer_dir else Path(".")
    target_time = target_time if target_time else dt.utcnow()
    get_subgrid = all([
        not arg is None for arg in (center_latlon, px_width, px_height) ])
    assert satellite.upper() in ("NP", "J1")

    product_key = f"V{satellite}02IMG"
    # Download and generate a pkl of the requested bands for the granule.
    data, info, geo, sunsat = viirs.download_viirs_granule(
            product_key=product_key,
            dest_dir=buffer_dir,
            token_file=token_file,
            bands=(3,2,1),
            target_time=target_time,
            latlon=center_latlon,
            keep_files=False,
            day_only=True,
            replace=True,
            include_sunsat=True,
            include_geo=True,
            mask=True,
            debug=debug
            )

    if bowtie_correct:
        data = [
            enhance.norm_to_uint(
                enhance.vertical_nearest_neighbor(A), 256).transpose()
            for A in data ]

    rgb = np.dstack([
        enhance.norm_to_uint(X, 256, np.uint8)
        for X in data])

    if debug:
        lat, lon, _ = geo
        print(f"lat range: ({np.amin(lat)}, {np.amax(lat)})",)
        print(f"lon range: ({np.amin(lon)}, {np.amax(lon)})",)
        print(f"geo avg:   ({np.average(lat)}, {np.average(lon)})",)

    #rgb = np.ma.masked_where(data > 400, data)
    #rgb = enhance.linear_gamma_stretch(data, lower=0, upper=1, gamma=2)

    if figure_path:
        gp.generate_raw_image(rgb, figure_path)
    return data, info, geo, sunsat

if __name__=="__main__":
    # Pixel location that must be present in the image
    latlon = (34.43, -86.35) # Huntsville
    #latlon = (.23, -50.07) # Amazon river delta
    #latlon = (36.39, 25.44) # Aegean sea
    #latlon = (36.44, -104.97) # Philmont (The West)
    #latlon = (-5.77, 141) # Papua
    #latlon = (24.65, 89.79) # Bangladesh
    #latlon = (45.93, 8.32) # Alps
    pkl_path = Path("data/pkls/ez_naturalcolor.pkl")

    response = ""
    if pkl_path.exists():
        while response not in ("y", "n"):
            print(f"Overwrite {pkl_path.as_posix()}?")
            response = input("(y/N): ").lower()

    if not pkl_path.exists() or response=="y":
        data, info, geo, sunsat = ez_naturalcolor(
                figure_path=Path("figures/test/ez_naturalcolor.png"),
                target_time=dt(year=2022, month=7, day=3, hour=1),
                buffer_dir=None,
                bowtie_correct=False,
                satellite="NP",
                #satellite="J1",
                center_latlon=latlon,
                px_width=None,
                px_height=None,
                debug=True
                )
        with pkl_path.open("wb") as pklfp:
            pkl.dump((data, info, geo, sunsat), pklfp)
    else:
        with pkl_path.open("rb") as pklfp:
            data, info, geo, sunsat = pkl.load(pklfp)
