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
from aes670hw2 import geo_helpers
from aes670hw2 import guitools

token_file = Path("./laads_token")

def ez_naturalcolor(
        figure_path:Path=None, target_time:dt=None, buffer_dir:Path=None,
        bowtie_correct:bool=False, satellite:str="NP", pkl_path:Path=None,
        center_latlon:tuple=None, px_width:int=None, px_height:int=None,
        day_only:bool=True, debug:bool=False):
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
            day_only=day_only,
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
    latlon = None
    debug = True
    #latlon = (34.43, -86.35) # Huntsville
    #latlon = (.23, -50.07) # Amazon river delta
    #latlon = (36.39, 25.44) # Aegean sea
    #latlon = (36.44, -104.97) # Philmont (The West)
    #latlon = (-5.77, 141) # Papua
    #latlon = (24.65, 89.79) # Bangladesh
    #latlon = (45.93, 8.32) # Alps
    latlon = (-10,-60) # Amazon
    pkl_path = Path("data/pkls/ez_naturalcolor.pkl")
    fig_dir = Path("figures/test")
    #xres, yres = (2096, 1179)
    xres, yres = (2560, 1440)
    dx = int(xres/2)
    dy = int(yres/2)
    #target_time=dt.utcnow()
    target_time=dt(year=2018, month=9, day=5, hour=14)
    file_template = "{atime}_{satellite}_naturalcolor.png"
    satellite = "NP"

    response = ""
    if pkl_path.exists():
        while response not in ("y", "n"):
            print(f"Overwrite {pkl_path.as_posix()}?")
            response = input("(y/N): ").lower()


    if not pkl_path.exists() or response=="y":
        data, info, geo, sunsat = ez_naturalcolor(
                #figure_path=fig_path,
                target_time=target_time,
                buffer_dir=None,
                bowtie_correct=False,
                satellite=satellite,
                center_latlon=latlon,
                px_width=None,
                px_height=None,
                debug=debug
                )
        with pkl_path.open("wb") as pklfp:
            pkl.dump((data, info, geo, sunsat), pklfp)
    else:
        with pkl_path.open("rb") as pklfp:
            data, info, geo, sunsat = pkl.load(pklfp)

    rgb = []
    # Drop the terrain height data
    geo = np.stack([ g for g in geo[:2] ], axis=2)

    for i in range(len(data)):
        #X = np.ma.masked_where(info['qflags'][info['bands'][i]] > 0, data[i])
        X = np.ma.masked_where(data[i] > 400, data[i])
        X = np.ma.filled(X, 0)
        X = X.data
        print(enhance.array_stat(X))
        X = enhance.norm_to_uint(X, 256, np.uint8)
        data[i] = X
        #print(X)

    px_ctr = geo_helpers.get_closest_pixel(geo, latlon)
    print(geo.shape, latlon, px_ctr)

    yrange, xrange = geo_helpers.get_geo_range(
            geo, latlon, dx, dy, from_center=True,
            boundary_error=False, debug=True)

    # subset data from the center pixel to the requested resolution
    data = np.dstack([
            X[int(yrange[0]):int(yrange[1]), int(xrange[0]):int(xrange[1])]
            for X in data
            ])
    geo = geo[int(yrange[0]):int(yrange[1]),
              int(xrange[0]):int(xrange[1])]

    print("data, geo, px_ctr", data.shape, geo.shape, px_ctr)
    px_ctr = geo_helpers.get_closest_pixel(geo, latlon)
    #print(info)
    render = guitools.label_at_index(data, px_ctr)
    fig_path = fig_dir.joinpath(Path(file_template.format(
        atime=info['file_time'].strftime('%Y%m%d_%H%M'), satellite=satellite)))
    gp.generate_raw_image(render, fig_path)
