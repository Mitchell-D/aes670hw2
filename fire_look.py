from pathlib import Path
from pyhdf.SD import SD, SDC
import numpy as np
from pprint import pprint as ppt
from datetime import datetime as dt

from aes670hw2 import modis
from aes670hw2 import laads
from aes670hw2 import enhance
from aes670hw2 import geo_helpers
from aes670hw2 import geo_plot as gp
from aes670hw2 import guitools
from aes670hw2 import FireDetector
from aes670hw2 import PixelCat

def get_fire_passes(start_time:dt, end_time:dt, latlon:tuple, buffer_dir:Path,
                    satellite="terra", token_file=Path("laads_token"),
                    width_px:int=512, height_px:int=512, remove_hdfs:bool=True,
                    gen_nir_sample:bool=False, debug=False):
    """  """
    assert satellite.lower() in {"terra", "aqua"}
    products = modis.query_modis_l2(product_key=modis.sat_to_l2key(satellite),
            start_time=start_time, end_time=end_time, latlon=latlon,
            day_only=False, debug=debug)
    '''
    # temporary, obviously
    products = [{
            'atime': dt(2019, 9, 28, 14, 40),
            'downloadsLink': 'https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/archives/MOD09.A2019271.1440.006.2019273020632.hdf',
            'illuminations': 'D'}]
    '''

    fire_detected = False
    FD = FireDetector()
    tmp_file = None
    while not fire_detected:
        if not len(products):
            print(f"No products found in time range {start_time} - {end_time}")
            return False
        try:
            tmp_prod = products.pop(0)
            tmp_file = laads.download(tmp_prod["downloadsLink"], buffer_dir,
                                      token_file=token_file, replace=True,
                                      debug=debug)
            '''
            tmp_file = buffer_dir.joinpath(Path(
                "MOD09.A2016245.1455.006.2016247025036.hdf"))
            '''
            my_bands = (
                    # Red, Green, Blue, LWIR, SWIR,
                    "R1_500", "R4_500", "R3_500", "T31_1000", "T20_1000",
                    # Summed to simulate AVHRR channel 2
                    "R15_1000","R4_1000", "R3_1000", "R1_1000",
                    )
            data, info, geo = modis.get_modis_data(
                    l2_file=tmp_file, bands=my_bands)
            if debug:
                for i in range(len(data)):
                    print(enhance.array_stat(data[i]))
                    print(info[i])
                    print()

            VIS = np.sum(np.dstack(data[5:]), axis=2)/4
            print(enhance.array_stat(VIS))
            print(VIS.shape)
            data = list(data[:5])
            data.append(VIS)
            my_bands = list(my_bands[:5])
            my_bands.append("VIS")
            info = info[:5]
            info.append({"key":"VIS", "name":"Mean Visible Radiances"})

            geo = np.dstack(geo)
            yrange, xrange = geo_helpers.get_geo_range(
                    latlon=geo,
                    target_latlon=latlon,
                    dx_px=width_px,
                    dy_px=height_px,
                    from_center=True,
                    boundary_error=False,
                    debug=debug
                    )
            # Establish subgrids for the 1km and half-km data
            yctr,xctr = geo_helpers.get_closest_pixel(geo, latlon)
            yrange_hk = (yrange[0]*2, yrange[1]*2)
            xrange_hk = (xrange[0]*2, xrange[1]*2)
            data_1k = [A[yrange[0]:yrange[1],xrange[0]:xrange[1]]
                       for A in data[3:]]
            data_hk = [A[yrange_hk[0]:yrange_hk[1],xrange_hk[0]:xrange_hk[1]]
                       for A in data[:3]]
            geo = geo[yrange[0]:yrange[1],xrange[0]:xrange[1],:]

            # New center y,x for the subgrid
            yctr,xctr = geo_helpers.get_closest_pixel(geo, latlon)

            # if True, generate thermal and .5km truecolor RGBs of the scene
            if gen_nir_sample:
                fire_bands = ("LWIR", "SWIR", "VIS")
                pc_1k = PixelCat(list(map(np.copy, data_1k)), fire_bands)
                pc_1k.set_band(data_1k[1]-data_1k[0], "IRdiff")
                rgb_fire = pc_1k.get_rgb(
                        bands=["VIS", "IRdiff", "LWIR"],
                        recipes=[
                            lambda X: enhance.norm_to_uint(X, 256, np.uint8)
                            for i in range(3)],
                        show=False,
                        debug=debug
                        )

                truecolor_bands = ("Red", "Green", "Blue")
                pc_hk = PixelCat(list(map(np.copy, data_hk)), truecolor_bands)
                rgb_tc = pc_hk.get_rgb(
                        bands=truecolor_bands,
                        recipes=[
                            lambda X: enhance.norm_to_uint(
                                enhance.histogram_equalize(
                                        X, nbins=256, debug=debug)[0],
                                    256,
                                    np.uint8)
                                for i in range(3)],
                        show=False,
                        debug=debug
                        )
                timestr = tmp_prod["atime"].strftime( "%Y%m%d_%H%M")
                gp.generate_raw_image(
                        guitools.label_at_index(
                            rgb_fire, (yctr, xctr), color=(255,255,255)),
                        buffer_dir.joinpath(Path(
                            f"{timestr}_{satellite}_fire.png"))
                        )
                gp.generate_raw_image(
                        guitools.label_at_index(
                            rgb_tc, (2*yctr, 2*xctr), color=(255,0,255),
                            ),
                        buffer_dir.joinpath(Path(
                            f"{timestr}_{satellite}_tc.png"))
                        )
            if not FD.get_candidates(*data_1k, debug):
                continue
            if debug:
                print("Found candidate fire pixels at " + \
                        f"{tmp_prod['atime']}: {tmp_prod['downloadsLink']}")
        except Exception as e:
            raise e
        finally:
            if tmp_file and tmp_file.exists() and remove_hdfs:
                if debug: print(f"Removing {tmp_file.as_posix()}")
                tmp_file.unlink()

if __name__=="__main__":
    debug = True
    buffer_dir = Path("figures/fire_look")
    get_fire_passes(
            satellite="aqua",
            start_time=dt(year=2018, month=9, day=3, hour=1),
            end_time=dt(year=2018, month=9, day=5, hour=22),
            latlon=(-10,-60),
            #latlon=(47.05,8.3),
            #latlon=(38.07,-112.06),
            #latlon=(31, -83),
            buffer_dir=buffer_dir,
            gen_nir_sample=True,
            remove_hdfs=False,
            debug=debug
            )
