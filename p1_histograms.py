""" """
from pathlib import Path
import numpy as np
import pickle as pkl
from scipy.interpolate import interp1d
from pprint import pprint as ppt
import matplotlib.pyplot as plt

from aes670hw2 import geo_plot as gp
from aes670hw2 import enhance
from aes670hw2 import imstat
from aes670hw2 import geo_helpers

debug = False
show_images = False
img_pkl = Path("data/pkls/NP-IMG-philmont_20160712_2006.pkl")
mod_pkl = Path("data/pkls/NP-MOD-philmont_20160712_2006.pkl")

image_width = 640
image_height = 512
nbins = 2048
dpi = "figure"
target_latlon = (37.4, -107) # San Juan

""" Load the pickle created with viirs.generate_viirs_pkl() """
with img_pkl.open("rb") as pklfp:
    data, info, geo, sunsat = pkl.load(pklfp)

lat_range, lon_range = geo_helpers.get_geo_range(
        latlon=np.dstack((geo[0], geo[1])),
        target_latlon=target_latlon,
        dx_px=image_width*2,
        dy_px=image_height*2,
        from_center=True,
        debug=debug,
        )

data = [ X[lat_range[0]:lat_range[1],lon_range[0]:lon_range[1]] for X in data ]
geo = [ G[lat_range[0]:lat_range[1],lon_range[0]:lon_range[1]] for G in geo ]


""" Get Frequency histogram and an equalized array for each band """
histograms = []
for i in range(len(data)):
    histograms.append(imstat.do_histogram_analysis(
        data[i], nbins, equalize=True, debug=debug))
    histograms[i].update({"band":info["bands"][i]})

""" Generate a plot of frequency histograms  """
gp.plot_lines(
        domain=range(nbins),
        ylines=[h["hist"] for h in histograms],
        labels=[
            f"Band {h['band']}; " + \
                "$\mu = $"+str(gp.round_to_n(h["mean"], 5)) + \
                "    $\sigma = $"+str(gp.round_to_n(h["stddev"], 5))
            for h in histograms],
        plot_spec = {
            "title": "VIIRS Imagery band frequency histograms",
            "xlabel": "Brightness level bin",
            "ylabel": "Brightness level frequency",
            "line_width":1,
            },
        image_path=Path("figures/histogram_analysis/img_histograms.png"),
        show=show_images
        )

""" Generate a plot of cumulative frequencies """
gp.plot_lines(
        domain=range(nbins),
        ylines=[h["c_hist"] for h in histograms],
        labels=[ f"Band {h['band']}" for h in histograms],
        plot_spec = {
            "title": "VIIRS Imagery cumulative frequency histograms",
            "xlabel": "Brightness level bin",
            "ylabel": "Cumulative frequency",
            "line_width":1,
            },
        image_path=Path("figures/histogram_analysis/img_cumulative.png"),
        show=show_images
        )

""" Generate uncorrected and corrected natural color RGBs """
uncorrected = np.dstack([
    enhance.norm_to_uint(X, 256, cast_type=np.uint8) for X in data])

gp.generate_raw_image(
        uncorrected[::-1,::-1,:],
        Path("figures/histogram_analysis/naturalcolor_original.png")
        )

equalized = [
    enhance.norm_to_uint(Y, 256, cast_type=np.uint8)
    for Y in (hist["equalized"] for hist in histograms)
    ]

gp.generate_raw_image(
        np.dstack(equalized)[::-1,::-1,:],
        Path("figures/histogram_analysis/naturalcolor_histnorm.png")
        )

""" Get a frequency and cumulative histogram for the corected image """
equalized_hists = []
for i in range(len(equalized)):
    equalized_hists.append(imstat.do_histogram_analysis(
        equalized[i], nbins, equalize=False, debug=debug))
    equalized_hists[i].update({"band":info["bands"][i]})

""" Generate a plot of corrected frequency histograms """
gp.plot_lines(
        domain=range(nbins),
        ylines=[h["hist"] for h in equalized_hists],
        labels=[
            f"Band {h['band']}; " + \
                "$\mu = $"+str(gp.round_to_n(h["mean"], 5)) + \
                "    $\sigma = $"+str(gp.round_to_n(h["stddev"], 5))
            for h in histograms],
        plot_spec = {
            "title": "Histogram-Corrected VIIRS Imagery Band Frequency",
            "xlabel": "Brightness level bin",
            "ylabel": "Brightness level frequency",
            "line_width":1,
            },
        image_path=Path(
            "figures/histogram_analysis/img_histograms_correct.png"),
        show=show_images
        )

""" Generate a plot of cumulative frequencies """
gp.plot_lines(
        domain=range(nbins),
        ylines=[h["c_hist"] for h in equalized_hists],
        labels=[ f"Band {h['band']}" for h in equalized_hists],
        plot_spec = {
            "title": "Histogram-Corrected VIIRS Imagery " + \
                    "Band Cumulative Histograms",
            "xlabel": "Brightness level bin",
            "ylabel": "Cumulative frequency",
            "line_width":1,
            },
        image_path=Path(
            "figures/histogram_analysis/img_cumulative_correct.png"),
        show=show_images
        )

""" Load the pickle created with viirs.generate_viirs_pkl() """
with mod_pkl.open("rb") as pklfp:
    data, info, geo, sunsat = pkl.load(pklfp)

lat_range, lon_range = geo_helpers.get_geo_range(
        latlon=np.dstack((geo[0], geo[1])),
        target_latlon=target_latlon, # San Juan
        dx_px=image_width,
        dy_px=image_height,
        from_center=True,
        debug=debug,
        )

data = [ X[lat_range[0]:lat_range[1],lon_range[0]:lon_range[1]] for X in data ]
geo = [ G[lat_range[0]:lat_range[1],lon_range[0]:lon_range[1]] for G in geo ]


""" Get Frequency histogram and an equalized array for each band """
histograms = []
for i in range(len(data)):
    histograms.append(imstat.do_histogram_analysis(
        data[i], nbins, equalize=True, debug=debug))
    histograms[i].update({"band":info["bands"][i]})

""" Generate a plot of frequency histograms  """
gp.plot_lines(
        domain=range(nbins),
        ylines=[h["hist"] for h in histograms],
        labels=[
            f"Band {h['band']}; " + \
                "$\mu = $"+str(gp.round_to_n(h["mean"], 5)) + \
                "    $\sigma = $"+str(gp.round_to_n(h["stddev"], 5))
            for h in histograms],
        plot_spec = {
            "title": "VIIRS Moderate band frequency histograms",
            "xlabel": "Brightness level bin",
            "ylabel": "Brightness level frequency",
            "line_width":1,
            "dpi":dpi,
            },
        image_path=Path("figures/histogram_analysis/mod_histograms.png"),
        show=show_images
        )

""" Generate a plot of cumulative frequencies """
gp.plot_lines(
        domain=range(nbins),
        ylines=[h["c_hist"] for h in histograms],
        labels=[ f"Band {h['band']}" for h in histograms],
        plot_spec = {
            "title": "VIIRS Moderate cumulative frequency histograms",
            "xlabel": "Brightness level bin",
            "ylabel": "Cumulative frequency",
            "line_width":1,
            "dpi":dpi,
            },
        image_path=Path("figures/histogram_analysis/mod_cumulative.png"),
        show=show_images
        )

""" Get a frequency and cumulative histogram for the corected image """
equalized = [
    enhance.norm_to_uint(Y, 256, cast_type=np.uint8)
    for Y in (hist["equalized"] for hist in histograms)
    ]

equalized_hists = []
for i in range(len(equalized)):
    equalized_hists.append(imstat.do_histogram_analysis(
        equalized[i], nbins, equalize=False, debug=debug))
    equalized_hists[i].update({"band":info["bands"][i]})

""" Generate a plot of corrected frequency histograms """
gp.plot_lines(
        domain=range(nbins),
        ylines=[h["hist"] for h in equalized_hists],
        labels=[
            f"Band {h['band']}; " + \
                "$\mu = $"+str(gp.round_to_n(h["mean"], 5)) + \
                "    $\sigma = $"+str(gp.round_to_n(h["stddev"], 5))
            for h in histograms],
        plot_spec = {
            "title": "Histogram-Corrected VIIRS Moderate Band Frequency",
            "xlabel": "Brightness level bin",
            "ylabel": "Brightness level frequency",
            "line_width":1,
            "dpi":dpi,
            },
        image_path=Path(
            "figures/histogram_analysis/mod_histograms_correct.png"),
        show=show_images
        )

""" Generate a plot of cumulative frequencies """

gp.plot_lines(
        domain=range(nbins),
        ylines=[h["c_hist"] for h in equalized_hists],
        labels=[ f"Band {h['band']}" for h in equalized_hists],
        plot_spec = {
            "title": "Histogram-Corrected VIIRS Moderate " + \
                    "Band Cumulative Histograms",
            "xlabel": "Brightness level bin",
            "ylabel": "Cumulative frequency",
            "line_width":1,
            "dpi":dpi,
            },
        image_path=Path(
            "figures/histogram_analysis/mod_cumulative_correct.png"),
        show=show_images
        )
