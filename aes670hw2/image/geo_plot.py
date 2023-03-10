"""
Methods for plotting geographic data on meshed axes
"""

import cartopy.crs as ccrs
import cartopy.feature as cf
import datetime as dt
import numpy as np
import math as m
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle as pkl
import xarray as xr
import metpy
import imageio

from pathlib import Path
from PIL import Image

from matplotlib.ticker import LinearLocator, StrMethodFormatter, NullLocator
from cartopy.mpl.gridliner import LongitudeLocator, LatitudeLocator

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    })


plot_spec_default = {
    "title":"",
    "title_size":8,
    "gridline_color":"gray",
    "fig_size":(16,9),
    "dpi":800,
    "borders":True,
    "border_width":0.5,
    "border_color":"black",
    "xlabel":"",
    "ylabel":"",
    "cb_orient":"vertical",
    "cb_label":"",
    "cb_tick_count":15,
    "cb_levels":80,
    "cb_size":.6,
    "cb_pad":.05,
    "cb_label_format":"{x:.1f}",
    "line_width":2,
    #"cb_cmap":"CMRmap",
    "cb_cmap":"jet",
    #"xtick_count":12,
    "xtick_size":8,
    #"ytick_count":12,
    }

def round_to_n(x, n):
    """
    Basic but very useful method to round a number to n significant figures.
    Placed here for rounding float values for labels.
    """
    try:
        return round(x, -int(m.floor(m.log10(x))) + (n - 1))
    except ValueError:
        return 0

def plot_lines(domain:np.ndarray, ylines:list, image_path:Path=None,
               labels:list=[], plot_spec={}, show:bool=False):
    """
    Plot a list of 1-d lines that share a domain and codomain.

    :@param domain: 1-d numpy array describing the common domain
    :@param ylines: list of at least 1 1-d array of data values to plot, which
            must be the same size as the domain array.
    :@param image_path: Path to the location to store the figure. If None,
            doesn't store an image.
    :@param labels: list of string labels to include in a legend describing
            each line. If fewer labels than lines are provided, the labels
            will apply to the first of the lines.
    :@param plot_spec: Dictionary of plot options see the geo_plot module
            for plot_spec options, but the defaults are safe.
    :@param show: if True, shows the image in the matplotlib Agg client.
    """
    # Merge provided plot_spec with un-provided default values
    old_ps = plot_spec_default
    old_ps.update(plot_spec)
    plot_spec = old_ps

    # Make sure all codomain arrays equal the domain size
    if not all((l.size == len(domain) for l in ylines)):
        raise ValueError(
                f"All codomain arrays must be the same size as the domain.")

    # Plot each
    fig, ax = plt.subplots()
    for i in range(len(ylines)):
        ax.plot(
                domain,
                ylines[i],
                label=labels[i] if len(labels) else "",
                linewidth=plot_spec.get("line_width")
                )

    ax.set_xlabel(plot_spec.get("xlabel"))
    ax.set_ylabel(plot_spec.get("ylabel"))
    ax.set_title(plot_spec.get("title"))
    if len(labels):
        plt.legend()
    if show:
        plt.show()
    if not image_path is None:
        print(f"Saving figure to {image_path}")
        fig.savefig(image_path, bbox_inches="tight", dpi=plot_spec.get("dpi"))


def basic_plot(x, y, image_path:Path, plot_spec:dict={}, scatter:bool=False):
    fig, ax = plt.subplots()
    if scatter:
        ax.scatter(x,y)
    else:
        ax.plot(x,y)

    if plot_spec.get("xtick_rotation"):
        plt.tick_params(axis="x", **{"labelrotation":plot_spec.get(
            "xtick_rotation")})
    if plot_spec.get("ytick_rotation"):
        plt.tick_params(axis="y", **{"labelrotation":plot_spec.get(
            "ytick_rotation")})

    plt.title(plot_spec.get("title"))
    plt.xlabel(plot_spec.get("xlabel"))
    plt.ylabel(plot_spec.get("ylabel"))
    print(f"Saving figure to {image_path}")
    plt.savefig(image_path, bbox_inches="tight")

def generate_raw_image(RGB:np.ndarray, image_path:Path, gif:bool=False,
                       fps:int=5):
    """
    Use imageio to write a raw full-resolution image to image_path
    :param RGB: (H, W, 3) array of RGB values to write as an image, or
            (H, W, T, 3) array of RGB values to write as a gif, if the gif
            attributeis True.
    :param image_path: Path of image file to write
    :param gif: if True, attempts to write as a full-resolution gif
    """
    if not gif:
        imageio.imwrite(image_path.as_posix(), RGB)
        print(f"Generated image at {image_path.as_posix()}")
        return
    RGB = np.moveaxis(RGB, 2, 0)
    '''
    if not RGB.dtype==np.uint8:
        print("converting")
        RGB = (np.moveaxis(RGB, 2, 0)*256).astype(np.uint8)
    '''
    imageio.mimwrite(uri=image_path, ims=RGB, format=".gif", fps=fps)
    print(f"Generated gif at {image_path.as_posix()}")

def geo_rgb_plot(R:np.ndarray, G:np.ndarray, B:np.ndarray, fig_path:Path,
                 xticks:np.ndarray=None, yticks:np.ndarray=None,
                 plot_spec:dict={}, animate:bool=False, extent:list=None):
    """
    Plot RGB values on a lat/lon grid specified by a ndarraay with 2d lat and
    lon coordinate meshes. If animate is False, R/G/B arrays must be 2d
    ndarrays with the same shape, or if animate is True, each of the R/G/B
    arrays must be 3d ndarrays with the same shape with the third axis
    representing a shared time dimension.

    :param data: ndarray. If animate is True, must be 3d with the third
            axes representing time, 2d if animate is False.
    :param lat: 2d ndarray representing latitude values on the data grid
    :param lon: 2d ndarray representing longitude values on the data grid
    :param fig_path: Path to the image generated by this method.
    :param plot_spec: Dictionary of valid plot settings. Options include:
    :param animate: If True and if the DataArray has a "time" dimension,
            animates along the time axis into a GIF, which is saved to
            fig_path.
    :param xticks: List of 2-tuples with the first element containing the int
            value of the pixel to label, and the second value providing the
            label as a string.
    :param yticks: List of 2-tuples with the first element containing the int
            value of the pixel to label, and the second value providing the
            label as a string.
    :param extent: Axes describes axes interval (ie wrt marker locations) and
            aspect ratio as a 4-tuple [left, right, bottom, top]
    """
    fig, ax = plt.subplots(figsize=plot_spec.get("fig_size"))
    #fig, ax = plt.subplots()

    if animate:
        data = np.stack((R,G,B), axis=3)
        if len(data.shape) != 4:
            raise ValueError("Data array must be 4d for animation, " + \
                    "with dimensions ordered like (x,y,time,color)")
        def anim(time):
            im.set_array(data[:,:,time])

        intv = plot_spec.get("anim_delay")
        ani = animation.FuncAnimation(
                fig=fig,
                func=anim,
                frames=data.shape[2],
                interval=intv if intv else 100
                )
        im = ax.imshow(
                data[:, :, 0],
                vmin=plot_spec.get("vmin"),
                vmax=plot_spec.get("vmax"),
                extent=extent # standard extent basis for tick labels
                )

    else:
        data = np.dstack((R,G,B))

        im = ax.imshow(
                data,
                vmin=plot_spec.get("vmin"),
                vmax=plot_spec.get("vmax"),
                extent=extent # standard extent basis for tick labels
                )

    # Set axes ticks if provided
    """
    x_locs, x_labels = (None, None) if xticks is None else list(zip(*xticks))
    y_locs, y_labels = (None, None) if yticks is None else list(zip(*yticks))
    if x_locs and x_labels:
        print(x_locs, x_labels)
        ax.set_xticks(x_locs)
        ax.set_xticklabels(x_labels)
    if y_locs and y_labels:
        ax.set_yticks(y_locs)
        ax.set_yticklabels(y_labels)
    """
    plt.yticks(fontsize=plot_spec.get("ytick_size"))
    plt.xticks(fontsize=plot_spec.get("xtick_size"))

    # markers are 2-tuple pixel coordinates x/y
    if plot_spec.get("markers"):
        marker_char = plot_spec.get("marcher_char")
        marker_char = "x" if not marker_char else marker_char
        marker_color = plot_spec.get("marcher_char")
        marker_color = "red" if not marker_color else marker_color
        for x,y in plot_spec.get("markers"):
            plt.plot(x, y, marker=marker_char, color=marker_color)

    if plot_spec.get("use_ticks") is False:
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())

    dpi = "figure" if not plot_spec.get("dpi") else plot_spec.get("dpi")
    if animate:
        if dpi != "figure":
            fig.set_dpi(dpi)
        ani.save(fig_path.as_posix(), dpi=dpi)
    else:
        plt.savefig(fig_path.as_posix(), dpi=dpi, bbox_inches='tight')

    myfig = plt.gcf()
    size = myfig.get_size_inches()
    print("data:",data.shape)
    print(f"size: {size}; dpi: {dpi}; resolution" + \
            f"{(dpi*size[0], dpi*size[1])}")
    print(f"Generated figure at: {fig_path}")

def geo_scalar_plot(data:np.ndarray, lat:np.ndarray, lon:np.ndarray,
                    fig_path:Path=None, plot_spec:dict={},
                    animate:bool=False):
    """
    Plot scalar values on a lat/lon grid specified by an xarray Dataset with
    2d coordinates "lat" and "lon". If the Dataset has "plot_spec" attribute
    mapping to a dictionary of supported key/value pairs, they will be used
    to configure the plot. If the Dataset has a "time" dimension and animate
    is True, a gif will be generated at fig_path.

    :param data: ndarray. If animate is True, must be 3d with the third
            axes representing time, 2d if animate is False.
    :param lat: 2d ndarray representing latitude values on the data grid
    :param lon: 2d ndarray representing longitude values on the data grid
    :param fig_path: Path to the image generated by this method.
    :param plot_spec: Dictionary of valid plot settings. Options include:
    :param animate: If True and if the DataArray has a "time" dimension,
            animates along the time axis into a GIF, which is saved to
            fig_path.
    """
    old_ps = plot_spec_default
    old_ps.update(plot_spec)
    plot_spec = old_ps
    fig = plt.figure(figsize=plot_spec.get("fig_size"))
    ax = plt.axes(projection=ccrs.PlateCarree())
    #fig, ax = plt.subplots(1, 1, subplot_kw={"projection":ccrs.PlateCarree()})
    cmap = 'jet' if not plot_spec.get('cb_cmap') else plot_spec.get('cb_cmap')


    if animate:
        if len(data.shape) != 3:
            raise ValueError("Data array must be 3-dimensional for animation")
        def anim(time):
            anim_mesh.set_array(data[:,:,time].flatten())

        intv = plot_spec.get("anim_delay")
        ani = animation.FuncAnimation(
                fig=fig,
                func=anim,
                frames=data.shape[2],
                interval=intv if intv else 100
                )
        anim_mesh = ax.pcolormesh(
                lon,
                lat,
                data[:, :, 0],
                vmin=np.amin(data),
                vmax=np.amax(data),
                #levels=plot_spec.get("cb_levels"),
                #add_colorbar=False,
                cmap=cmap,
                zorder=0,
                )

    if plot_spec.get("borders"):
        linewidth = plot_spec.get("border_width")
        if not linewidth:
            linewidth = 0.5
        b_color=plot_spec.get("border_color")
        ax.coastlines(zorder=1, linewidth=linewidth)
        states_provinces = cf.NaturalEarthFeature(category='cultural',
                name='admin_1_states_provinces_lines',
                scale='10m',
                linewidth=linewidth,
                facecolor='none')
        ax.add_feature(cf.BORDERS, linewidth=linewidth,
                       edgecolor=b_color, zorder=2)
        ax.add_feature(states_provinces, edgecolor=b_color, zorder=3)
        xlocs = plot_spec.get("xtick_count")
        ylocs = plot_spec.get("ytick_count")
        gl = ax.gridlines(
                draw_labels=True,
                linewidth=linewidth,
                xlocs=LongitudeLocator(xlocs) if xlocs else None,
                ylocs=LatitudeLocator(ylocs) if ylocs else None,
                color='gray',
                zorder=4,
                )
        gl.right_labels = False
        gl.top_labels = False
        gl.xlabel_style = {"size":plot_spec.get("xtick_size")}
        gl.ylabel_style = {"size":plot_spec.get("ytick_size")}

    if plot_spec.get("title"):
        ax.set_title(plot_spec.get("title"),
                     fontsize=plot_spec.get("title_size"))
    contour = ax.contourf(
            lon, lat, data,
            levels=plot_spec.get("cb_levels"),
            cmap=cmap,
            zorder=0,
            transform=ccrs.PlateCarree(),
            vmin=np.amin(data),
            vmax=np.amax(data),
            ) if not animate else anim_mesh
    cb_orient = plot_spec.get('cb_orient')
    orientation =  cb_orient if cb_orient else 'vertical'
    fmt = plot_spec.get("cb_label_format")
    pad = 0 if not plot_spec.get("cb_pad") else plot_spec.get("cb_pad")
    shrink = 0 if not plot_spec.get("cb_size") else plot_spec.get("cb_size")
    cbar = fig.colorbar(
            mappable = contour,
            orientation=orientation,
            format=StrMethodFormatter(fmt) if fmt else None,
            pad=pad,
            shrink=shrink,
            ticks=LinearLocator(plot_spec.get("cb_tick_count")),
            )
    cbar.set_label(plot_spec.get('cb_label'))
    dpi = "figure" if not plot_spec.get("dpi") else plot_spec.get("dpi")
    if animate:
        ani.save(fig_path.as_posix(), dpi=dpi)
    else:
        if fig_path:
            plt.savefig(fig_path.as_posix(), dpi=dpi, bbox_inches='tight')
        else:
            plt.show()
    print(f"Generated figure at: {fig_path}")
