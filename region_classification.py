import numpy as np
from pathlib import Path
import pickle as pkl
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from aes670hw2 import classify
from aes670hw2 import geo_plot as gp
from aes670hw2 import geo_helpers as gh
from aes670hw2 import guitools as gt
from aes670hw2 import enhance

region_cmap_xkcd = {
        "cloud":"white", "water":"xkcd:royal blue",
        "clay":"xkcd:burnt orange", "mtn_veg":"xkcd:deep green",
        "crops":"xkcd:bright green", "grassland":"xkcd:khaki",
        }
region_cmap_rgb = {"cloud":(0,255,255), "water":(61,126,255),
                 "mtn_veg":(13,89,69), "clay":(186,101,65),
                 "grassland":(255,195,77), "crops":(59,179,0) }
cmap_rgb = list(region_cmap_rgb.values())
cmap_xkcd = list(region_cmap_xkcd.values())[::-1]

def make_selection_rgb(region:np.ndarray, my_cats:dict, out_path:Path,
                       show:bool=False):
    """ Make an RGB of selected pixels with the above mappings """
    nc_rgb = np.dstack([ enhance.norm_to_uint(
        enhance.histogram_equalize(X, nbins=1000)[0], 256/3, np.uint8)
                        for X in region[-3:] ])
    for cat in my_cats.keys():
        for y,x in my_cats[cat]["pixels"]:
            nc_rgb[y,x] = region_cmap_rgb[cat]
    if show:
        gt.quick_render(nc_rgb)
    gp.generate_raw_image(nc_rgb, out_path)
    return nc_rgb

def merge_category_jsons(category_files:list):
    """
    """
    cat_jsons = [ json.load(catf.open("r")) for catf in category_files ]
    my_cats = {}
    for i in range(len(cat_jsons)):
        for j in range(len(cat_jsons[i])):
            cat = cat_jsons[i][j]["name"]
            if cat not in my_cats.keys():
                my_cats[cat] = {
                        "pixels":set(map(tuple, cat_jsons[i][j]["pixels"])),
                        "color":cat_jsons[i][j]["color"]
                        }
            else:
                new_px = set(map(tuple, cat_jsons[i][j]["pixels"]))
                my_cats[cat]["pixels"] = my_cats[cat]["pixels"].union(new_px)
    #categories = set([ c["name"] for c in [ cat for cat in cat_jsons ]])
    return my_cats

def plot_classes(class_array:np.ndarray, fig_path:Path, class_labels:list,
                 color_map:dict):
    """
    Plots an integer array mapping pixels to a list of class labels
    """
    colors = [ color_map[l] for l in class_labels ]
    print(colors)
    cmap, norm = matplotlib.colors.from_levels_and_colors(
            list(range(len(colors)+1)), colors)
    im = plt.imshow(class_array, cmap=cmap, norm=norm, interpolation="none")
    handles = [ Patch(label=class_labels[i], color=colors[i])
               for i in range(len(class_labels)) ]
    plt.legend(handles=handles)
    print(f"saving figure as {fig_path.as_posix()}")
    plt.tick_params(axis="both", which="both", labelbottom=False,
                    labelleft=False, bottom=False, left=False)
    fig = plt.gcf()
    fig.set_size_inches(16,9)
    plt.savefig(fig_path, bbox_inches="tight", dpi=100)

def get_pixel_selection_map(base_img:np.ndarray):
    pass

def get_latex_table(categories:dict):
    """
    Use the category json dicts to print a latex-formatted table containing
    the mean and standard deviation of each class for each band.
    """
    cat_strings = []
    for cat in my_cats.keys():
        cat_data = np.dstack(region)[tuple(zip(*my_cats[cat]["pixels"]))]
        row_str = cat
        for i in range(cat_data.shape[1]):
            cat += f" & {np.average(cat_data[:,i]):.3f} "
            cat += f"& {np.std(cat_data[:,i]):.3f} "
        cat += "\\\\"
        cat_strings.append(cat)
    return "\n".join(cat_strings)


if __name__=="__main__":
    # out-of-package helper script. Must be in same directory.
    region_pkl = Path("/home/krttd/uah/23.s/aes670/aes670hw2/data/pkls/" + \
            "my_region.pkl")
    fig_dir = Path("figures/classification")
    debug=True

    """ Load the region pkl """
    if not region_pkl.exists():
        from restore_pkl import restore_pkl
        restore_pkl(region_pkl, debug=debug)
    region, info, _, sunsat = pkl.load(region_pkl.open("rb"))


    '''
    """ Print information on and render each band """
    for i in range(len(region)):
        print(f"{enhance.array_stat(region[i])}")
        #gt.quick_render(enhance.norm_to_uint(r, 256, np.uint8))
        band_path = Path(f"band_{info['bands'][i]}.png")
        gp.generate_raw_image(
                enhance.norm_to_uint(region[i], 256, np.uint8),
                fig_dir.joinpath(band_path)
                )
    '''

    '''
    """ Generate heatmaps comparing bands """
    # [(0, 'M15'), (1, 'M14'), (2, 'M12'), (3, 'M10'), (4, 'M09'), (5, 'M07'),
    #  (6, 'M05'), (7, 'M04'), (8, 'M03')]
    band1 = 6
    band2 = 5
    print(list(enumerate(info['bands'])))
    hmap = enhance.get_heatmap(np.dstack((region[band1], region[band2])), 1024)
    hmap = enhance.linear_gamma_stretch(hmap)
    # Flip y axis due to python indexing standards
    hmap = hmap[:,::-1]
    print(enhance.array_stat(hmap))
    #gt.quick_render(hmap)
    plot_spec = {
        "title":"Normalized intensity heatmap: Band " + \
                f"{info['bands'][band1]} vs Band {info['bands'][band2]}",
        "cb_label":"Normalized Pixel Count (log scale)",
        "xlabel":f"Band {info['bands'][band1]}",
        "ylabel":f"Band {info['bands'][band2]}",
        "cb_orient":"horizontal"
        }
    gp.plot_heatmap(
            hmap,
            Path("figures/classification/heatmap_" + \
                    f"{info['bands'][band1]}+{info['bands'][band2]}.png"),
            plot_spec=plot_spec,
            show=False)
    '''

    # Get the pixel areas
    area = gh.cross_track_area(sunsat[2])*(.742*.776)

    #km_bands = ["M07", "M12"]
    km_bands = ["M10", "M15"]
    #'''
    """ Do k-means classification """
    tolerance = 1e-4
    class_num = 4
    km_inputs = [ region[info["bands"].index(b)] for b in km_bands ]
    km = classify.k_means(np.dstack(km_inputs), class_num,
                          tolerance=tolerance, debug=debug)
    pkl_name = Path(f"k-means_{'+'.join(km_bands)}_{class_num}cat.pkl")
    pkl_path = Path("data/pkls/").joinpath(pkl_name)
    with pkl_path.open("wb") as pklfp:
        print(f"Generating pickle at {pkl_path.as_posix()}")
        pkl.dump(km, pklfp)
    #'''

    #'''
    """ Plot k-means classes """
    # Yeah this is a shitty way to do it.
    #km_pkl = Path("data/pkls/k-means_M10+M15_6cat.pkl")
    km_pkl = Path("data/pkls/k-means_" + \
            f"{'+'.join(km_bands)}_{class_num}cat.pkl")
    with km_pkl.open("rb") as pklfp:
        km = pkl.load(pklfp)

    #print(km)
    labels = [f"Class {i+1}" for i in range(len(km))]
    #print(cmap_xkcd)
    #print(labels)
    cmap = {labels[i]:cmap_xkcd[i] for i in range(len(labels))}
    Y = np.zeros_like(region[0])
    km_arrays = np.dstack([region[info["bands"].index(b)]
                           for b in km_bands ])
    for c in range(class_num):
        # (px in category, band)
        tmp_vals = np.zeros((len(km[c]), len(km_bands)))
        for i in range(len(km[c])):
            y,x = km[c][i]
            tmp_vals[i,:] = km_arrays[y,x]
            Y[y,x] = c
        tmp_avg = np.average(tmp_vals, axis=0)
        tmp_std = np.std(tmp_vals, axis=0)
        # Format a latex-like table string for this class
        class_area = np.sum(area[np.where(Y==c)])
        class_line = labels[c] + " & " + " & ".join(
                [f"{len(km[c])} & {tmp_avg[j]:.3f} & {tmp_std[j]:.3f}"
                 for j in range(tmp_avg.size) ]) + f"& {class_area} \\"
        print(class_line)
    plot_classes(Y, fig_dir.joinpath(Path(km_pkl.stem+".png")),
                 labels, cmap)
    #'''

    #'''
    """ Merge all dictionaries from pixel selection sessions """
    my_cats = merge_category_jsons([
        #Path("data/pixel_selection/selection_1.json"),
        #Path("data/pixel_selection/selection_2.json"),
        #Path("data/pixel_selection/selection_3.json")
        Path("data/pixel_selection/selection_4.json"),
        Path("data/pixel_selection/selection_5.json")
        ])
    cat_px = { k:list(my_cats[k]["pixels"]) for k in my_cats.keys() }
    #print(get_latex_table(my_cats))
    #'''

    """ Generate a figure showing selected pixel locations """
    #fig_path = fig_dir.joinpath(Path("selections.png"))
    # make_selection_rgb(region, my_cats, fig_path)

    '''
    """ Do principle component analysis on all 9 bands """
    components = [0, 1, 2]
    print_table = True
    #pca = classify.pca(np.dstack(region)[:,:,-3:])
    pca = classify.pca(np.dstack(region), print_table=print_table)
    pca_rgb = np.dstack([
        enhance.norm_to_uint(pca[:,:,i], 256, np.uint8)
        for i in components ])
    pca_rgb_eq = np.dstack([
        enhance.norm_to_uint(enhance.histogram_equalize(
            pca[:,:,i], 256)[0], 256, np.uint8)
        for i in components ])
    gp.generate_raw_image(pca_rgb, Path("figures/classification/" + \
            f"pca_{'+'.join(list(map(str, components)))}.png"))
    gp.generate_raw_image(pca_rgb_eq, Path("figures/classification/" + \
            f"pca_{'+'.join(list(map(str, components)))}_eq.png"))
    '''


    '''
    """ Do maximum-likelihood classification with the pixel classes. """
    mlc, labels = classify.mlc(np.dstack(region), cat_px)
    print(f"Total Area: {np.sum(area):.2f}")
    for i in range(len(cat_px.keys())):
        cat_area = np.sum(area[np.where(mlc==i)])
        print(f"Category {labels[i]} Area: {cat_area:.2f}")
    gt.quick_render(mlc)
    plot_classes(mlc, Path("figures/classification/mlc_4+5.png"),
                 labels, region_cmap_xkcd)
    '''

    '''
    """ Do minimum-distance classification with the pixel classes. """
    # Get the area at each pixel
    fig_path = fig_dir.joinpath(Path("classes.png"))
    classified, labels = classify.minimum_distance(np.dstack(region), cat_px)
    print(f"Total Area: {np.sum(area):.2f}")
    for i in range(len(cat_px.keys())):
        cat_area = np.sum(area[np.where(classified==i)])
        print(f"Category {labels[i]} Area: {cat_area:.2f}")
    plot_classes(classified, fig_path, labels, region_cmap_xkcd)
    gt.quick_render(enhance.norm_to_uint(classified, 256, np.uint8))
    '''
