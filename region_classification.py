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
from aes670hw2 import guitools as gt
from aes670hw2 import enhance

color_map = {
        "cloud":"white", "water":"xkcd:royal blue",
        "mtn_veg":"xkcd:deep green", "clay":"xkcd:burnt orange",
        "grassland":"xkcd:khaki", "crops":"xkcd:bright green"
        }
color_map_rgb = {"cloud":(0,255,255), "water":(61,126,255),
                 "mtn_veg":(13,89,69), "clay":(186,101,65),
                 "grassland":(255,195,77), "crops":(59,179,0) }


def make_selection_rgb(region:np.ndarray, my_cats:dict, show:bool=False):
    """ Make an RGB of selected pixels with the above mappings """
    nc_rgb = np.dstack([ enhance.norm_to_uint(
        enhance.histogram_equalize(X, nbins=1000)[0], 256/3, np.uint8)
                        for X in region[-3:] ])
    for cat in my_cats.keys():
        for y,x in my_cats[cat]["pixels"]:
            nc_rgb[y,x] = color_map_rgb[cat]
    if show:
        gt.quick_render(nc_rgb)
    gp.generate_raw_image(
            nc_rgb, Path("figures/classification/selections.png"))
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
    cmap, norm = matplotlib.colors.from_levels_and_colors(
            list(range(len(colors)+1)), colors)
    im = plt.imshow(class_array, cmap=cmap, norm=norm, interpolation="none")
    handles = [ Patch(label=class_labels[i], color=colors[i])
               for i in range(len(class_labels)) ]
    plt.legend(handles=handles)
    print(f"saving figure as {fig_path.as_posix()}")
    plt.savefig(fig_path, bbox_inches="tight")

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
    fig_path = Path("figures/classification/classes.png")
    debug=True

    """ Load the region pkl """
    if not region_pkl.exists():
        from restore_pkl import restore_pkl
        restore_pkl(region_pkl, debug=debug)
    region, info, _, _ = pkl.load(region_pkl.open("rb"))


    """ Print information on and render each band """
    for i in range(len(region)):
        print(f"{enhance.array_stat(region[i])}")
        #gt.quick_render(enhance.norm_to_uint(r, 256, np.uint8))
        gp.generate_raw_image( region[i],
                Path(f"figures/classification/band_{info['bands'][i]}.png"))

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

    """ Merge all dictionaries from pixel selection sessions """
    my_cats = merge_category_jsons([
        Path("data/pixel_selection/selection_1.json"),
        #Path("data/pixel_selection/selection_2.json"),
        Path("data/pixel_selection/selection_3.json")
        ])

    cat_px = { k:list(my_cats[k]["pixels"]) for k in my_cats.keys() }
    pca = classify.pca(region)


    '''
    # Make an RGB showing sample pixel locations
    # make_selection_rgb(region, my_cats)
    mlc, labels = classify.mlc(np.dstack(region), cat_px)
    gt.quick_render(mlc)
    #gp.generate_raw_image(mlc, )
    plot_classes(mlc, Path("figures/classification/mlc_1+3.png"),
                 labels, color_map)
    '''

    '''
    """ Do minimum-distance classification with the pixel classes. """
    classified, labels = classify.minimum_distance(
            np.dstack(region),{k:cat_px[k]["pixels"] for k in cat_px.keys()})
    plot_classes(classified, fig_path, labels, color_map)
    gt.quick_render(enhance.norm_to_uint(classified, 256, np.uint8))
    '''
