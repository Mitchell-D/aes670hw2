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

def merge_category_jsons(category_files:list):
    """
    """
    cat_jsons = [ json.load(catf.open("r")) for catf in category_files ]
    my_cats = {}
    for i in range(len(cat_jsons)):
        for j in range(len(cat_jsons[i])):
            cat_name = cat_jsons[i][j]["name"]
            if cat_name not in my_cats.keys():
                my_cats[cat_name] = set(map(tuple, cat_jsons[i][j]["pixels"]))
            else:
                my_cats[cat_name].update(
                        set(map(tuple, cat_jsons[i][j]["pixels"])))

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
    plt.savefig(fig_path, bbox_inches="tight")

def get_class_map(base_img:np.ndarray, $h)

if __name__=="__main__":
    # out-of-package helper script. Must be in same directory.
    region_pkl = Path("/home/krttd/uah/23.s/aes670/aes670hw2/data/pkls/" + \
            "my_region.pkl")
    if not region_pkl.exists():
        from restore_pkl import restore_pkl
        restore_pkl(region_pkl, debug=debug)

    color_map = {
            "cloud":"white", "water":"xkcd:royal blue",
            "mtn_veg":"xkcd:deep green", "clay":"xkcd:burnt orange",
            "grassland":"xkcd:khaki", "crops":"xkcd:bright green"
            }

    fig_path = Path("figures/classification/classes.png")

    region, info, _, _ = pkl.load(region_pkl.open("rb"))
    my_cats = merge_category_jsons([
        Path("data/pixel_selection/selection_1.json"),
        Path("data/pixel_selection/selection_2.json"),
        Path("data/pixel_selection/selection_3.json")
        ])

    classified, labels = classify.minimum_distance(np.dstack(region), my_cats)
    plot_classes(classified, fig_path, labels, color_map)
    gt.quick_render(enhance.norm_to_uint(classified, 256, np.uint8))
