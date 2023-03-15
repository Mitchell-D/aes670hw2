""" cv2 GUI tools for selecting from a raster image. """

import numpy as np
from pathlib import Path
import cv2 as cv
import itertools
from datetime import datetime as dt
from PIL import Image
from pprint import pprint as ppt


def png_to_np(image_file:Path, debug=False):
    """
    Uses PIL to convert a png image to a numpy array, and returns the result.
    """
    X = np.asarray(Image.open(image_file.as_posix()))
    if debug:
        print(f"Loading image with shape {X.shape} of type {X.dtype}")
    return X

def get_category_series(X:np.ndarray, cat_count:int, category_names:list=None,
                        show_pool=False, debug=False):
    """
    High-level UI method for selecting pixels for multiple categories using
    a cv2 window, which are returned as a list of (y,x) coordinates in a
    dictionary labeled with either the category_names parameter, or
    user-provided names by default.

    For each of cat_count categories, this method will launch a cv2 window
    enabling the user to click individual pixels to include their coordinates
    in the list dict entry for that category.

    Useful information on current selections are printed if debug=True.

    :@param X: (M,N,3) RGB array. The array must strictly be 3d RGB, so if
            you're working with a grayscale (2d) array, duplicate along axis 2.
    :@param cat_count: Number of categories to ask the user to populate.
    :@param category_names: Rather than asking the user for category names
            after they select them,
    :@param show_pool: If True, once the user finishes selecting pixels,
            a square image containing the pixels they selected will be
            rendered, which can help indicate the homogeneity of the category.
    :@param debug: if True, prints useful information to the terminal as the
            user selects pixels.
    """
    color_loop = itertools.cycle([
        color[:3] for color in itertools.permutations((256, 256, 128, 0))
        ])
    cats = []
    if category_names and not len(set(category_names))==cat_count:
        raise ValueError(f"If category names are provided, there must be " + \
                f" the same number of categories as cat_count ({cat_count})")
    for i in range(cat_count):
        new_cat = { "pixels":None, "name":None, "color":None}
        if not category_names is None:
            cname = category_names[i]
            print(f"\n\033[93mSelect pixels for category \033[0m" + \
                    f"\033[1m{cname}\033[0m\033[93m...\033[0m")
        else:
            cname = input(f"\033[93mName for category {i+1}: \033[0m")
            choice = ""
            if cname in [ cat["name"] for cat in cats ]:
                while choice.lower() not in ("y","n"):
                    print(f"\033[91mClass {cname} is already loaded. " + \
                            "Replace?\033[93m")
                    choice = input("\n(Y/n): \033[0m")
                if choice=="n":
                    continue
        new_cat["name"] = cname
        new_cat["color"] = next(color_loop)
        new_cat["pixels"] = get_category(X, fill_color=new_cat["color"],
                              show_pool=show_pool, debug=debug)
        cats.append(new_cat)
    return cats

def get_category(X:np.ndarray, fill_color:tuple=(0,255,255),
                 show_pool:bool=True, debug=False):
    """
    Launch a GUI window for the user to choose pixels on an image that belong
    to a category, and return a list of array indeces corresponding to their
    selection.

    :@param X: (M,N,3) RGB array from which to pick pixels
    :@param fill_color: Color to mark pixels that have already been selected.
    :@param show_pool: If True, once the user finishes selecting pixels,
            a square image containing the pixels they selected will be
            rendered, which can help indicate the homogeneity of the category.
    :@param debug: if True, prints information about pixels to stdout as the
            user selects them.
    """
    # Without copying, this edits the provided ndarray inplace.
    Xnew = np.copy(X)
    coords = []
    fill_color = np.asarray(fill_color)
    # Calculate RGB standard deviation by taking the sum of all bands' stddevs.
    array_std = sum([ np.std(X[:,:,i]) for i in range(X.shape[2]) ])
    def mouse_callback(event,x,y,flags,param):
        if event == cv.EVENT_LBUTTONDOWN:
            coords.append((y, x))
            Xnew[y,x,:] = fill_color
            std = 0
            for i in range(X.shape[2]):
                std += np.std([ X[y,x,i] for y,x in coords ])
            if debug:
                print(f"\033[92mNew px:\033[0m \033[1m({y}, {x})\t" + \
                    f"\033[92mTotal:\033[0m \033[1m{len(set(coords))}\t" + \
                    f"\033[92mStdDev:\033[0m \033[1m{std:.4f}\033[0m")

    cv.namedWindow("catselect") # Can be resized
    cv.setMouseCallback("catselect", mouse_callback)

    print("\033[1m\033[91mPress 'q' key to exit.\033[0m")
    print("\033[1m\033[91mPress 'x' key to cancel previous pixel.\033[0m")
    while True:
        cv.imshow('catselect', Xnew)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("x"):
            if not len(coords):
                continue
            y, x = coords.pop()
            Xnew[y,x,:] = X[y,x,:]

    cv.destroyAllWindows()
    if show_pool and len(coords):
        show_pixel_pool(np.stack([ X[y,x] for y,x in set(coords) ], axis=0),
                        debug=debug)

    return list(coords)

def trackbar_select(X:np.ndarray, func, label:"", resolution:int=256,
                    debug=False):
    """
    Use a trackbar to select an integer value in [0,255] which is used to
    change and re-render the provided array with the modified values.

    When the user hits "q", the selected values for each function are
    returned as a tuple in the same order that

    If the trackbar associated with one of the functions is modified,
    the function is called with only the
    """
    wname = "catselect"
    # make a new copy of X that can be modified
    Xnew = np.copy(X)
    if not type(func)==type(lambda m:1):
        raise ValueError(f"func argument must be a function with 2 " + \
                "arguments for an array X and value a in [0,resolution-1]")

    if debug: print(f"\033[92mSelecting for array \033[96m{label}\033[0m")
    # Ordered list of selections from the user
    selection = 0
    def callback(user_value:int):
        """
        Applies user-defined function to Xnew with the user's selected value.
        """
        selection = user_value
        cv.imshow(wname, func(np.copy(X), selection))

    cv.namedWindow(wname)
    cv.createTrackbar(label, wname, 0, resolution-1,
                      lambda v: callback(v,selection))
    cv.imshow("catselect", Xnew)

    #cv.waitKey(1) & 0xFF == ord("q"):
    #    break
    cv.waitKey(0)
    cv.destroyAllWindows()
    print(selection)
    return selection

def region_select(X:np.ndarray, show_selection:bool=False, debug=False):
    """
    Render the image in a full-resolution cv2 and call selectROI
    """
    #im = cv.imread(cv.samples.findFile(image_file.as_posix()))
    if len(X.shape) == 2:
        X = cv.cvtColor(X, cv.COLOR_GRAY2BGR)
    else:
        X = cv.cvtColor(X, cv.COLOR_RGB2BGR)

    x,y,w,h = cv.selectROI("select_window", X,
                           fromCenter=False, showCrosshair=False)
    if debug:
        print(f"Selected pixel boundary boxes y:({y}, {y+h}) x:({x}, {x+w})")
    subregion = X[y:y+h, x:x+w]
    cv.destroyWindow("select_window")

    if show_selection:
        cv.imshow("subregion", subregion)
        cv.waitKey(-1)
        cv.destroyWindow("subregion")

    return ((y,y+h), (x,x+w))

def quick_render(X:np.ndarray):
    """
    Method for rapidly rendering a 2d or 3d array as a sanity check.
    """
    if len(X.shape) == 2:
        X = np.dstack((X, X, X))
    elif len(X.shape) == 3:
        assert X.shape[2]==3
    cv.namedWindow("quick render")
    print("\033[1m\033[91mPress 'q' key to exit\033[0m")
    while True:
        cv.imshow("quick render", X)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    cv.destroyAllWindows()

def show_pixel_pool(pixels:np.ndarray, debug=False):
    """
    render a square array populated with as many of the selected pixels as
    possible, exit the window when the user presses "q".

    :param pixels: (N, 3)-shaped ndarray of RGB pixel values.
    """
    if not len(pixels.shape)==2 and pixels.shape[1]==3:
        raise ValueError(f"Pixel array must have shape (N,3) for N pixels.")
    if not len(pixels):
        if debug: print(f"No pixels provided; not showing pixel pool.")
        return
    side_length = int((pixels.shape[0])**(1/2) + 1)
    px_count = side_length**2
    pixels = np.concatenate((pixels,pixels[:px_count-len(pixels)]), axis=0)
    pool = np.dstack([
        np.reshape(pixels[:,i], (side_length, side_length))
        for i in range(3) ])
    cv.namedWindow("pixel pool")
    if debug:
        print(f"\033[1mDisplaying selected pixel colors\033[0m")
    print("\033[1m\033[91mPress 'q' key to exit\033[0m")
    while True:
        cv.imshow('pixel pool', pool)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cv.destroyAllWindows()

def label_at_index(X:np.ndarray, location:tuple, text:str=None, size:int=11,
                   text_offset:tuple=None, color:tuple=(0,0,255),
                   thickness:int=2, font_scale:float=1):
    """
    Mark an index with an 'X', and optionally include a text label.

    :@param X: (M,N,3) RGB numpy array on which to rasterize the label.
    :@param text: Text to label the location (NOT IMPLEMENTED)
    :@param size: Vertical and horizontal pixel size of the 'X' mark.
    :@param text_offset: Pixel offset of text corner (NOT IMPLEMENTED)
    :@param color: color of text and mark.
    :@param thickness: pixel width of mark.
    :@param font_scale: Coefficient multiplyer of standard font size
            (NOT IMPLEMENTED)
    """
    mask = X.mask if np.ma.is_masked(X) else None
    # Round up to the nearest odd number
    size += not size%2
    # center x and y pixel
    c_y, c_x = location
    offset = (size-1)/2

    # Get X corner pixel locations in (y,x) coordinates
    TL, BR = ((int(c_y-offset), int(c_x-offset)),
              (int(c_y+offset), int(c_x+offset)))
    BL, TR = ((int(c_y+offset), int(c_x-offset)),
              (int(c_y-offset), int(c_x+offset)))

    # Must flip coordinates for cv
    img = cv.line(X, TL[::-1], BR[::-1], color, thickness)
    img = cv.line(X, BL[::-1], TR[::-1], color, thickness)
    '''
    if not text is None:
        if text_offset is None:
            # place text under point by default. text_offset is in (y,x) coords
            # here despite cv using (x,y) coords.
            text_offset = (0,-1*size)
        dy, dx = text_offset
        font = cv.FONT_HERSHEY_SIMPLEX
        img = cv.putText(img, c_y)
    '''
    return np.ma.array(img, mask) if not mask is None else img

def rect_on_rgb(X:np.ndarray, yrange:tuple, xrange:tuple,
                color:tuple=(0, 0, 255), thickness:int=2,
                normalize:bool=False):
    """
    Use cv2 to draw a colored pixel rectangle on the array at native res.
    For simplicity, this method requires that the array is uint8-normalized.

    :@param X: 2d or 3d numpy image array to draw a rectangle to.
    :@param yrange: (min, max) integer pixel coordinates of rectangle on
            axis 0, which typically corresponds to vertical.
    :@param xrange: (min, max) integer pixel coordinates of rectangle on
            axis 1, which typically corresponds to horizontal.
    :@param color: uint8 (0-255) color value of the rectangle.
    :@param thickness: Rectangle line width.
    :@param normalize: If True, linear-normalizes the image to uint8 if not
            already uint8 instead of raising a ValueError.
    """
    mask = X.mask if np.ma.is_masked(X) else None
    if X.dtype!=np.uint8:
        if normalize:
            X = enhance.norm_to_uint(X, resolution=256, cast_type=np.uint8)
        else:
            raise ValueError(
                f"Array must be type uint8. Pass normalize=True to convert.")

    TL, BR = zip(yrange, xrange)
    # Must flip coordinates for cv
    Xrect = cv.rectangle(X, TL[::-1], BR[::-1], color, thickness)
    return np.ma.array(Xrect, mask) if not mask is None else Xrect

if __name__ == "__main__":
    debug = True
    image_dir = Path("/home/krttd/uah/23.s/aes670/aes670hw2/figures/test/")
    image_file = image_dir.joinpath("region.png")
    categories = ("land", "water", "cloud")
    image = png_to_np(image_file)

    ppt(get_category_series(
            X=image,
            cat_count=len(categories),
            category_names=categories,
            show_pool=True,
            debug=debug,
            ))
