""" cv2 GUI tools for selecting from a raster image. """
import numpy as np
from pathlib import Path
import cv2 as cv
from datetime import datetime as dt
from PIL import Image

def png_to_np(image_file:Path, debug=False):
    """
    Uses PIL to convert a png image to a numpy array, and returns the result.
    """
    X = np.asarray(Image.open(image_file.as_posix()))
    if debug:
        print(f"Loading image with shape {X.shape} of type {X.dtype}")
    return X

def select_region(X:np.ndarray, show_selection:bool=False, debug=False):
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

def label_at_index(X:np.ndarray, location:tuple, text:str=None, size:int=11,
                   text_offset:tuple=None, color:tuple=(0,0,255),
                   thickness:int=2, font_scale:float=1):
    """
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
                color:tuple=(0, 0, 255), thickness:int=2, normalize:bool=False):
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
    image_file = Path("figures/snpp-img-philmont-truecolor_20160712-2006.png")
    select_region(image_file)
