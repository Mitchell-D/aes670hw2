
import numpy as np
from pathlib import Path

from aes670hw2 import geo_plot as gp
from aes670hw2 import enhance
from aes670hw2 import guitools as gt

aster_path = Path("data/aster.img")
image_template = "figures/aster/aster_{label}.png"

image = []
width = 700
height = 700
show_images = True
hist_equalize = True

with aster_path.open("rb") as afp:
    for i in range(width):
        image.append(list(afp.read(height)))
image = np.asarray(image)

""" Show original image """
print("Original image")
im_norm = enhance.norm_to_uint(np.copy(image), 256, np.uint8)
if show_images: gt.quick_render(im_norm)
gp.generate_raw_image(im_norm, Path(image_template.format(label="original")))

""" Apply roberts operator """
print("Roberts gradient operation")
if hist_equalize:
    roberts = enhance.norm_to_uint(enhance.histogram_equalize(
        enhance.roberts_edge(np.copy(image)), 256)[0], 256, np.uint8)
else:
    roberts = enhance.norm_to_uint(enhance.roberts_edge(
        np.copy(image)), 256, np.uint8)
if show_images: gt.quick_render(roberts)
gp.generate_raw_image(roberts, Path(image_template.format(label="roberts")))

""" Apply sobel operator """
print("Sobel edge kernels")
if hist_equalize:
    sobel = enhance.norm_to_uint(enhance.histogram_equalize(
        enhance.sobel_edge(np.copy(image)), 256)[0], 256, np.uint8)
else:
    sobel = enhance.norm_to_uint(enhance.sobel_edge(
        np.copy(image)), 256, np.uint8)
if show_images: gt.quick_render(sobel)
gp.generate_raw_image(sobel, Path(image_template.format(label="sobel")))

""" Apply each of the edge kernels independently """
print("Horizontal edge kernel")
if hist_equalize:
    H = enhance.norm_to_uint(enhance.histogram_equalize(
        enhance.kernel_convolve(np.copy(image), "horizontal"), 256)[0],
                             256, np.uint8)
else:
    H = enhance.norm_to_uint(enhance.kernel_convolve(
        np.copy(image), "horizontal"), 256, np.uint8)
if show_images:gt.quick_render(H)
gp.generate_raw_image(H, Path(image_template.format(label="horizontal")))
print("Vertical edge kernel")
if hist_equalize:
    V = enhance.norm_to_uint(enhance.histogram_equalize(
        enhance.kernel_convolve(np.copy(image), "vertical"), 256)[0],
                             256, np.uint8)
else:
    V = enhance.norm_to_uint(enhance.kernel_convolve(
        np.copy(image), "vertical"), 256, np.uint8)
if show_images:gt.quick_render(V)
gp.generate_raw_image(V, Path(image_template.format(label="vertical")))
print("Forward diagonal edge kernel")
if hist_equalize:
    D1 = enhance.norm_to_uint(enhance.histogram_equalize(
        enhance.kernel_convolve(np.copy(image), "diagonal_fwd"), 256)[0],
                              256,np.uint8)
else:
    D1 = enhance.norm_to_uint(enhance.kernel_convolve(
        np.copy(image), "diagonal_fwd"), 256, np.uint8)
if show_images:gt.quick_render(D1)
gp.generate_raw_image(D1, Path(image_template.format(label="diagonal1")))
print("Backward diagonal edge kernel")
if hist_equalize:
    D2 = enhance.norm_to_uint(enhance.histogram_equalize(
        enhance.kernel_convolve(np.copy(image), "diagonal_bck"), 256)[0],
                              256,np.uint8)
else:
    D2 = enhance.norm_to_uint(enhance.kernel_convolve(
        np.copy(image), "diagonal_bck"), 256, np.uint8)
if show_images:gt.quick_render(D2)
gp.generate_raw_image(D2, Path(image_template.format(label="diagonal2")))

""" Apply gradient with 4 cardinal edge kernels """
print("Multiple edge kernels, independently applied")
if hist_equalize:
    multi = enhance.norm_to_uint(enhance.histogram_equalize(
        enhance.multi_edge(np.copy(image), sequence=False), 256)[0],
                                 256, np.uint8)
else:
    multi = enhance.norm_to_uint(enhance.multi_edge(
        np.copy(image), sequence=False), 256, np.uint8)
if show_images:gt.quick_render(multi)
gp.generate_raw_image(multi, Path(image_template.format(label="multi")))

""" Apply sequential convolution with 4 cardinal edge kernels """
print("Multiple edge kernels, sequentially applied")
if hist_equalize:
    seq_multi = enhance.norm_to_uint(enhance.histogram_equalize(
        enhance.multi_edge(np.copy(image), sequence=True), 256)[0],
                                     256, np.uint8)
else:
    seq_multi = enhance.norm_to_uint(enhance.multi_edge(
        np.copy(image), sequence=True), 256, np.uint8)
if show_images: gt.quick_render(seq_multi)
gp.generate_raw_image(
        seq_multi, Path(image_template.format(label="sequential")))

#gp.generate_raw_image(image, image_path)
