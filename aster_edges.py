
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

with aster_path.open("rb") as afp:
    for i in range(width):
        image.append(list(afp.read(height)))
image = np.asarray(image)

# Show original image
im_norm = enhance.norm_to_uint(image, 256, np.uint8)
if show_images: gt.quick_render(im_norm)
gp.generate_raw_image(im_norm, Path(image_template.format(label="original")))

# Apply roberts operator
roberts = enhance.norm_to_uint(enhance.roberts_edge(image), 256, np.uint8)
if show_images: gt.quick_render(roberts)
gp.generate_raw_image(roberts, Path(image_template.format(label="roberts")))

# Apply sobel operator
sobel = enhance.norm_to_uint(enhance.sobel_edge(image), 256, np.uint8)
if show_images: gt.quick_render(sobel)
gp.generate_raw_image(sobel, Path(image_template.format(label="sobel")))

# Apply gradient with 4 cardinal edge kernels
multi = enhance.norm_to_uint(enhance.multi_edge(
    image, sequence=False), 256, np.uint8)
if show_images:gt.quick_render(multi)
gp.generate_raw_image(multi, Path(image_template.format(label="multi")))

# Apply sequential convolution with 4 cardinal edge kernels
seq_multi = enhance.norm_to_uint(enhance.multi_edge(
    image, sequence=True), 256, np.uint8)
if show_images: gt.quick_render(seq_multi)
gp.generate_raw_image(
        seq_multi, Path(image_template.format(label="sequential")))

#gp.generate_raw_image(image, image_path)
