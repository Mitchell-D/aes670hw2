
import numpy as np
from pathlib import Path
import sys

from aes670hw2 import geo_plot as gp
from aes670hw2 import enhance
from aes670hw2 import guitools as gt

aster_path = Path("data/aster.img")
image_template = "edges_{label}.png"

show_images = True
"""
Provide a PNG path as a terminal argument or input it after the
script is called. Otherwise comment this block and assign image_path.
"""

if len(sys.argv)==1:
    while True:
        uin = input(f"File path (or 'exit'): ")
        if uin=="exit":
            exit(0)
        image_path = Path(uin)
        if uin.exists():
            break
        print(f"Path doesn't exist: {image_path.as_posix()}")
else:
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Path doesn't exist: {image_path.as_posix()}")
        exit(0)
    if len(sys.argv) == 3:
        output

image = gt.png_to_np(image_path)
bands = [ image[:,:,i] for i in range(3) ]

# Show original image
if show_images: gt.quick_render(image)

sobel = []
roberts = []
multi = []
seq_multi = []
for i in range(len(bands)):
    sobel.append(enhance.norm_to_uint(enhance.sobel_edge(
        bands[i]), 256, np.uint8))
    roberts.append(enhance.norm_to_uint(enhance.roberts_edge(
        bands[i]), 256, np.uint8))
    multi.append(enhance.norm_to_uint(enhance.multi_edge(
        bands[i]), 256, np.uint8))
    seq_multi.append(enhance.norm_to_uint(enhance.multi_edge(
        bands[i], sequence=True), 256, np.uint8))

if show_images: gt.quick_render(np.dstack(sobel))
gp.generate_raw_image(np.dstack(sobel),
                      Path(image_template.format(label="sobel")))
if show_images: gt.quick_render(np.dstack(roberts))
gp.generate_raw_image(np.dstack(roberts),
                      Path(image_template.format(label="roberts")))
if show_images:gt.quick_render(np.dstack(multi))
gp.generate_raw_image(np.dstack(multi), Path(image_template.format(label="multi")))
if show_images: gt.quick_render(np.dstack(seq_multi))
gp.generate_raw_image(np.dstack(seq_multi), Path(image_template.format(label="sequential")))

#gp.generate_raw_image(image, image_path)
