
import numpy as np
from pathlib import Path

from aes670hw2 import geo_plot as gp

aster_path = Path("data/aster.img")
image_path = Path("figures/aster/original.png")

image = []
width = 700
height = 700
with aster_path.open("rb") as afp:
    for i in range(width):
        image.append(list(afp.read(height)))
image = np.asarray(image)

gp.generate_raw_image(image, image_path)
