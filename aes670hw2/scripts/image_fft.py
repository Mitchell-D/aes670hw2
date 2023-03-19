"""
Quick script to input a png image and try different frequency radius
values for low and high pass filters.
"""

import numpy as np
import sys
from pathlib import Path

from aes670hw2 import geo_plot as gp
from aes670hw2 import enhance
from aes670hw2 import guitools as gt
from aes670hw2 import PixelCat

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

# Convert the image to a numpy (M,N,3) array
image = gt.png_to_np(image_path)
image_phase = enhance.dft2D(image, use_scipy=True)

pc = PixelCat([image[:,:,i] for i in range(3)], ("R", "G", "B"))
low_pass = []
low_radii = []
high_pass = []
high_radii = []
for b in pc.bands:
    rad_low, lpass = pc.get_filter(b, low_pass=True)
    low_pass.append(lpass)
    low_radii.append(rad_low)
    #rad_high, hpass = pc.get_filter(b, low_pass=False)
    #high_pass.append(hpass)
    #high_radii.append(rad_high)

image_lp = np.dstack([enhance.norm_to_uint(X, 256, np.uint8)
                      for X in low_pass])
#image_hp = np.dstack([enhance.norm_to_uint(X, 256, np.uint8)
#                      for X in high_pass])

print(f"Showing low-pass filtered image with RGB radii {low_radii}")
gt.quick_render(image_lp)
#print(f"Showing high-pass filtered image with RGB radii {high_radii}")
#gt.quick_render(image_hp)

'''
gp.generate_raw_image(
        enhance.norm_to_uint(image_lp, 256, np.uint8),
        figure_dir.joinpath("fft_filtered_lowpass.png"))

gp.generate_raw_image(
        enhance.norm_to_uint(image_hp, 256, np.uint8),
        figure_dir.joinpath("fft_filtered_highpass.png"))
'''
