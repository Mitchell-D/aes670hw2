
import numpy as np
from pathlib import Path

from aes670hw2 import geo_plot as gp
from aes670hw2 import enhance
from aes670hw2 import guitools as gt
from aes670hw2 import PixelCat

aster_path = Path("data/aster.img")
noise_path = Path("data/noise.img")
image_template = "figures/aster/aster_{label}.png"
figure_dir=Path("figures/aster")

width = 700
height = 700
show_images = True
use_scipy = True

image = []
with aster_path.open("rb") as afp:
    for i in range(width):
        image.append(list(afp.read(height)))
    image = np.asarray(image)
noise = []
with noise_path.open("rb") as nfp:
    for i in range(width):
        noise.append(list(nfp.read(height)))
    noise = np.asarray(noise)

""" Show original image phase """
image_phase = enhance.dft2D(image, use_scipy=True)
image_phase = np.roll(image_phase, int(image_phase.shape[0]/2), axis=0)
image_phase = np.roll(image_phase, int(image_phase.shape[1]/2), axis=1)
#print(f"Original phase")
#gt.quick_render(enhance.norm_to_uint(
#    np.log(1+np.abs(image_phase)), 256, np.uint8))
gp.generate_raw_image(
        enhance.norm_to_uint(
            np.log(1+np.abs(image_phase)), 256, np.uint8),
        figure_dir.joinpath("fft_centered_phase_original.png"))

""" Show noisy image phase """
noise_phase = enhance.dft2D(noise, use_scipy=True)
noise_phase = np.roll(noise_phase, int(noise_phase.shape[0]/2), axis=0)
noise_phase = np.roll(noise_phase, int(noise_phase.shape[1]/2), axis=1)
#print(f"Noise phase")
#gt.quick_render(enhance.norm_to_uint(
#    np.log(1+np.abs(noise_phase)), 256, np.uint8))
gp.generate_raw_image(
        enhance.norm_to_uint(
            np.log(1+np.abs(noise_phase)), 256, np.uint8),
        figure_dir.joinpath("fft_centered_phase_noise.png"),
        )

""" Show phase difference """
pdiff = image_phase-noise_phase
enhance.linear_gamma_stretch(pdiff)
print("Phase difference")
#gt.quick_render(enhance.norm_to_uint(
#    np.log(1+np.abs(pdiff)), 256, np.uint8))
gp.generate_raw_image(
        enhance.norm_to_uint(
            np.log(1+np.abs(pdiff)), 256, np.uint8),
        figure_dir.joinpath("fft_centered_phase_diff.png"),
        )

""" Show image reconstructed from noise difference """
diff_image = enhance.dft2D(pdiff, inverse=True, use_scipy=True)
#gt.quick_render(enhance.norm_to_uint(diff_image, 256, np.uint8))
gp.generate_raw_image(enhance.norm_to_uint(diff_image, 256, np.uint8),
                      figure_dir.joinpath("fft_noise_image.png"))

noise_phase = np.roll(noise_phase, -int(noise_phase.shape[0]/2), axis=0)
noise_phase = np.roll(noise_phase, -int(noise_phase.shape[1]/2), axis=1)

# Ask the user for a box filter selection
pc = PixelCat((image, noise), ("image", "noise"))
inbox = pc.get_box_filter("noise", outside=False, roll=True)
gt.quick_render(enhance.norm_to_uint(inbox, 256, np.uint8))
outbox = pc.get_box_filter("noise", outside=True, roll=True)
gt.quick_render(enhance.norm_to_uint(outbox, 256, np.uint8))

'''
rad_low, lpass = pc.get_filter("noise", low_pass=True)
rad_high, hpass = pc.get_filter("noise", low_pass=False)
edge_low, elow = pc.get_edge_filter("noise", high_frequency=True)
edge_high, ehigh = pc.get_edge_filter("noise", high_frequency=False)
'''
exit(0)
