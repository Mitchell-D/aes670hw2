
import numpy as np
from pathlib import Path

from aes670hw2 import geo_plot as gp
from aes670hw2 import enhance
from aes670hw2 import guitools as gt
from aes670hw2 import PixelCat

def my_implementation(image, show:bool=True, gen_images:bool=True,
                      figure_dir:Path=None):
    """ Show my fft implementation and inversion with subsetted array """
    if gen_images: assert not figure_dir is None
    next2power = lambda x: next(( i for i in range(100) if 2**i>x ))
    new_height = 2**(next2power(image.shape[0])-1)
    new_width = 2**(next2power(image.shape[1])-1)
    resized = image[:new_height,:new_width]
    if show:
        print(f"Displaying resized array")
        gt.quick_render(enhance.norm_to_uint(resized, 256, np.uint8))
    if gen_images:
        gp.generate_raw_image(
                enhance.norm_to_uint(resized, 256, np.uint8),
                figure_dir.joinpath("fft_resized_input.png"))
    resized_fft_phase = enhance.dft2D(resized, use_scipy=False)
    if show:
        print(f"Displaying resized phase")
        gt.quick_render(enhance.norm_to_uint(
            np.log(1+resized_fft_phase)+1, 256, np.uint8))
    if gen_images:
        gp.generate_raw_image(
                enhance.norm_to_uint(
                    np.log(1+resized_fft_phase), 256, np.uint8),
                figure_dir.joinpath("fft_resized_phase.png"))

    resized_fft_recon = enhance.dft2D(resized_fft_phase, inverse=True,
                                      use_scipy=True)
    if show:
        print(f"Displaying resized reconstruction")
        gt.quick_render(enhance.norm_to_uint(resized_fft_recon, 256, np.uint8))
    if gen_images:
        gp.generate_raw_image(
                enhance.norm_to_uint(resized_fft_recon, 256, np.uint8),
                figure_dir.joinpath("fft_resized_reconstruction.png"))

    """ Show my fft implementation and inversion with padded array """
    new_height = 2**next2power(image.shape[0])
    new_width = 2**next2power(image.shape[1])
    padded = np.pad(image, ((0,new_height-image.shape[0]),
                            (0,new_width-image.shape[1])))
    if show:
        print(f"Displaying padded array")
        gt.quick_render(enhance.norm_to_uint(padded, 256, np.uint8))
    if gen_images:
        gp.generate_raw_image(
                enhance.norm_to_uint(resized_fft_recon, 256, np.uint8),
                figure_dir.joinpath("fft_padded_input.png"))
    padded_fft_phase = enhance.dft2D(padded, use_scipy=False)
    if show:
        print(f"Displaying padded phase")
        gt.quick_render(enhance.norm_to_uint(
            np.log(1+padded_fft_phase)+1, 256, np.uint8))
    if gen_images:
        gp.generate_raw_image(
                enhance.norm_to_uint(
                    np.log(1+padded_fft_phase), 256, np.uint8),
                figure_dir.joinpath("fft_padded_phase.png"))
    padded_fft_recon = enhance.dft2D(padded_fft_phase, inverse=True,
                                     use_scipy=True)
    if show:
        print(f"Displaying padded reconstruction")
        gt.quick_render(enhance.norm_to_uint(padded_fft_recon, 256, np.uint8))
    if gen_images:
        gp.generate_raw_image(
                enhance.norm_to_uint(padded_fft_recon, 256, np.uint8),
                figure_dir.joinpath("fft_padded_reconstruction.png"))

if __name__=="__main__":
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

    #my_implementation(image, figure_dir=figure_dir,
    #                  gen_images=True, show=False)

    gp.generate_raw_image(
            enhance.norm_to_uint(image, 256, np.uint8),
            figure_dir.joinpath("fft_input_original.png"))
    gp.generate_raw_image(
            enhance.norm_to_uint(noise, 256, np.uint8),
            figure_dir.joinpath("fft_input_noise.png"))

    """ Show scipy fft implementation and inversion """
    image_phase = enhance.dft2D(image, use_scipy=True)
    #print("Showing image phase space")
    #gt.quick_render(enhance.norm_to_uint(
    #    np.log(1+np.abs(image_phase)), 256, np.uint8))
    gp.generate_raw_image(
            enhance.norm_to_uint(
                np.log(1+np.abs(image_phase)), 256, np.uint8),
            figure_dir.joinpath("fft_phase_original.png"))

    """ """
    noise_phase = enhance.dft2D(noise, use_scipy=True)
    #print("Showing noisy phase space")
    #gt.quick_render(enhance.norm_to_uint(
    #    np.log(1+np.abs(noise_phase)), 256, np.uint8))
    gp.generate_raw_image(
            enhance.norm_to_uint(
                np.log(1+np.abs(noise_phase)), 256, np.uint8),
            figure_dir.joinpath("fft_phase_noise.png"),
            )

    pdiff = image_phase-noise_phase
    enhance.linear_gamma_stretch(pdiff)
    #print("showing phase difference")
    #gt.quick_render(enhance.norm_to_uint(
    #    np.log(1+np.abs(pdiff)), 256, np.uint8))
    gp.generate_raw_image(
            enhance.norm_to_uint(
                np.log(1+np.abs(pdiff)), 256, np.uint8),
            figure_dir.joinpath("fft_phase_diff.png"),
            )

    pc = PixelCat((image, noise), ("image", "noise"))
    rad_low, lpass = pc.get_filter("noise", low_pass=True)
    rad_high, hpass = pc.get_filter("noise", low_pass=False)
    edge_low, elow = pc.get_edge_filter("noise", high_frequency=True)
    edge_high, ehigh = pc.get_edge_filter("noise", high_frequency=False)

    print("Showing low-pass filter selected")
    gt.quick_render(lpass)
    print("Showing high-pass filter selected")
    gt.quick_render(hpass)

    gp.generate_raw_image(
            enhance.norm_to_uint(lpass, 256, np.uint8),
            figure_dir.joinpath("fft_filtered_radiuslow.png"))
    gp.generate_raw_image(
            enhance.norm_to_uint(hpass, 256, np.uint8),
            figure_dir.joinpath("fft_filtered_radiushigh.png"))
    gp.generate_raw_image(
            enhance.norm_to_uint(elow, 256, np.uint8),
            figure_dir.joinpath("fft_filtered_edgelow.png"))
    gp.generate_raw_image(
            enhance.norm_to_uint(ehigh, 256, np.uint8),
            figure_dir.joinpath("fft_filtered_edgehigh.png"))

    noise = enhance.norm_to_uint(noise, 256, np.uint8)
    noise[np.where(noise==0)] = ehigh[np.where(noise==0)]
    noise[np.where(noise==255)] = ehigh[np.where(noise==255)]
    #for i in noise.shape[0]:
    #    for j in noise.shape[1]:
    #        if noise[i,j] >
    print(noise)
    print(noise.shape)
    print(noise.dtype)
    gt.quick_render(noise)

    gp.generate_raw_image(
            noise,
            figure_dir.joinpath("fft_filtered_reconstruction.png"))

