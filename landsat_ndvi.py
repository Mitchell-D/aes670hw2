
import cv2 as cv
import numpy as np
from pathlib import Path

from aes670hw2 import PixelCat
from aes670hw2 import enhance
from aes670hw2 import geo_plot as gp
from aes670hw2 import guitools as gt

#'''
def sk_cloud_mask(B1:np.ndarray, B3:np.ndarray, t1_thresh=.15, t2_thresh=1.6):
    t1 = B1>t1_thresh
    B1[np.where(B1==0)] = 1
    B3[np.where(B1==0)] = 0
    t2 = (B3/B1)<t2_thresh
    return np.logical_and(t1, t2)
#'''

landsat_paths = {
        "B1":Path("data/landsat/p125r48_3m19790311_01.tif"),
        "B2":Path("data/landsat/p125r48_3m19790311_02.tif"),
        "B3":Path("data/landsat/p125r48_3m19790311_03.tif"),
        "B4":Path("data/landsat/p125r48_3m19790311_04.tif"),
        }

fig_dir = Path("figures/landsat")

# All 3 bands in the data are identical
bands, data = zip(*[(k,np.asarray(cv.imread(p.as_posix()), dtype=float)[:,:,0])
                    for k,p in landsat_paths.items() ])
#for X in data:
#    print(enhance.array_stat(X))
#print([ enhance.array_stat(d) for d in data ])
pc = PixelCat(data,bands)

# Image enhancement parameters
value, saturation = (255, 255)
hsv_range = (0,125)
ndvi_scale_offset = .4
ndvi_scale_stretch = .6
#ndvi_scale_offset = 1
#ndvi_scale_stretch = 0

#print(enhance.array_stat(pc.band("B4")))
#print(enhance.array_stat(pc.band("B1")))
#print(enhance.array_stat(pc.band("B1")+pc.band("B4")))
#print(enhance.array_stat(pc.band("B1")-pc.band("B4")))

""" Get the NDVI """
ndvi_denom = pc.band("B3")+pc.band("B2")
ndvi_num = pc.band("B3")-pc.band("B2")
ndvi = np.zeros_like(pc.band("B4"))
valid_px = np.where(ndvi_denom != 0)
ndvi[valid_px] = (ndvi_num[valid_px]/ndvi_denom[valid_px])
pc.set_band(ndvi, "NDVI")
ndvi_enh = enhance.linear_gamma_stretch(np.copy(ndvi), gamma=1.1)

""" Get the NDWI """
ndwi_denom = pc.band("B1")+pc.band("B3")
ndwi_num = pc.band("B1")-pc.band("B3")
ndwi = np.zeros_like(pc.band("B3"))
valid_px = np.where(ndwi_denom != 0)
print(valid_px)
ndwi[valid_px] = (ndwi_num[valid_px]/ndwi_denom[valid_px])
pc.set_band(ndwi, "NDWI")
ndwi_enh = enhance.norm_to_uint(np.copy(ndwi), 256, np.uint8, norm=False)

""" Get a NDVI RGB """
asat = np.full_like(ndvi_enh, saturation)
asat = (asat*(ndvi_enh*ndvi_scale_stretch+ndvi_scale_offset)).astype(np.uint8)
aval = np.full_like(ndvi_enh, value, dtype=np.uint8)
aval = (aval*(ndvi_enh*ndvi_scale_stretch+ndvi_scale_offset)).astype(np.uint8)
ndvi_hue = np.round(ndvi_enh*(hsv_range[1]-hsv_range[0])+hsv_range[0]
                    ).astype(np.uint8)
ndvi_hsv = np.dstack([ndvi_hue, asat, aval])
ndvi_rgb = cv.cvtColor(ndvi_hsv, cv.COLOR_HSV2RGB)
#gt.quick_render(ndvi_rgb)

""" Get a color bar """
width = 64
cb_hue = np.stack( [np.arange(*hsv_range, dtype=np.uint8) for i in range(width)] )
cb_sat = np.full_like(cb_hue, asat[0,0], dtype=np.uint8)
cb_val = np.full_like(cb_hue, aval[0,0], dtype=np.uint8)
cb_rgb = cv.cvtColor(np.dstack([ cb_hue, cb_sat, cb_val ]), cv.COLOR_HSV2RGB)
gt.quick_render(cb_rgb)

""" Get a NDWI RGB """
'''
ndwi_rgb = cv.cvtColor(
        np.dstack([ enhance.norm_to_uint(X, 256, np.uint8, norm=False)
                           for X in [ ndwi_enh, aval, asat]]),
        cv.COLOR_HSV2RGB)
'''
#gt.quick_render(ndwi_rgb)
#ndwi_enh[np.where(ndwi_)]
#gt.quick_render(ndwi_enh)
"""
Get water masks with ndwi and cloud masks with the saunders & kriebel method.
"""
water_mask = ndwi_enh > 100
cloud_mask = sk_cloud_mask(pc.band("B1")/255, pc.band("B4")/255,
#                          t1_thresh=.15, t2_thresh=1.6)
                          t1_thresh=.48, t2_thresh=.9)
ndvi_rgb[np.where(cloud_mask)] = np.asarray([255, 255, 255])
ndvi_rgb[np.where(water_mask)] = np.asarray([0, 0, 0])

invalid_ndvi_mask = np.logical_or(water_mask, cloud_mask)
valid_ndvi_mask = np.logical_not(invalid_ndvi_mask)
valid_ndvi = enhance.linear_gamma_stretch(
        pc.band("NDVI"))[np.where(valid_ndvi_mask)]
print(f"Mean of NDVI: ",np.average(valid_ndvi))
print(f"Std dev of NDVI: ",np.std(valid_ndvi))
gt.quick_render(ndvi_rgb)
gp.generate_raw_image(
        ndvi_rgb, fig_dir.joinpath(Path("ndvi_masked_skparams.png")))
