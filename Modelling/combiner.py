import os
import numpy as np
from conversions import arcsec_to_pixel

path = os.getcwd() + "Toliboy/full_cycle/npy/"
save_path = os.getcwd() + "Toliboy/full_cycle/"
file_names = np.array(os.listdir(path))

images = []
Xs = []
for i in range(len(file_names)):
    im = np.load(path+file_names[i])
    images.append(im)
    i, x = file_names[i][-4].split("_")
    Xs.append(float(x))

on_axis_full = images[0]
c = on_axis_full.shape[0]//2

# pixel_shift = arcsec_to_pixel(6, 6.5e-6, 3.635)
# Full psf has about a 30 pixel radus
# 5 arcsecond seperation(real) is about 16 pix
# Set sepearation to be 35 pixels

pixel_shift = 35
s = 256//2

psfs = []
for i in range(len(images)):
    psf_out = np.zeros([2*s,2*s])
    psf_out += on_axis_full[c-s:c+s, c-s:c+s]
    psf_out += images[i][c-s:c+s, c-s-pixel_shift:c+s-pixel_shift]
    psfs.append(psf_out)

final_psfs = []
for i in range(3):
    for j in range(len(psfs)):
        final_psfs.append(psfs[j])

photon_counts = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
photon_counts_str = ["1e4", "1e5", "1e6", "1e7", "1e8", "1e9"]
for i in range(len(photon_counts)):
    for j in range(len(final_psfs)):

        photons = np.random.poisson(final_psfs[j])
        noise = np.round(np.random.normal(scale=20, size=photons.shape))
        image = photons+noise

        file_name = "{}/{}_{}".format(photon_counts_str[i], j, Xs[j%len(Xs)])
        fits.writeto(save_path + file_name, image)


