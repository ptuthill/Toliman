import sys
import os
from shutil import rmtree
from astropy.io import fits
from functions import *
import multiprocessing as mp

# Set simulation name
sim_name = "test_test_test"

# Simulation Paramters
N = 10 # Number of images

# Science signal paramters
signal_size = 1e-6 # Pixels
signal_cycles = 3

# Random number seed
seed = 0

# Jitter paramters
stdev = 0.5 # Pixels

# Rotation Parameters
n_rots = 0.3 # Number of rotations over total data set

# Multiprocessing
n_cores = 4

# Set flux ratio (total flux acenA / total flux acenB)
diverse_spec = True # If True the two stars spectrums will not be identical
flux_ratio = 'natural_ratio' # Can be set to float or 'natural_ratio'
# flux_rato = 3
    
# Telescope parameters
aperture = 0.1 # m
m2_obsc = 0 # m
focal_length = 3.65 # 15cm camera lens
npix = 256 # Only simulate central region to decrease computational complexity
pix_size = 6.5e-6 # 1.12 microns
transform_size = 2**10 # Size of the transformed array

# Astrophysical paramters
separation = 16.5 # Arceseconds
pix_sep = arcsec_to_pixel(separation, pix_size, focal_length)
pos_A = [0, 0] # Alpha cen A starting positon
pos_B = [pix_sep, 0] # Start with stars aligned along the x axis

# Camera parameters
dark_current = 80
read_noise = 1.3
QE = 0.95
full_well = 45e3
integration_time = 24 # Hours 
frame_rate = 10 # Hz
spd = 86400 # Seconds per day
fpd = spd*frame_rate # Frames per day

#######################################################

files = os.listdir()
if sim_name in files:
    raise NameError("Simulation already exists!") 
else:
    os.makedirs("{}/PSFs".format(sim_name))
    os.makedirs("{}/pupils".format(sim_name))
    os.makedirs("{}/images".format(sim_name))
    
try:

    np.random.seed(seed)
    
    # Detector noise
    dark_noise = read_noise + (dark_current/frame_rate)

    specA = np.load("AcenA_cps_1.npy")
    specB = np.load("AcenB_cps_1.npy")
    wavels = np.load("Wavelengths.npy")
    wavels = np.array(wavels, dtype=np.int)
    natural_ratio = np.sum(specA)/np.sum(specB)

    if flux_ratio == "natural_ratio":
        flux_ratio = natural_ratio

    if diverse_spec:
        specB = specB*natural_ratio/flux_ratio
    else:
        specB = specA/flux_ratio

    # Science signal
    theta = np.linspace(0, signal_cycles*2*np.pi, num=N, endpoint=False)
    drs = signal_size*np.sin(theta)
    
    # Write signal file
    with open("{}/signal.csv".format(sim_name), 'w') as f:
        f.write("t,x\n")
        for i in range(N):
            f.write("{},{}\n".format(i, drs[i]))

    # Generate rotations
    phis = np.linspace(0, n_rots*2*np.pi, num=N, endpoint=False)
    Xs = np.vstack((pix_sep+drs) * np.cos(phis))
    Ys = np.vstack((pix_sep+drs) * np.sin(phis))
    coords = np.hstack((Xs, Ys))

    # 2D Gaussian with a standard deviation of 0.5 pixels
    jitters = np.random.normal(scale=stdev, size=2*N).reshape(N, 2)
    A_coords_xy = jitters
    B_coords_xy = coords + jitters

    A_rs = (A_coords_xy[:,0]**2 + A_coords_xy[:,1]**2)**0.5
    B_rs = (B_coords_xy[:,0]**2 + B_coords_xy[:,1]**2)**0.5

    A_phis = np.arctan2(A_coords_xy[:,1], A_coords_xy[:,0])
    B_phis = np.arctan2(B_coords_xy[:,1], B_coords_xy[:,0])

    pupil_in = pupil_from_fits("pupil.fits")
    sizes = []

    for wl in wavels:
        # Caclaute how big the pupil array needs to be to get our desired Fourier Transformed array size
        plate_scale = pix_size/focal_length    # Radians per pixel
        spatial_freq = (wl*1e-9)/aperture
        scale_factor = spatial_freq/plate_scale
        size_out = int(transform_size/scale_factor)
        sizes.append([size_out, wl, pupil_in, sim_name])


    pool = mp.Pool(processes=n_cores)
    null = pool.map(make_pupil, sizes)

    static = [aperture, m2_obsc, npix, focal_length, pix_size, transform_size, sim_name]
    inputs = []

    for n in range(N):
        for wl in wavels:
            for i in range(2):
                if i %2 == 0:
                    r = A_rs[n]
                    phi = A_phis[n]
                    star = "A"
                else:
                    r = B_rs[n]
                    phi = B_phis[n]
                    star = "B"

                dynamic = [n, wl, r, phi, star]
                inputs.append(static + dynamic)


    pool = mp.Pool(processes=n_cores)
    null = pool.map(model_image, inputs)

    # Combine images
    for n in range(N):
        # Add detector noise
        im_out =  np.abs(np.round(np.random.normal(scale=dark_noise, size=[npix, npix])))

        for i in range(len(wavels)):
            wl = wavels[i]
            countsA = specA[i]
            countsB = specB[i]

            PSF_A = np.load("{}/PSFs/{}_{}_A.npy".format(sim_name, n, wl))
            PSF_B = np.load("{}/PSFs/{}_{}_B.npy".format(sim_name, n, wl))

            im_out += countsA*PSF_A + countsB*PSF_B

        fits.writeto("{}/images/{}.fits".format(sim_name, n), im_out)
    #     np.save("images/{}.npy".format(n), im_out)


    # Delete temporary directories
    rmtree("{}/PSFs".format(sim_name))
    rmtree("{}/pupils".format(sim_name))
    
except BaseException as e:
    print("Sim failed, all temporary file and folders deleted")
    print("Error: {}".format(e))
    
    rmtree("{}".format(sim_name))