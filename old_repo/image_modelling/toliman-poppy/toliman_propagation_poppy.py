from __future__ import (absolute_import, division, print_function, unicode_literals)
import poppy
# import numpy as np
# from astropy import *
# from astropy.table import Table
#
# from IPython.core.pylabtools import figsize, getfigs
#
# from pylab import *
# from numpy import *
#
# import scipy.special
# import scipy.ndimage.interpolation
# import matplotlib
# import astropy.io.fits as fits
import astropy.units as u
# import warnings
# from poppy.poppy_core import OpticalElement, Wavefront, PlaneType, _PUPIL, _IMAGE, _RADIANStoARCSEC
# from poppy import utils
#
# import pysynphot as S
import logging

# Positions for un-folded telescope, relative to input pupil
pupil_m2_dist = 0.5 * u.m
m1_m2_dist = 549.337630333726 * u.mm
m2_focus_dist = 589.999999989853 * u.mm

# Dimensions
m1_rad = 0.3 * u.m
m2_rad = 0.059 * u.m
m2_strut_width = 0.01 * u.m
m2_supports = 5
m1_fl = 1.143451 * u.m
m2_fl = 0.0467579189727913 * u.m
m1_conic = -1.0001147
m2_conic = -1.16799179177759
m1_aperture_rad = 7.74E-05

# from matplotlib import pylab, mlab
# import matplotlib.pyplot as plt
# plt = pyplot
# %pylab inline --no-import-all
# matplotlib.rcParams['image.origin'] = 'lower'
# matplotlib.rcParams['figure.figsize']=(10.0,10.0)    #(6.0,4.0)
# matplotlib.rcParams['font.size']=16              #10
# matplotlib.rcParams['savefig.dpi']= 200             #72
# matplotlib.rcParams['savefig.format']='png'
# poppy.__version__

logging.getLogger('poppy').setLevel(logging.WARN)
# Can be logging.CRITICAL, logging.WARN, logging.INFO, logging.DEBUG for increasingly verbose output

import toliman_optics

m1 = toliman_optics.phase_rosette(
    m1_rad,  # edge radius
    5,  # 5-fold rotational symmetry
    0.25,  # rosette radius
    0.8,
    0.065,  # 7.5 cm semimajor
    0.5  # phase
)
m1.name = 'TOLIMAN rosette primary mirror'

toliman = poppy.FresnelOpticalSystem(name='TOLIMAN telescope', pupil_diameter=m1_rad, npix=1024, beam_ratio=0.5)

# Secondary mirror obscuration & spider
toliman.add_optic(poppy.SecondaryObscuration(
    name='Secondary mirror support',
    secondary_radius=m2_rad,
    n_supports=m2_supports,
    support_width=m2_strut_width,
    support_angle_offset=0 * u.deg),
    pupil_m2_dist)

# Primary mirror
# Rosette components
toliman.add_optic(m1, m1_m2_dist)
# Central aperture - treat as a secondary obscuration with no spider
toliman.add_optic(poppy.SecondaryObscuration(
    name='Secondary mirror support',
    secondary_radius=m1_aperture_rad,
    n_supports=0),
    0 * u.m)  # actually should be a bit further along
# Mirror component
toliman.add_optic(poppy.fresnel.ConicLens(f_lens=m1_fl, K=m1_conic, radius=m1_rad, name="Primary mirror"),
                  0 * u.m)

# Secondary mirror
toliman.add_optic(poppy.fresnel.ConicLens(f_lens=m2_fl, K=m2_conic, radius=m2_rad, name="Secondary mirror"),
                  m1_m2_dist)

# Aperture in primary
toliman.add_optic(poppy.CircularAperture(radius=m1_aperture_rad), m1_m2_dist)

toliman.add_optic(poppy.ScalarTransmission(planetype=poppy.fresnel.PlaneType.image, name='focus'), distance=m2_focus_dist)

poppy.describe()
# values = rosette.sample(npix=2048)    # evaluate on 512 x 512 grid

# fig=plt.figure(figsize=(10,5))
# rosette.display(what='both')         # display phase and amplitude transmission;
# pylab.savefig('test.png')
# plt.close(fig)
# plt.show(block=False)

# npix = 1024 # ~512 is minimum to accurately recover the central diffraction spike
# wf = poppy.FresnelWavefront(0.5*u.m,wavelength=2200e-9,npix=npix,oversample=4)
# wf *= poppy.CircularAperture(radius=0.5)
