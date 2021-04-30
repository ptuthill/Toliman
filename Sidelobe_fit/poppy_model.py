import poppy
import numpy as np
import matplotlib
import astropy.units as u
from poppy.poppy_core import OpticalElement, Wavefront, PlaneType, _PUPIL, _IMAGE, _RADIANStoARCSEC
from poppy import utils,AnalyticOpticalElement
import pysynphot as S
from scipy.special import jv
from astropy import constants

import time

poppy.conf.use_multiprocessing = True
poppy.conf.use_fftw = False

class TolimanAperture(poppy.AnalyticOpticalElement):
    """ Defines the Toliman pupil by interpolating the pupil file.
    Used in poppy to make Toliman image simulations including the sidelobes.
    Parameters
    ----------
    name : string
        Descriptive name
    central_wav : float
        Central wavelength in metres. OPD is calculated assuming perfect pi or -pi phase at this wavelength
    telescope_diam : float:
        Primary mirror diameter in metres. Pupil coordinates are scaled to match this, so that the furthest pixel
        with nonzero phase is assumed to be at radius telescope_diam/2.
    add_grating : boolean
        Switch to add a grating function on top of the normal diffractive pupil
    grating_amp : float
        Amplitude of grating phase function
    grating_fact : float
        Spacing factor for grating, for which the phase function is calculated as:
        grating_phase = grating_amp*np.sin(x_pupil/grating_fact+y_pupil/grating_fact)
        (neglecting the fancy sign flips that give a better first-order PSF)
    achromatic_pupil : boolean (default False)
        Switch to force the grating and diffractive pupil elements to be perfect phase steps rather than 
        OPD steps
    sidelobe_fn: (default np.cos)
        Function to use for the phase grating.
    grating_offset1 and grating_offset2: (default 0)
        Phase offset used inside sidelobe_fn for each diagonal grating. Was used for debugging, and to investigate
        resistance to zernike modes.
    pupil_file: (default ../fixed_pupil.npy)
        Location of the file containing the Toliman diffractive pupil element. Should be a complex array.
    """
    
    def __init__(self,name=None, planetype=PlaneType.unspecified,central_wav=595e-9,telescope_diam=0.1,
                 add_grating=False,grating_amp=1.0,grating_fact=5,achromatic_pupil=False,sidelobe_fn=np.cos,
                 grating_offset1=0.,grating_offset2=0.,pupil_file='../fixed_pupil.npy',**kwargs):
        if name is None:
            name = "Toliman"
        super(TolimanAperture, self).__init__(name=name, planetype=planetype, **kwargs)
            
        self.central_wav = central_wav
        self.add_grating = add_grating
        self.grating_amp = grating_amp
        self.grating_fact = grating_fact
        self.achromatic_pupil = achromatic_pupil
        self.sidelobe_fn = sidelobe_fn
        self.grating_offset1 = grating_offset1
        self.grating_offset2 = grating_offset2
        
        self.pupil_file = pupil_file
        
        self.circle_diam = telescope_diam # how big the inscribed circle is, i.e. telescope diameter
        self.pupil = np.load(pupil_file)
        self.pupil_phase = np.angle(self.pupil)
        self.npix = self.pupil.shape[0]

        # Centre definition:
        # centred between pixels for even sizes, centred on a pixel for odd sizes.
        x = np.linspace(0,self.npix,num=self.npix,endpoint=False)-self.npix/2.+0.5 
        X,Y = np.meshgrid(x,x)
        r = np.hypot(X,Y)
        
        max_r = np.max(r[self.pupil_phase > 0]) # this is the radius of the circle
        coords = x/max_r
        
        # Coords is now relative to the circle diam, so the pupil diam is:
        self.pupil_diam = (coords[-1]-coords[0])/2*self.circle_diam        
    
    def get_opd(self,wave):
        
        if self.achromatic_pupil:
            design_wavelength = wave.wavelength.to(u.m).value
        else:
            design_wavelength = self.central_wav
            
        y, x = self.get_coordinates(wave)
        # convert to the pupil array coords
        x_pupil = x / self.pupil_diam * self.npix + self.npix//2
        y_pupil = y / self.pupil_diam * self.npix + self.npix//2
        y_int = np.round(y_pupil).astype(int)
        x_int = np.round(x_pupil).astype(int)
        
        # Set the outliers to zero
        x_int[x_int<0] = 0
        x_int[x_int>(self.npix-1)] = 0
        y_int[y_int<0] = 0
        y_int[y_int>(self.npix-1)] = 0
        
        opd = self.pupil_phase[x_int,y_int]/(2*np.pi) * design_wavelength
        
        if self.add_grating:
            phase_sign = (-1)**(1*(self.pupil_phase[x_int,y_int] > 0)) # maps [0,pi] to [1,-1]
            
            # Make a diagonal grating in each direction
            # Perhaps we should use cos since it's symmetric
            # There's an offset here for debugging
            grating1 = phase_sign*self.grating_amp*self.sidelobe_fn((x+y)/self.grating_fact+self.grating_offset1)
            grating2 = phase_sign*self.grating_amp*self.sidelobe_fn((x-y)/self.grating_fact+self.grating_offset2)

            opd += (grating1+grating2)/(2*np.pi) * design_wavelength
        
        return opd
    
    def get_transmission(self,wave):
        y, x = self.get_coordinates(wave)
        r = np.hypot(x,y)
        transmission = np.ones(wave.shape)
        transmission[r > self.circle_diam/2] = 0
        return transmission

def diffraction_spot_offset(wavelength,aperture,plate_scale):
    """ Compute the radial distance (in pixels) to the diffraction spot at a given wavelength.
    wavelength in metres
    plate_scale in arsec/pix
    aperture is a TolimanAperture object
    """
    offset_pix = wavelength*180*3600/(np.pi**2*aperture.grating_fact*plate_scale) # using plate scale in arcsec
    return offset_pix

def make_image_corners_only(poppy_optical_system,npix,wavs,subimage_pix=256):
    """ Make an image of the sidelobes only. This is done by simulating 4 detector offsets.
    Unfortunately it didn't speed up the calculation significantly compared to doing the whole image.
    """

    # Get the TolimanAperture object from the optical system object
    for p in poppy_optical_system.planes:
        if isinstance(p,TolimanAperture):
            aperture = p
        elif isinstance(p,poppy.Detector):
            detector = p
            
    pixelscale = detector.pixelscale.value
    
    image = np.zeros((npix,npix))
    
    # Where are the sidelobes?
    wmin = np.min(wavs)
    wmax = np.max(wavs)
    ideal_offset_pix = diffraction_spot_offset((wmax+wmin)/2,aperture,pixelscale)
    
    poppy_optical_system = poppy.OpticalSystem()
    poppy_optical_system.add_pupil(aperture)
    poppy_optical_system.add_detector(pixelscale=pixelscale, fov_arcsec=subimage_pix*pixelscale/2*u.arcsec)
    
    # Loop through the 4 quadrants and calculate them, then add it to the full image
    for offset_angle in [45,135,225,315]:
        for j, wavelength in enumerate(wavs):
            
            ideal_offset_pix = diffraction_spot_offset(wavelength,aperture,pixelscale)
        
            # Snap it to a grid
            ideal_offset_x_pix = ideal_offset_pix*np.sign(np.sin(offset_angle*np.pi/180.))
            ideal_offset_y_pix = ideal_offset_pix*np.sign(np.cos(offset_angle*np.pi/180.))

            offset_x_pix = np.int(np.round(ideal_offset_x_pix))
            offset_y_pix = np.int(np.round(ideal_offset_y_pix))
            actual_offset_angle = np.arctan2(offset_y_pix,offset_x_pix)*180./np.pi # should be == offset_angle...
            actual_offset_r_pix = np.sqrt(offset_x_pix**2+offset_y_pix**2)

            poppy_optical_system.source_offset_theta = -offset_angle
            poppy_optical_system.source_offset_r =  0.5*actual_offset_r_pix*pixelscale#*u.arcsec

            corner_im = poppy_optical_system.calc_psf(wavelength)[0].data
            
            # Add it to the full image
            x1 = npix//2-offset_x_pix-subimage_pix//2
            x2 = npix//2-offset_x_pix+subimage_pix//2
            y1 = npix//2-offset_y_pix-subimage_pix//2
            y2 = npix//2-offset_y_pix+subimage_pix//2
            image[y1:y2,x1:x2] += corner_im
            
    return image

def supergaussian(wavs,centre,sigma=100e-9,n=8):
    """ Supergaussian function used for the filter bandwidth
    n = power inside the exponential
    sigma = width parameter
    """
    return np.exp(-((wavs-centre)/(sigma/2.))**n)


def airy_pattern(x,y,x0,y0,amplitude,radius):
    """ Returns a 2D array with an airy pattern
    on a grid (x,y), with centre (x0,y0) and a given
    amplitude and width parameter
    """
    # the magic number is the zero of the bessel function, so when dist = radius, airy=0
    r = np.pi*np.sqrt((x-x0)**2 +(y-y0)**2) / (radius/1.2196698912665045)
    
    # Handle r=0 to avoid divide by zero errors
    airy = np.ones(r.shape)
    r_gt_0 = (r>0)
    airy[r_gt_0] = amplitude*(2*jv(1,r[r_gt_0])/(r[r_gt_0]))**2
    
    return airy
    
def airy_model_image(npix,wavs,spectrum,psf_radius,aperture,plate_scale,offset=[0,0],cutout_sz=50):
    """
    Makes a model of the sidelobes using Airy patterns located in the 4 detector corners
    at the correct offset for each wavelength.
    
    npix : int
        Number of pixels in final image
    wavs : numpy array
        Array of wavelengths (in metres)
    spectrum : numpy array
        Weights for the images at each wavelength. Array of the same size as wavs
    psf_radius : float
        The distance to the first null of the Airy disk at the central wavelength
    aperture : TolimanAperture
        A TolimanAperture object representing the best guess of the optical setup
    plate_scale : float
        Size of one image pixel in arcsec/pix
    cutout_sz : int (default = 50)
        Each Airy disk is calculated in a box of length cutout_sz before being added to the final image.
    offset : [float,float]
        [y,x] position to centre the sidelobes around (in pixels).
    """
    
    im = np.zeros((npix,npix))
    x,y = np.indices(im.shape)
    central_wav = wavs.mean()
    
    for wav_ix,wavelength in enumerate(wavs):
        
        # Where will the sidelobe PSF be at this wavelength?
        offset_pix = diffraction_spot_offset(wavelength,aperture,plate_scale)
        
        # Scale the Airy pattern width and amplitude by the wavelength
        effective_radius = psf_radius*wavelength/central_wav # it gets bigger as lambda
        amplitude = spectrum[wav_ix]*(central_wav/wavelength)**2 # energy spreads out as 1/lambda**2
        
        # Loop over the sidelobes
        for xsign,ysign in [[-1,-1],[-1,1],[1,-1],[1,1]]:
            
            # Work out the indices for the 4 corners of the box we'll calculate over
            xstart = int(npix//2+xsign*offset_pix-cutout_sz//2+offset[0])
            xend = int(npix//2+xsign*offset_pix+cutout_sz//2+offset[0])
            ystart = int(npix//2+ysign*offset_pix-cutout_sz//2+offset[1])
            yend = int(npix//2+ysign*offset_pix+cutout_sz//2+offset[1])

            # You'll get an error here if the sidelobes are too close to the edges
            x_cut = x[xstart:xend,ystart:yend]
            y_cut = y[xstart:xend,ystart:yend]
            
            # Centre of image, centre definition, sidelobe offset, jitter offset
            xcen = npix/2 - 0.5 + xsign*offset_pix + offset[0]
            ycen = npix/2 - 0.5 + ysign*offset_pix + offset[1]
            
            model = airy_pattern(x_cut,y_cut,xcen,ycen,amplitude,effective_radius)
            
            im[xstart:xend,ystart:yend] += model

    return im

def airy_model_residuals(params,*args,**kwargs):
    """ Residual function for an L-M fit (i.e. scipy.optimize.least_squares).
    Params:
        [plate_scale, psf_radius, stellar_teff,flux,ycentre,xcentre]
    If extra parameters are given to this function, more things will be fit:
        [plate_scale, psf_radius, stellar_teff,flux,ycentre,xcentre,
         filter_centre,RV]
    Units:
        [arcsec/pix, pixels, K, ~max counts in image, pixels, pixels,
        metres, km/s]
    
    if kwargs['filter_centre'] or kwargs['RV'] are set, they will _not_ be fit,
    and the value in kwargs will be used instead.
    
    Setting kwargs['mask'] will return residuals[mask]
    
    """

    # Unpack everything
    image = kwargs['image']
    aperture = kwargs['aperture']
    wavs = kwargs['wavs']
    
    if 'cutout_sz' in kwargs.keys():
        cutout_sz = kwargs['cutout_sz']
    else:
        cutout_sz = 50
        
    if 'debug' in kwargs.keys():
        print(params)
    
    npix = image.shape[0] # assume square
    nwavs = wavs.size
    
    plate_scale = params[0]
    psf_radius = params[1]
    
    # Use a Phoenix spectrum and the filter
    stellar_teff = params[2]
    flux = params[3]
    
    if 'filter_centre' in kwargs.keys():
        filter_centre = kwargs['filter_centre']
    elif len(params) > 6:
        filter_centre = params[6]
    else:
        filter_centre = 550e-9 # m
    
    if 'RV' in kwargs.keys():
        rv = kwargs['RV']
    elif len(params) > 7:
        rv = params[7]
    else:
        rv = 0. # km/s
    source_rest_wavs = wavs/(1+rv*u.km/u.s/constants.c)
    acenA = S.Icat('phoenix',stellar_teff,0.2,4.3)
    specA = acenA.sample(source_rest_wavs*1e10) # This needs angstroms as input
    
    if 'filter_sigma' in kwargs.keys():
        filter_sigma = kwargs['filter_sigma']
    else:
        filter_sigma = 105e-9
        
    if 'filter_n' in kwargs.keys():
        filter_n = kwargs['filter_n']
    else:
        filter_n = 8
    
    filter_spec = supergaussian(wavs,filter_centre,sigma=filter_sigma,n=filter_n)
    spectrum = filter_spec*specA
    spectrum *= flux/np.max(spectrum)
    
    # Calculate model image
    model = airy_model_image(npix,wavs,spectrum,psf_radius,aperture,plate_scale,offset=params[4:6],cutout_sz=cutout_sz)
    
    resids = image-model
    
    if 'mask' in kwargs.keys():
        resids = resids[kwargs['mask']]

    return resids.ravel()