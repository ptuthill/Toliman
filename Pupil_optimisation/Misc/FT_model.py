import numpy as np

def model_FT(mask, mask_size, chip_dim, wavels, foc_length, pix_size, test=False):
    """
    Inputs:
        mask: Phase mask complex array 
        mask_size: Size of the phase mask, ie the aperture diameter (m)
        chip dim: Units of num pixels
        wavels: Array of wavelengths (m)
        foc_lenght: Focal length of lens/distance to focal plane of telescope (m)
        pix_size: Detector pitch, size of pixel unit cell (m)
    Note: Assumes all wavelengths have equal intesity (add intesities later)
    """
        
    grid_size = mask.shape[1]           
    plate_scale = pix_size / foc_length    # Radians per pixel
    im_out = np.zeros((chip_dim,chip_dim))
    
    for wavel in wavels:
        spatial_freq = wavel/mask_size
        array_size = int(grid_size*spatial_freq/plate_scale)
        complex_array = np.array(np.zeros((array_size,array_size)),dtype=complex)
        complex_array[0:grid_size,0:grid_size] = mask
        im = np.fft.fftshift(np.abs(np.fft.fft2(complex_array))**2)

        for y in range(chip_dim):
            for x in range(chip_dim):
                # Split line below and multiply line below by some scale/ratio to normalise
                # Pass in all wavelengths across all bands together and sum then normalise to 1
                im_out[y][x] += im[int((array_size-chip_dim)/2) + y][int((array_size-chip_dim)/2) + x]
    if test:
        return im_out, complex_array
    return im_out
    
if __name__ == "__main__":

    # Example imlpementation of the function

    import matplotlib.pyplot as plt
    import numpy as np
    from math import sin, log10, cos, atan2, hypot
    import math


    # Kierans spiral code, with some minor modifications for this use
    def binarized_ringed_flipped(r, phi, phase, thresh=0., white=0, empty=0., r_max=300., r_min=50., r_split=246.):
        # Spiral parameters
        alpha1 = 20.186
        m1 = 5
        eta1 = -1.308
        m2 = -5
        alpha2 = 16.149
        eta2 = -0.733
        m3 = 10
        alpha3 = 4.0372
        eta3 = -0.575

        s = 0.15/300. # m/internal sampling dist
    #     Physical dimensions
        r_max = 300.
        r_max = 10.
        r_min = 50.
        r_min = 0.
        r_split = 6.9 # interface between main sprial and outer rim

        black = phase
        v = empty
        r = r/s

        if (r<=r_max and r>r_min):
            logr = log10(r)
            chi1 = alpha1*logr+m1*phi+eta1
            c1 = cos(chi1)
            chi2 = alpha2*logr+m2*phi+eta2
            c2 = cos(chi2)
            chi3 = alpha3*logr+m3*phi+eta3
            c3 = sin(chi3)
            if (r>r_split): # Outer rim
                if (c3<thresh):
                    if (sin(chi3/2.)<thresh):
                        v=black if (c1*c2*c3>thresh) else white
                    else:
                        v=black
                else:
                    v=black if (c1*c2*c3>thresh) else white
            else: # Main spiral
                v=black if (c1*c2*c3>thresh) else white
        return v


    # Bryns implimentation of the code with minor modifications for this use
    ngrid = 1024
    ratio = 30
    diam = 0.3
    sampling = diam/(ngrid*ratio)
    c = ngrid/2.
    spiral = np.zeros([ngrid, ngrid], dtype = np.complex)

    for i in range(ngrid):
        for j in range(ngrid):
            x = i - c
            y = j - c
            phi = math.atan2(y, x)
            r = sampling*math.hypot(x,y)
            spiral[i][j] = binarized_ringed_flipped(r, phi, complex(-1,0),white=complex(1,0))



    mask = spiral
    mask_size = 20.e-3                  # Size of mask, Units assumed to be m
    chip_dim = 128                       # Size of chip in pixels - real chip is 3280 x 2464
    foc_length=77.e-3                   # focal length of lens (m)
    pix_size = 1.12e-6  

    FT_red = model_FT(mask, mask_size, chip_dim, np.linspace(575e-9,650e-9,5), foc_length, pix_size)
    FT_green = model_FT(mask, mask_size, chip_dim, np.linspace(475e-9,575e-9,5), foc_length, pix_size)
    FT_blue = model_FT(mask, mask_size, chip_dim, np.linspace(400e-9,500e-9,5), foc_length, pix_size)


    # Now make up Bayer pattern to down-sample the image
    bayer_green = np.remainder((np.arange(chip_dim**2))+np.arange(chip_dim**2)/chip_dim,2)
    im_green = bayer_green.reshape(chip_dim,chip_dim)*FT_green

    bayer_blue = np.remainder((np.arange(chip_dim**2))+1+np.arange(chip_dim**2)/chip_dim,4)/3
    im_blue = bayer_blue.reshape(chip_dim,chip_dim)*FT_blue

    bayer_red = np.remainder((np.arange(chip_dim**2))-1+np.arange(chip_dim**2)/chip_dim,4)/3
    im_red = bayer_red.reshape(chip_dim,chip_dim)*FT_red

    im_out = im_red + im_blue + im_green
    
    print(type(im_out))

    plt.imshow(im_out**0.05)
    plt.colorbar()
    plt.title("Image final")
    plt.show()