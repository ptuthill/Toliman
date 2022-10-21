

import numpy as np

# def mag2jy(wavelength, magnitude):
#     exponent = -0.4*magnitude
#     fv = 3631*(10**exponent) 
#     f = (wavelength)*fv
#     return f*4 # factor of 4 is a fudge factor

def spline(X, Y, T, sigma = 1.0):
    '''
    Python version of IDL's spline function source code
    Code comes from:
    https://stackoverflow.com/questions/63285118/how-to-translate-idls-spline-function-to-python-particularly-for-the-case-we-h
    There are some subtle differences between how normal implementations of spline functions act in python vs IDL
    '''
    n = min(len(X), len(Y))
    if n <= 2:
        print('X and Y must be arrays of 3 or more elements.')
    if sigma != 1.0:
        sigma = min(sigma, 0.001)
    yp = np.zeros(2*n)
    delx1 = X[1]-X[0]
    dx1 = (Y[1]-Y[0])/delx1
    nm1 = n-1
    nmp = n+1
    delx2 = X[2]-X[1]
    delx12 = X[2]-X[0]
    c1 = -(delx12+delx1)/(delx12*delx1)
    c2 = delx12/(delx1*delx2)
    c3 = -delx1/(delx12*delx2)
    slpp1 = c1*Y[0]+c2*Y[1]+c3*Y[2]
    deln = X[nm1]-X[nm1-1]
    delnm1 = X[nm1-1]-X[nm1-2]
    delnn = X[nm1]-X[nm1-2]
    c1 = (delnn+deln)/(delnn*deln)
    c2 = -delnn/(deln*delnm1)
    c3 = deln/(delnn*delnm1)
    slppn = c3*Y[nm1-2]+c2*Y[nm1-1]+c1*Y[nm1]
    sigmap = sigma*nm1/(X[nm1]-X[0])
    dels = sigmap*delx1
    exps = np.exp(dels)
    sinhs = 0.5*(exps-1/exps)
    sinhin = 1/(delx1*sinhs)
    diag1 = sinhin*(dels*0.5*(exps+1/exps)-sinhs)
    diagin = 1/diag1
    yp[0] = diagin*(dx1-slpp1)
    spdiag = sinhin*(sinhs-dels)
    yp[n] = diagin*spdiag
    delx2 = X[1:]-X[:-1]
    dx2 = (Y[1:]-Y[:-1])/delx2
    dels = sigmap*delx2
    exps = np.exp(dels)
    sinhs = 0.5*(exps-1/exps)
    sinhin = 1/(delx2*sinhs)
    diag2 = sinhin*(dels*(0.5*(exps+1/exps))-sinhs)
    diag2 = np.concatenate([np.array([0]), diag2[:-1]+diag2[1:]])
    dx2nm1 = dx2[nm1-1]
    dx2 = np.concatenate([np.array([0]), dx2[1:]-dx2[:-1]])
    spdiag = sinhin*(sinhs-dels)
    for i in range(1, nm1):
        diagin = 1/(diag2[i]-spdiag[i-1]*yp[i+n-1])
        yp[i] = diagin*(dx2[i]-spdiag[i-1]*yp[i-1])
        yp[i+n] = diagin*spdiag[i]
    diagin = 1/(diag1-spdiag[nm1-1]*yp[n+nm1-1])
    yp[nm1] = diagin*(slppn-dx2nm1-spdiag[nm1-1]*yp[nm1-1])
    for i in range(n-2, -1, -1):
        yp[i] = yp[i]-yp[i+n]*yp[i+1]
    m = len(T)
    subs = np.repeat(nm1, m)
    s = X[nm1]-X[0]
    sigmap = sigma*nm1/s
    j = 0
    for i in range(1, nm1+1):
        while T[j] < X[i]:
            subs[j] = i
            j += 1
            if j == m:
                break
        if j == m:
            break
    subs1 = subs-1
    del1 = T-X[subs1]
    del2 = X[subs]-T
    dels = X[subs]-X[subs1]
    exps1 = np.exp(sigmap*del1)
    sinhd1 = 0.5*(exps1-1/exps1)
    exps = np.exp(sigmap*del2)
    sinhd2 = 0.5*(exps-1/exps)
    exps = exps1*exps
    sinhs = 0.5*(exps-1/exps)
    spl = (yp[subs]*sinhd1+yp[subs1]*sinhd2)/sinhs+((Y[subs]-yp[subs])*del1+(Y[subs1]-yp[subs1])*del2)/dels
    if m == 1:
        return spl[0]
    else:
        return spl
    
def mag2jy(wavelengths, magnitude):
    #NEW set of standards from VEGA (Allen, astrophysical quantities)
    # _EXCEPT_ ones indicated in brackets taken from above
    # bands= (U),(B),V,(R),(I),J,H,Ks,K,L,L',M,8.7,N,11.7,Q

    mylambda =  np.array([0.36,0.44,0.5556,0.70,0.90,1.215,1.654,2.157,2.179,3.547,3.761,4.769,8.756,10.472,11.653,20.130])
    jansky0 = np.array([1880,4440,3540,  2880,2240,1630, 1050, 667,  655,  276, 248,  160,  50.0, 35.2,  28.6,  9.70  ])

    if np.max(wavelengths) >= np.max(mylambda) or np.min(wavelengths) <= np.min(mylambda):
        print('### ERROR - Lambda out of interpolation range ###')
        
    jansky = spline(mylambda, jansky0, wavelengths)

    # convert magnitudes
    star_flx = 10**(magnitude/(-2.5)) * jansky
    return star_flx
    
    
def flux(w_min, w_max, t_area, instr_eff, det_eff, polz_eff, t_integ, atm_loss, s_mag):
    # fundamental constants etc
    jy = 1.0e-26   # Wattz/(Hz M^2)
    c  = 2.99792e8 # m/s
    h  = 6.625e-34
    
    w_mean = (w_max+w_min)/2
    w_bw   = (w_max - w_min)
    
    # Work out how much flux we have from the star. 
    # Per model time bin, waveband bin
    s_lum = mag2jy([w_mean*1e6], s_mag)  # this is now in jansky (not scaled) at the top of the atmosphere?
    
    f_bin = c*( 1/(w_mean-w_bw/2.0) - 1/(w_mean+w_bw/2.0))
    
    s_lum = s_lum * jy * f_bin * t_area # this should be in joules/sec
    
    photon_energy = h*c/w_mean
    
    s_lum = s_lum / photon_energy # should now be in photons/sec
    s_lum = s_lum * t_integ # photons / integration
    
    print('Luminosity before losses: {:.2e} photons/integration'.format(s_lum))
    
    s_lum = s_lum * atm_loss * instr_eff * polz_eff
    
    print('Luminosity at detector:   {:.2e} photons/integration'.format(s_lum))
    
    s_lum = s_lum * det_eff
    
    print('Final detected flux:      {:.2e} photons/integration'.format(s_lum))
    print('-------------------')
    return s_lum

def rad2mas(x):
    ''' Convenient little function to convert radians to milliarcseconds '''
    return x/np.pi*(180*3600*1000)
