import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

def display_split(image, npixels=1024, radius=13, size=200, inner_size=50, sizes=(12,6)):
    central, outer = split_central(image, npixels, radius)
    fig, axs = plt.subplots(1, 2, figsize=sizes)
    c = image.shape[0]/2

    im_out = central[int(c-inner_size//2):int(c+inner_size//2),int(c-inner_size//2):int(c+inner_size//2)]
    im = axs[0].imshow(im_out)
    plt.colorbar(im,ax=axs[0],fraction=0.046, pad=0.04)

    ctr_out = outer[int(c-size//2):int(c+size//2),int(c-size//2):int(c+size//2)]
    im = axs[1].imshow(ctr_out)
    plt.colorbar(im,ax=axs[1],fraction=0.046, pad=0.04)

    plt.show()

def display(wf, image, size=200, sizes=(12,6)):
    fig, axs = plt.subplots(1, 2, figsize=sizes)
    
    im = axs[0].imshow(np.abs(np.angle(wf)))
    plt.colorbar(im,ax=axs[0],fraction=0.046, pad=0.04)
    
    c = image.shape[0]/2
    im_out = image[int(c-size//2):int(c+size//2),int(c-size//2):int(c+size//2)]
    im = axs[1].imshow(im_out)
    plt.colorbar(im,ax=axs[1],fraction=0.046, pad=0.04)
    
    plt.show()
    
def display_spiral(wf,sizes=(12,12)):
    plt.figure(figsize=sizes)
    plt.imshow(np.abs(np.angle(wf)))
    plt.colorbar()
    plt.show()
    
def split_central(image, npixels, radius):
    central = np.zeros([npixels, npixels], dtype = np.float)
    outer = np.zeros([npixels, npixels], dtype = np.float)
    c = npixels//2
    for i in range(npixels):
        for j in range(npixels):
            x = i - c
            y = j - c
            r = math.hypot(x,y)
            if r <= radius:
                central[i][j] = image[i][j]
            else:
                outer[i][j] = image[i][j]
    return central, outer 