from astropy.io import fits
import os
import numpy as np

path = os.getcwd() + "/single_psf/"
file_names = np.array(os.listdir(path))

data = []
meta = []
fails = []
errors = []


for i in range(len(file_names)):
    if ".npy" in file_names[i]:
        print(file_names[i])
        try:
#             with fits.open(path + file_names[i]) as f:
#                 data_array = f[0].data

                data_array = np.load(path + file_names[i])
                data.append(data_array)

                px, py, wl = file_names[i][:-11].split("_")
                meta_array = [float(px), float(py), float(wl)*1e-9]
                meta.append(meta_array)
        except BaseException as e:
            fails.append(i)
            errors.append(e)
        
        
print(fails)
print(set(errors))
np.save("data_arrays", data)
np.save("meta_arrays", meta)