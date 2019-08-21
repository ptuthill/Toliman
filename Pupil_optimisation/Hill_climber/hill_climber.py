import numpy as np
from math import sin, log10, cos, atan2, hypot
from copy import deepcopy

def model_FT(mask, mask_size, chip_dim, wavels, foc_length, pix_size):
	"""
	mask: Complex array with phase mask
	mask_size: Size of the phase mask, ie the aperture diameter (m)
	chip dim: Units of num pixels
	wavels: Array of wavelengths (m)
	foc_lenght: Focal length of lens/distance to focal plane of telescope (m)
	pix_size: Detector pitch, size of pixel unit cell (m)
	Note: Assumes all wavelengths have equal intesity (add intesities later)
	"""

	grid_size = mask.shape[1]		   
	plate_scale = pix_size / foc_length	# Radians per pixel
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
	
	return im_out

def generate_spiral(gridsize, aperture, r_max, r_min, splits, settings):
	split = splits
	first = settings[0]
	second = settings[1]
	third = settings[2]
	fourth = settings[3]
	
	sampling = aperture/(gridsize//2)
	wfarr = np.zeros([gridsize, gridsize], dtype = np.complex128)
	c = gridsize//2
	for i in range(gridsize):
		for j in range(gridsize):
			x = i - c
			y = j - c
			phi = atan2(y, x)
			r = sampling*hypot(x,y)
			wfarr[i][j] = spiral(r, phi, aperture, r_max, r_min, split, first, second, third, fourth)
	return wfarr

def spiral(r, phi, aperture, r_max, r_min, split, first, second, third, fourth):
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
	
	white = np.complex(1,0)
	black = -np.complex(1,0)
	v = np.complex(0,0)
	offset = np.complex(-(3**0.5)/2,1/2)

	if (r<=r_max and r>r_min):
		logr = log10(r)
		chi1 = alpha1*logr+m1*phi+eta1
		c1 = cos(chi1)
		chi2 = alpha2*logr+m2*phi+eta2
		c2 = cos(chi2)
		chi3 = alpha3*logr+m3*phi+eta3
		c3 = sin(chi3)
		
		z = 0 if (c1*c2*c3>0) else 1 
		for i in range(len(split)):
			if (r <= split[i] and r > split[i+1]): # Finds which region we are in
				
				if i%2 != 0:
					z = np.abs(z-1)
				
				# First quadrant
				if c3 < 0 and sin(chi3/2.) <= 0:
					return black if first[i][z] else white

				# Second qudrant
				elif c3 >= 0 and sin(chi3/2.) <= 0:
					return black if second[i][np.abs(z-1)] else white
				
				# Third quadrant
				elif c3 < 0 and sin(chi3/2.) > 0:
					return black if third[i][z] else white
		
				# Fourth qudrant
				else: 
					return black if fourth[i][np.abs(z-1)] else white
					
	elif r < r_min:
#		 v = black
		v = np.complex(0,0)
	return v

def generate_new(N=40, p=0.5):
	return np.reshape(np.random.choice(a=[0, 1], size=(1, N), p=[p, 1-p])[0], (4, 5, 2)).tolist()

def flip1(settings):
	new_settings = deepcopy(settings)
	r1 = np.random.randint(0, high=4)
	r2 = np.random.randint(0, high=5)
	r3 = np.random.randint(0, high=2)
	new_settings[r1][r2][r3] = np.abs(settings[r1][r2][r3] - 1)
	return new_settings
	
def flip2(settings):
	new_settings = deepcopy(settings)
	r1 = np.random.randint(0, high=4)
	r2 = np.random.randint(0, high=5)
	new_settings[r1][r2][0] = np.abs(settings[r1][r2][0] - 1)
	new_settings[r1][r2][1] = np.abs(settings[r1][r2][1] - 1)
	return new_settings

def get_value(settings):
	aperture = 0.015				 # Aperture (m)
	gridsize = 1024				  # Gridsize of the wavefront array
	npixels = 1024				   # Size of detector, in pixels
	wl = 0.525e-9					# Wavelength values (micrometers)
	fl = 0.15						# Focal length (m)
	detector_pitch = 1.12e-6		 # m/pixel on detector (pixel spacing)

	split_values = [15.15, 12.4, 10.17, 8.33, 6.83, 5.6]
	r_max = split_values[0]
	r_min = split_values[-1]
	splits = split_values
	
	wf = generate_spiral(gridsize, aperture*1e3, r_max, r_min, splits, settings)
	FT = model_FT(wf, aperture*1e-3, npixels, [wl], fl, detector_pitch)
	
	return FT.max()*1e-8

def optimise(seed, num_fails, i):
	# Generate initial spiral
	if seed is None:
		curr_settings = generate_new()
	else:
		curr_settings = seed
		
	# Create the wf and FT and evaluate its performance
	curr_val = get_value(curr_settings)
	count = 0		# Number of sucessive unsuccessful flips
	total_count = 0	# Total Number of unsuccessful flips
	flips = 0		# Number of successful flips

	# Array of the two flipping functions
	flippers = [flip1, flip2]
	
	# Print the initial condition of the optimisation proccess
	print("Spiral number: {}".format(i+1))
	print("Initial value: {:.4f}".format(curr_val))
	print("Initial settings: {}\n----------------".format(curr_settings))

	# Repeat untill we have unsuccessfully flipped sections num_fails times
	while count < num_fails:
		# Randomly choose to flip a pair of trianges or a single one
		flipper = flippers[np.random.randint(0,2)]
		# Generate a successor by flipping some section of the current spiral
		next_settings = flipper(curr_settings)
		# Create the wf, FT and evaluate the performance of the successive pupil
		next_val = get_value(next_settings)

		# print("Current value: {}".format(curr_val))
		# print("Next value: {}".format(next_val))

		# Sucessful change to spiral, accept change
		if next_val < curr_val:
			curr_settings = next_settings
			curr_val = next_val
			count = 0			# Reset count of successive unsuccessful flips count
			flips += 1			# Increment successfull flips counts

		# Unsucessful change to spiral, do not accept change
		else:
			count += 1			# Increment successive unsuccessful flips count
			total_count += 1	# Increment total unsuccessful flips count
	
	print("Final value: {:.4f}".format(curr_val))
	print("Final settings: {}\n----------------".format(curr_settings))

	# print("Next value: {:.4f}".format(next_val))
	# print("Next settings: {}\n----------------".format(next_settings))

	print("Number of successful flips: {}".format(flips))
	print("Number of unsuccessful flips: {}\n__________________\n".format(total_count))
	return curr_settings, curr_val

if __name__ == "__main__":

	# Full spiral
	seed0 = [[1,1],[1,1],[1,1],[1,1],[1,1]],[[1,1],[1,1],[1,1],[1,1],[1,1]],[[1,1],[1,1],[1,1],[1,1],[1,1]],[[1,1],[1,1],[1,1],[1,1],[1,1]]
	# Completely balance standard pupil
	seed1 = [[1,0],[0,1],[1,0],[0,1],[1,0]],[[1,1],[1,0],[0,1],[1,0],[0,1]],[[1,0],[0,1],[1,0],[0,1],[1,0]],[[0,1],[1,0],[0,1],[1,0],[0,1]]
	# Random pupil wil good start point
	seed2 = [[1,0],[0,1],[0,1],[1,0],[1,0]],[[1,1],[0,1],[1,0],[1,0],[0,1]],[[1,0],[0,1],[0,1],[1,0],[1,0]],[[0,1],[0,1],[1,0],[1,0],[0,1]]

	seeds = [seed1, seed2]
	num_fails = 25
	num_spirals = 15
	outputs = []
	values = []
	with open("output.txt", 'w') as output_file:
		for i in range(num_spirals):
			if i < len(seeds):
				setting, value = optimise(seeds[i], num_fails, i)
				line = "{:.4f} \n{} \n\n".format(value, setting)
				output_file.write(line)
			else:
				setting, value = optimise(None, num_fails, i)
				line = "{:.4f} \n{} \n\n".format(value, setting)
				output_file.write(line)
				 
