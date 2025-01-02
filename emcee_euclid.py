import numpy as np
import emcee

INPUT  = 'input/euclid_photometric_w0waMN.param'
CHAINS = 'chains/euclid_photometric'
COVMAT = 'covmat/euclid_photometric_w0waMN.covmat'

euclid_photo_fidu = np.load('data/euclid_photometric_fiducial.npz')
print(f'Keys of the shear data file: {list(euclid_photo_fidu.keys())}')
