import numpy as np
from classy import Class

def compute_shear_spectrum(params):
    """
    Function to compute the weak lensing shear power spectrum using CLASS.
    
    Args:
    params (dict): A dictionary containing the cosmological parameters.
                   Example: {'h': 0.67, 'Omega_b': 0.045, 'Omega_cdm': 0.27,
                             'A_s': 2.1e-9, 'n_s': 0.96, 'tau_reio': 0.06}
    
    Returns:
    ell (np.array): Multipoles (l) values.
    cl_shear (np.array): Weak lensing shear power spectrum (C_l^shear).
    """
    # Create a CLASS instance
    cosmo = Class()
    cosmo.empty()

    # Set up the input parameters for CLASS, adding weak lensing options
    cosmo_arguments = {
        'P_k_max_h/Mpc': 1.5, 
        'ln10^{10}A_s': 3.0, 
        'N_ur': 3.04, 
        'h': 0.701,
        'omega_b': 0.035*0.701**2, 
        'non linear': ' halofit ', 
        'YHe': 0.24, 
        'k_pivot': 0.05,
        'n_s': 0.96, 
        'tau_reio': 0.084, 
        'z_max_pk': 0.5, 
        'output': ' mPk ',
        'omega_cdm': 0.215*0.701**2, 
        'T_cmb': 2.726
        }
    cosmo.set(cosmo_arguments)

    # Run CLASS
    cosmo.compute(['lensing'])

    # Extract the lensing shear power spectrum (C_l^shear)
    cl_shear = cosmo.lensed_cl(3000)['pp']  # 'pp' is the lensing potential spectrum

    # Get the multipoles (ell) values
    ell = np.arange(2, len(cl_shear) + 2)

    # Return the results
    return ell, cl_shear


# Example usage
params = {
    'h': 0.67,
    'Omega_b': 0.045,
    'Omega_cdm': 0.27,
    'A_s': 2.1e-9,
    'n_s': 0.96,
    'tau_reio': 0.06
}

ell, cl_shear = compute_shear_spectrum(params)

# Print the first few results
print("Multipoles (ell):", ell[:5])
print("Shear Power Spectrum (C_l^shear):", cl_shear[:5])
