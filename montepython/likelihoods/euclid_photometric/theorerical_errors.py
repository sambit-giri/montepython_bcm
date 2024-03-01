import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from functools import partial

"""
This function is to calculate the covariance matrix error.
The main ingredient, the relative power spectrum error, is calculated in the first half of the function.
"""
def get_covariance_error(cosmo, data, lkl, k, Pk_WL, Pk_GC, Pk_XC, W_WL, W_GC):
    nell_WL = len(lkl.l_WL)
    nell_GC = len(lkl.l_GC)
    nell_XC = len(lkl.l_XC)

    #calculate the relative theoretical error
    alpha = np.zeros_like(k)
    if  lkl.theoretical_error == 'nonlinear':

        k_sigma = cosmo.nonlinear_scale(lkl.z, lkl.nzmax)
        for index_z in range(lkl.nzmax):
            pknn_mask = np.where((k[:,index_z]>lkl.kmin_in_inv_Mpc) & (k[:,index_z]<lkl.kmax_in_inv_Mpc))
            alpha[pknn_mask, index_z] = np.reshape(np.log(1.+k[pknn_mask, index_z]/k_sigma[index_z])/(1+ np.log(1.+k[pknn_mask, index_z]/k_sigma[index_z])),alpha[pknn_mask, index_z].shape)

    elif lkl.theoretical_error == 'react_cast':

        # obtained fits from ratio of ReACT with Forge at HS6.
        def A_plateau(z):
            Delta = 0.0356204
            Base = 0.029017
            Temperature = 0.06154762
            potential = 0.56233084
            return Delta / (np.exp((z-potential) / Temperature)+1)+Base

        def k_plateau(z):
            kinf = 0.20253502
            pos = 1.55716999
            sigma = 0.55169064
            return kinf*np.exp(np.tanh((z-pos)/sigma))

        def smoothish_step(x):
            return (np.power(x,2)+x)/(np.power(x,2)+x+1)

        for index_z, z_value in enumerate(lkl.z):
            pknn_mask = np.where((k[:,index_z]>lkl.kmin_in_inv_Mpc) & (k[:,index_z]<lkl.kmax_in_inv_Mpc))
            alpha[pknn_mask,index_z] = np.reshape(A_plateau(z_value) * smoothish_step(k[pknn_mask,index_z]/k_plateau(z_value)), alpha[pknn_mask, index_z].shape)

    else:
        raise ValueError("Theoretical error model not recognized.")

    return_dict = dict()

    # calculate the covariance matrix error
    if 'WL' in lkl.probe or 'WL_GCph_XC' in lkl.probe:
        El_LL_int = W_WL[:,:,:,None] * W_WL[:,:,None,:] * Pk_WL[:,:,None,None] / lkl.H_z[None,:,None,None] / lkl.r[None,:,None,None] / lkl.r[None,:,None,None] * alpha[:,:,None,None]
        El_LL     = trapz(El_LL_int,lkl.z,axis=1)[:nell_WL,:,:]
        return_dict["El_LL"] = El_LL

    if 'GCph' in lkl.probe or 'WL_GCph_XC' in lkl.probe:
        El_GG_int = W_GC[None,:,:,None] * W_GC[None,: , None, :] * Pk_GC[:,:,None,None] / lkl.H_z[None,:,None,None] / lkl.r[None,:,None,None] / lkl.r[None,:,None,None] * alpha[:,:,None,None]
        El_GG     = trapz(El_GG_int,lkl.z,axis=1)[:nell_GC,:,:]
        return_dict["El_GG"] = El_GG

    if 'WL_GCph_XC' in lkl.probe:
        El_LG_int = W_WL[:,:,:,None] * W_GC[None,: , None, :] * Pk_XC[:,:,None,None] / lkl.H_z[None,:,None,None] / lkl.r[None,:,None,None] / lkl.r[None,:,None,None] * alpha[:,:,None,None]
        El_LG     = trapz(El_LG_int,lkl.z,axis=1)[:nell_XC,:,:]
        El_GL     = np.transpose(El_LG,(0,2,1))
        return_dict["El_LG"] = El_LG
        return_dict["El_GL"] = El_GL

    return return_dict

"""
Spline the covariance error for all integer multipole and construct the covariance error matrix similar
"""
def spine_error(lkl, El_dict):

    return_dict = dict()

    if 'WL' in lkl.probe or 'WL_GCph_XC' in lkl.probe:
        inter_LL = interp1d(lkl.l_WL,El_dict["El_LL"],axis=0, kind='cubic',fill_value="extrapolate")(lkl.ells_WL)
        norm = np.sqrt(lkl.lmax_WL - lkl.lmin + 1)
        return_dict["T_Rerr"] =  norm * inter_LL
    if 'GCph' in lkl.probe or 'WL_GCph_XC' in lkl.probe:
        inter_GG = interp1d(lkl.l_GC,El_dict["El_GG"],axis=0, kind='cubic',fill_value="extrapolate")(lkl.ells_GC)
        norm = np.sqrt(lkl.lmax_GC - lkl.lmin + 1)
        return_dict["T_Rerr"] =  norm * inter_GG
    if 'WL_GCph_XC' in lkl.probe:
        inter_GL = interp1d(lkl.l_XC,El_dict["El_GL"],axis=0, kind='cubic',fill_value="extrapolate")(lkl.ells_XC)
        inter_LG = np.transpose(inter_GL,(0,2,1))
        norm = np.sqrt(np.maximum(lkl.lmax_WL,lkl.lmax_GC) - lkl.lmin + 1)
        if lkl.lmax_WL > lkl.lmax_GC:
            return_dict["T_Rerr"] = norm * np.block([[inter_LL[:lkl.ell_jump,:,:],inter_LG],[inter_GL,inter_GG]])
            return_dict["T_Rerr_high"] = norm * inter_LL[lkl.ell_jump:,:,:]
        else:
            return_dict["T_Rerr"] = norm * np.block([[inter_LL,inter_LG],[inter_GL,inter_GG[:lkl.ell_jump,:,:]]])
            return_dict["T_Rerr_high"] = norm * inter_GG[lkl.ell_jump:,:,:]

    return return_dict

"""
This function minimizes the chi^2 w.r.t. to epsilon nuisance parameters
"""
def minimize_chisq(lkl, compute_chisq, ells, Cov_observ_dict, Cov_theory_dict, T_Rerr_dict):
    if lkl.minimize_Terr_binned:

        # re-bin the integer multipoles in log space
        if 'WL' in lkl.probe or 'GCph' in lkl.probe:
            if 'WL' in lkl.probe:
                ells_binned = np.unique(np.geomspace(lkl.lmin,lkl.lmax_WL, lkl.lbin, dtype = np.uint64))
            if 'GCph' in lkl.probe:
                ells_binned = np.unique(np.geomspace(lkl.lmin,lkl.lmax_GC, lkl.lbin, dtype = np.uint64))
            index_low = ells_binned - lkl.lmin

        elif 'WL_GCph_XC' in lkl.probe:
            ells_binned = np.unique(np.geomspace(lkl.lmin,np.maximum(lkl.lmax_WL,lkl.lmax_GC), lkl.lbin, dtype = np.uint64))
            index_low = ells_binned[np.where(ells_binned < lkl.ell_jump)] - lkl.lmin
            index_high = ells_binned[np.where(ells_binned >= lkl.ell_jump)] - lkl.ell_jump - lkl.lmin
        
        Cov_observ_dict_binned = dict()
        Cov_theory_dict_binned = dict()
        T_Rerr_dict_binned = dict()
        eps_binned = np.zeros_like(ells_binned)

        Cov_observ_dict_binned["Cov_observ"] = Cov_observ_dict["Cov_observ"][index_low]
        Cov_theory_dict_binned["Cov_theory"] = Cov_theory_dict["Cov_theory"][index_low]
        T_Rerr_dict_binned["T_Rerr"] = T_Rerr_dict["T_Rerr"][index_low]

        if 'WL_GCph_XC' in lkl.probe:
            Cov_observ_dict_binned["Cov_observ_high"] = Cov_observ_dict["Cov_observ_high"][index_high]
            Cov_theory_dict_binned["Cov_theory_high"] = Cov_theory_dict["Cov_theory_high"][index_high]
            T_Rerr_dict_binned["T_Rerr_high"] = T_Rerr_dict["T_Rerr_high"][index_high]

        pcompute_chisq = partial(compute_chisq, ells = ells_binned, Cov_observ_dict = Cov_observ_dict_binned, Cov_theory_dict = Cov_theory_dict_binned, T_Rerr_dict = T_Rerr_dict_binned)
        pjac = partial(jac, ells = ells_binned, Cov_observ_dict = Cov_observ_dict_binned, Cov_theory_dict = Cov_theory_dict_binned, T_Rerr_dict = T_Rerr_dict_binned,lkl=lkl)

        res = minimize(pcompute_chisq, eps_binned, tol=1e-3, method='Newton-CG',jac=pjac, hess='3-point')
        eps_binned = res.x

        eps_l = interp1d(ells_binned, eps_binned, kind='cubic',fill_value="extrapolate")(ells)
    else:

        pcompute_chisq = partial(compute_chisq,ells=ells, Cov_observ_dict= Cov_observ_dict, Cov_theory_dict= Cov_theory_dict, T_Rerr_dict= T_Rerr_dict)
        pjac = partial(jac,ells=ells, Cov_observ_dict= Cov_observ_dict, Cov_theory_dict= Cov_theory_dict, T_Rerr_dict= T_Rerr_dict,lkl=lkl)

        res = minimize(pcompute_chisq, eps_l, tol=1e-3, method='Newton-CG',jac=pjac, hess='3-point')
        eps_l = res.x
    
    return eps_l

"""
This function gives the analytical derivative of the chi^2 with respect to the nuisance parameters epsilon.
Similar to the chi^2
"""
def jac(eps_l, lkl, ells, Cov_observ_dict, Cov_theory_dict, T_Rerr_dict):

    Cov_observ = Cov_observ_dict["Cov_observ"]
    Cov_theory = Cov_theory_dict["Cov_theory"]
    T_Rerr = T_Rerr_dict["T_Rerr"]

    # find the  #redshift bins X #probes from the covariance matrix
    nbin = Cov_observ.shape[1]
    # find the multipole with the jump
    ell_jump = Cov_observ.shape[0]

    shifted_Cov = Cov_theory + eps_l[:ell_jump, None, None] * T_Rerr
    dtilde_the = np.linalg.det(shifted_Cov)
    inv_shifted_Cov = np.linalg.inv(shifted_Cov)
    dprime_the = dtilde_the * np.trace(
        np.matmul(inv_shifted_Cov, T_Rerr), axis1=1, axis2=2
    )

    d_obs = np.linalg.det(Cov_observ)

    dtilde_mix = np.zeros_like(dtilde_the)
    dprime_mix = np.zeros_like(dtilde_the)
    for i in range(nbin):
        newCov = np.copy(shifted_Cov)
        newCov[:, i] = Cov_observ[:, :, i]
        dnewCov = np.linalg.det(newCov)
        dtilde_mix += dnewCov

        newCovprime = np.copy(T_Rerr)
        newCovprime[:, i] = 0
        inv_newCov = np.linalg.inv(newCov)
        dprime_mix += dnewCov * np.trace(
            np.matmul(inv_newCov, newCovprime), axis1=1, axis2=2
        )

    # if the probe is 3x2pt calculate the part with no cross correlation
    if "WL_GCph_XC" in lkl.probe:

        Cov_observ_high = Cov_observ_dict["Cov_observ_high"]
        Cov_theory_high = Cov_theory_dict["Cov_theory_high"]
        T_Rerr_high = T_Rerr_dict["T_Rerr_high"]

        nbin = Cov_observ_high.shape[1]

        shifted_Cov_high = Cov_theory_high + eps_l[ell_jump:, None, None] * T_Rerr_high
        dtilde_the_high = np.linalg.det(shifted_Cov_high)
        inv_shifted_Cov_high = np.linalg.inv(shifted_Cov_high)
        dprime_the_high = dtilde_the_high * np.trace(
            np.matmul(inv_shifted_Cov_high, T_Rerr_high), axis1=1, axis2=2
        )

        d_obs_high = np.linalg.det(Cov_observ_high)

        dtilde_mix_high = np.zeros_like(dtilde_the_high)
        dprime_mix_high = np.zeros_like(dtilde_the_high)
        for i in range(nbin):
            newCov = np.copy(shifted_Cov_high)
            newCov[:, i] = Cov_observ_high[:, :, i]
            dnewCov_high = np.linalg.det(newCov)
            dtilde_mix_high += dnewCov_high

            newCovprime_high = np.copy(T_Rerr_high)
            newCovprime_high[:, i] = 0
            inv_newCov = np.linalg.inv(newCov)
            dprime_mix_high += dnewCov_high * np.trace(
                np.matmul(inv_newCov, newCovprime_high), axis1=1, axis2=2
            )

        dtilde_the = np.concatenate([dtilde_the, dtilde_the_high])
        dprime_the = np.concatenate([dprime_the, dprime_the_high])
        d_obs = np.concatenate([d_obs, d_obs_high])
        dtilde_mix = np.concatenate([dtilde_mix, dtilde_mix_high])
        dprime_mix = np.concatenate([dprime_mix, dprime_mix_high])

    return (2 * ells + 1) * lkl.fsky * ((dprime_mix + dprime_the) / dtilde_the - (dtilde_mix * dprime_the) / np.power(dtilde_the, 2)) + 2 * eps_l