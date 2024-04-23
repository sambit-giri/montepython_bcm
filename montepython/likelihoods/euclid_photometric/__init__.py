########################################################
# Euclid photometric likelihood
########################################################
# written by Maike Doerenkamp in 2020
# following the recipe of 1910.09237 (Euclid preparation: VII. Forecast validation for Euclid
# cosmological probes)
# (adapted from euclid_lensing likelihood)
# edited by Lena Rathmann in 2021

from montepython.likelihood_class import Likelihood

from scipy.integrate import trapz
from scipy.interpolate import interp1d, RectBivariateSpline
from copy import copy

import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from time import time

from functools import partial

import numpy as np
import warnings
from scipy.special import erf

class euclid_photometric(Likelihood):

    def __init__(self, path, data, command_line):
        self.debug_save  = False
        Likelihood.__init__(self, path, data, command_line)

        # Force the cosmological module to store Pk for redshifts up to
        # max(self.z) and for k up to k_max
        self.need_cosmo_arguments(data, {'output': 'mPk, dTk'})
        self.need_cosmo_arguments(data, {'z_max_pk': self.zmax})
        self.need_cosmo_arguments(data, {'P_k_max_1/Mpc': 1.5*self.k_max_h_by_Mpc})

        # Define array of l values, evenly spaced in logscale
        if self.lmax_WL > self.lmax_GC:
            self.l_WL = np.logspace(np.log10(self.lmin), np.log10(self.lmax_WL), num=self.lbin, endpoint=True)
            self.idx_lmax = int(np.argwhere(self.l_WL >= self.lmax_GC)[0])
            self.l_GC = self.l_WL[:self.idx_lmax+1]
            self.l_XC = self.l_WL[:self.idx_lmax+1]
            self.l_array = 'WL'

            self.ells_WL = np.array(range(self.lmin,self.lmax_WL+1))
            self.ell_jump = self.lmax_GC - self.lmin +1
            self.ells_GC = self.ells_WL[:self.ell_jump]
            self.ells_XC = self.ells_GC
        else:
            self.l_GC = np.logspace(np.log10(self.lmin), np.log10(self.lmax_GC), num=self.lbin, endpoint=True)
            self.idx_lmax = int(np.argwhere(self.l_GC >= self.lmax_WL)[0])
            self.l_WL = self.l_GC[:self.idx_lmax+1]
            self.l_XC = self.l_GC[:self.idx_lmax+1]
            self.l_array = 'GC'

            self.ells_GC = np.array(range(self.lmin,self.lmax_GC+1))
            self.ell_jump = self.lmax_WL - self.lmin +1
            self.ells_WL = self.ells_GC[:self.ell_jump]
            self.ells_XC = self.ells_WL

        if self.debug_save :
            np.savetxt('ls.txt',self.l_GC)

        #########################################
        # Find distribution of n(z) in each bin #
        #########################################

        # Create the array that will contain the z boundaries for each bin.

        self.z_bin_edge = np.array([self.zmin, 0.418, 0.560, 0.678, 0.789, 0.900, 1.019, 1.155, 1.324, 1.576, self.zmax])
        self.z_bin_center = np.array([(self.z_bin_edge[i]+self.z_bin_edge[i+1])/2 for i in range(self.nbin)])

        # Fill array of discrete z values
        self.z = np.linspace(self.zmin, self.zmax, num=self.nzmax)

        # Fill distribution for each bin (convolving with photo_z distribution)
        # n_i = int n(z) dz over bin
        self.eta_z = np.zeros((self.nzmax, self.nbin), 'float64')
        self.photoerror_z = np.zeros((self.nzmax, self.nbin), 'float64')
        for Bin in range(self.nbin):
            for nz in range(self.nzmax):
                z = self.z[nz]
                self.photoerror_z[nz,Bin] = self.photo_z_distribution(z,Bin+1)
                self.eta_z[nz, Bin] = self.photoerror_z[nz,Bin] * self.galaxy_distribution(z)
        if self.debug_save : np.savetxt('./photoz.txt',self.photoerror_z) ## agrees
        if self.debug_save : np.savetxt('./unnorm_nofz.txt',self.eta_z) ## agrees
        # integrate eta(z) over z (in view of normalizing it to one)
        self.eta_norm = np.zeros(self.nbin, 'float64')
        #norm = np.array([trapz([self.photo_z_distribution(z1, i+1) for z1 in zint],dx=dz) for i in range(self.nbin)])
        for Bin in range(self.nbin):
            #self.eta_z[:,Bin] /= trapz(self.eta_z[:,Bin],dx=self.zmax/self.nzmax)
            self.eta_z[:,Bin] /= trapz(self.eta_z[:,Bin],self.z[:])

        if self.debug_save : np.savetxt('./n.txt',self.eta_z)
        # the normalised galaxy distribution per bin (dimensionless)
        #print('eta_z: ', self.eta_z)
        # the number density of galaxies per bin in inv sr
        self.n_bar = self.gal_per_sqarcmn * (60.*180./np.pi)**2
        self.n_bar /= self.nbin

        ###########################
        # Add nuisance parameters #
        ###########################

        if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
            self.bias_names = []
            for ibin in range(self.nbin):
                self.bias_names.append('bias_'+str(ibin+1))
            self.nuisance += self.bias_names

        if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
            self.nuisance += ['aIA', 'etaIA', 'betaIA']

            # Read the file for the IA- contribution
            lum_file = open(os.path.join(self.data_directory,'scaledmeanlum_E2Sa.dat'), 'r')
            content = lum_file.readlines()
            zlum = np.zeros((len(content)))
            lum = np.zeros((len(content)))
            for index in range(len(content)):
                line = content[index]
                zlum[index] = line.split()[0]
                lum[index] = line.split()[1]
            self.lum_func = interp1d(zlum, lum,kind='linear')

        if self.use_BCemu or (self.fit_different_data and self.data_use_BCemu):
            self.nuisance += ['log10Mc']
            self.nuisance += ['nu_Mc']

        #############
        # Read data #
        #############

        # If the file exists, read the fiducial values
        self.fid_values_exist = False
        fid_file_path = os.path.join(self.data_directory, self.fiducial_file+'.npz')
        if os.path.exists(fid_file_path):
            self.Cov_observ_dict = dict()
            self.fid_values_exist = True
            fid_file = np.load(fid_file_path)
            if fid_file['probe'] != self.probe:
                warnings.warn("Probes in fiducial file does not match the probes asked for.\n The fiducial Probe is {} and the probe asked for is {}.\n Please proceed with caution".format(fid_file['probe'],self.probe))
            try:
                if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
                    l_WL = fid_file['ells_LL']
                    if not np.isclose(l_WL,self.l_WL).all():
                        raise Exception("Maximum multipole of WL has changed between fiducial and now.\n Fiducial lmax = {}, new lmax = {}. \n Please remove old fiducial and generate a new one".format(max(l_WL),max(self.l_WL)))
                    Cl_LL = fid_file['Cl_LL']
                    inter_LL = interp1d(l_WL,Cl_LL,axis=0, kind='cubic',fill_value="extrapolate")(self.ells_WL)
                    self.Cov_observ_dict["Cov_observ"] = inter_LL

                if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
                    l_GC = fid_file['ells_GG']
                    if not np.isclose(l_GC,self.l_GC).all():
                        raise Exception("Maximum multipole of GC has changed between fiducial and now.\n Fiducial lmax = {}, new lmax = {}. \n Please remove old fiducial and generate a new one".format(max(l_GC),max(self.l_GC)))
                    Cl_GG = fid_file['Cl_GG']
                    inter_GG = interp1d(l_GC,Cl_GG,axis=0, kind='cubic',fill_value="extrapolate")(self.ells_GC)
                    self.Cov_observ_dict["Cov_observ"] = inter_GG

                if 'WL_GCph_XC' in self.probe:
                    l_XC = fid_file['ells_GL']
                    Cl_GL = fid_file['Cl_GL']
                    inter_GL = interp1d(l_XC,Cl_GL,axis=0, kind='cubic',fill_value="extrapolate")(self.ells_XC)
                    inter_LG = np.transpose(inter_GL,(0,2,1))
                    if self.lmax_WL > self.lmax_GC:
                        self.Cov_observ_dict["Cov_observ"] = np.block([[inter_LL[:self.ell_jump,:,:],inter_LG],[inter_GL,inter_GG]])
                        self.Cov_observ_dict["Cov_observ_high"] = inter_LL[self.ell_jump:,:,:]
                    else:
                        self.Cov_observ_dict["Cov_observ"] = np.block([[inter_LL,inter_LG],[inter_GL,inter_GG[:self.ell_jump,:,:]]])
                        self.Cov_observ_dict["Cov_observ_high"] = inter_GG[self.ell_jump:,:,:]

            except KeyError:
                raise KeyError("The probe asked for in the survey specifications is not in the fiducial file. \n Please remove old fiducial and generate a new one")
        else:
            if self.fit_different_data:
                self.use_BCemu = self.data_use_BCemu
                self.use_tracer = self.data_use_tracer

        return


    def galaxy_distribution(self, z):
        """
        Galaxy distribution returns the function D(z) from the notes

        Modified by S. Clesse in March 2016 to add an optional form of n(z) motivated by ground based exp. (Van Waerbeke et al., 2013)
        See google doc document prepared by the Euclid IST - Splinter 2
        """
        zmean = 0.9
        z0 = zmean/np.sqrt(2)

        galaxy_dist = (z/z0)**2*np.exp(-(z/z0)**(1.5))

        return galaxy_dist


    def photo_z_distribution(self, z, bin):
        """
        Photo z distribution

        z:      physical galaxy redshift
        zph:    measured galaxy redshift
        """
        c0, z0, sigma_0 = 1.0, 0.1, 0.05
        cb, zb, sigma_b = 1.0, 0.0, 0.05
        f_out = 0.1

        if bin == 0 or bin >= 11:
            return None

        term1 = cb*f_out*    erf((0.707107*(z-z0-c0*self.z_bin_edge[bin - 1]))/(sigma_0*(1+z)))
        term2 =-cb*f_out*    erf((0.707107*(z-z0-c0*self.z_bin_edge[bin    ]))/(sigma_0*(1+z)))
        term3 = c0*(1-f_out)*erf((0.707107*(z-zb-cb*self.z_bin_edge[bin - 1]))/(sigma_b*(1+z)))
        term4 =-c0*(1-f_out)*erf((0.707107*(z-zb-cb*self.z_bin_edge[bin    ]))/(sigma_b*(1+z)))
        return (term1+term2+term3+term4)/(2*c0*cb)

    def loglkl(self, cosmo, data):

        if self.printtimes:
            t_start = time()

        # One wants to obtain here the relation between z and r, this is done
        # by asking the cosmological module with the function z_of_r
        self.r = np.zeros(self.nzmax, 'float64')
        self.dzdr = np.zeros(self.nzmax, 'float64')

        self.r, self.dzdr = cosmo.z_of_r(self.z)

        # H(z)/c in 1/Mpc
        self.H_z = self.dzdr
        # H_0/c in 1/Mpc
        H0 = cosmo.h()/2997.92458

        self.kmin_in_inv_Mpc = self.k_min_h_by_Mpc * cosmo.h()
        self.kmax_in_inv_Mpc = self.k_max_h_by_Mpc * cosmo.h()

        if self.l_array == 'WL':
            l = self.l_WL
        if self.l_array == 'GC':
            l = self.l_GC
        k =(l[:,None]+0.5)/self.r

        if self.printtimes:
            t_init = time()
            print("Initalisation Time:", t_init-t_start)

        ######################
        # Get power spectrum #
        ######################
        # [P(k)] = Mpc^3
        # Get power spectrum P(k=(l+1/2)/r,z) from cosmological module
        Pk_m_nl_grid, k_grid, z_grid = cosmo.get_pk_and_k_and_z()
        Pk_m_l_grid, _, _ = cosmo.get_pk_and_k_and_z(nonlinear=False)

        z_grid = z_grid[::-1]
        Pk_m_nl_grid = np.flip(Pk_m_nl_grid,axis=1)
        Pk_m_l_grid = np.flip(Pk_m_l_grid,axis=1)

        Pk_m_nl_spline = RectBivariateSpline(k_grid,z_grid,Pk_m_nl_grid)
        Pk_m_l_spline = RectBivariateSpline(k_grid,z_grid,Pk_m_l_grid)

        Pk_m_nl = np.zeros_like(k, 'float64')
        Pk_m_l  = np.zeros_like(k, 'float64')

        for iz, zi in enumerate(self.z):
            pknn_mask = np.where((k[:,iz]>self.kmin_in_inv_Mpc) & (k[:,iz]<self.kmax_in_inv_Mpc))

            Pk_m_nl[pknn_mask, iz] = np.reshape(Pk_m_nl_spline(k[pknn_mask,iz],zi),Pk_m_nl[pknn_mask, iz].shape)
            Pk_m_l [pknn_mask, iz] = np.reshape(Pk_m_l_spline(k[pknn_mask,iz],zi),Pk_m_l [pknn_mask, iz].shape)

        Pk_WL = copy(Pk_m_nl)
        Pk_GC = copy(Pk_m_nl)

        if self.printtimes:
            t_power = time()
            print("Power spectrum obtained in:", t_power-t_init)

        ########################
        # Boosts and Emulators #
        ########################

        if self.use_BCemu:
            import baryonic_feedback
            # choice to only effect the Lensing power spectrum by baryonic effects
            Pk_WL *= baryonic_feedback.get_boost_baryonic_feedback(cosmo,data,self,k,self.z)

        Pk_XC = np.sqrt(Pk_GC * Pk_WL)

        if self.printtimes:
            t_nonlinear = time()
            print("Nonlinear effects and Boosts obtained in", t_nonlinear - t_power)

        #####################
        # Get Growth Faktor #
        #####################

        if self.scale_dependent_f == False:
            D_z= np.ones_like((self.nzmax), 'float64')
            for iz, zi in enumerate(self.z):
                D_z[iz] = cosmo.scale_independent_growth_factor(zi)
            D_z= D_z[None,:]

        elif self.scale_dependent_f ==True:
            D_z =np.ones_like(k, 'float64')
            for iz, zi in enumerate(self.z):
                pknn_mask = np.where((k[:,iz]>self.kmin_in_inv_Mpc) & (k[:,iz]<self.kmax_in_inv_Mpc))
                D_z[pknn_mask,iz] = np.reshape(np.sqrt(Pk_m_l_spline(k[pknn_mask,iz],zi) / Pk_m_l_spline(k[pknn_mask,iz],0)),D_z[pknn_mask,iz].shape)

        if self.printtimes:
            t_growth = time()
            print("Growthfactor obtained in", t_growth-t_nonlinear)

        ################################################
        # Window functions W_L(l,z,bin) and W_G(z,bin) #
        ################################################
        # in units of [W] = 1/Mpc

        if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:

            # cosmic shearing
            integral = 3./2.*H0**2. *cosmo.Omega_m()*self.r[None,:,None]*(1.+self.z[None,:,None])*self.eta_z.T[:,None,:]*(1-self.r[None,:,None]/self.r[None,None,:])
            W_gamma  = np.trapz(np.triu(integral),self.z,axis=-1).T

            W_IA = self.eta_z *self.H_z[:,None]
            C_IA = 0.0134
            A_IA = data.mcmc_parameters['aIA']['current']*(data.mcmc_parameters['aIA']['scale'])
            eta_IA = data.mcmc_parameters['etaIA']['current']*(data.mcmc_parameters['etaIA']['scale'])
            beta_IA = data.mcmc_parameters['betaIA']['current']*(data.mcmc_parameters['betaIA']['scale'])

            F_IA = (1.+self.z)**eta_IA * (self.lum_func(self.z))**beta_IA
            W_L = W_gamma[None,:,:] - A_IA*C_IA*cosmo.Omega_m()*F_IA[None,:,None]/D_z[:,:,None] *W_IA[None,:,:]

        if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
            # Compute window function W_G(z) of galaxy clustering for each bin:

            bias_values = np.zeros((self.nbin),'float64')
            for ibin in range(self.nbin):
                bias_values[ibin] = data.mcmc_parameters[self.bias_names[ibin]]['current']*data.mcmc_parameters[self.bias_names[ibin]]['scale']

            if self.bias_model == 'binned_constant' :
                galaxy_bias = bias_values[None,None,:]

            elif self.bias_model == 'binned' :
                biaspars = dict()
                for ibin in range(self.nbin):
                    biaspars['b'+str(ibin+1)] = bias_values[ibin]
                brang = range(1,len(self.z_bin_edge))
                last_bin_num = brang[-1]

                def binbis(zz):
                    lowi = np.where( self.z_bin_edge <= zz )[0][-1]
                    if zz >= self.z_bin_edge[-1] and lowi == last_bin_num:
                        bii = biaspars['b'+str(last_bin_num)]
                    else:
                        bii = biaspars['b'+str(lowi+1)]
                    return bii

                vbinbis = np.vectorize(binbis)
                galaxy_bias = vbinbis(self.z)[None,:,None]

            elif self.bias_model == 'interpld' :
                biasfunc = interp1d(self.z_bin_center, bias_values, bounds_error=False, fill_value="extrapolate")
                galaxy_bias = biasfunc(self.z)[None,:,None]

            # Handle the neutrino induced scale dependant bias following the prescription of the neurino paper
            if self.use_tracer == 'clustering':
                import neutrino_bias_model
                galaxy_bias = np.ones(self.lbin)[:,None,None] * galaxy_bias
                galaxy_bias *= np.sqrt(neutrino_bias_model.get_boost_neutrino_bias(cosmo, data, self, Pk_m_nl_grid, k, self.z)[:,:, None])

            W_G = np.zeros((self.nzmax, self.nbin), 'float64')
            W_G = galaxy_bias * self.H_z[None,:,None] * self.eta_z[None,:,:]

        if self.printtimes:
            t_window = time()
            print("window function obtained in:", t_window-t_nonlinear)

        ###########
        # Calc Cl #
        ###########
        # dimensionless

        nell_WL = len(self.l_WL)
        nell_GC = len(self.l_GC)
        nell_XC = len(self.l_XC)

        # the indices are ell, z, bin_i, bin_j in the int and ell, bin_i, bin_j in the Ceeell
        if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
            Cl_LL_int = W_L[:,:,:, None] * W_L[:,:, None,:] * Pk_WL[:,:, None, None] / self.H_z[None,:, None, None] / self.r[None,:, None, None] / self.r[None,:, None, None]
            Cl_LL     = trapz(Cl_LL_int,self.z,axis=1)[:nell_WL,:,:]

        if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
            Cl_GG_int = W_G[:,:,:, None] * W_G[:,:, None,:] * Pk_GC[:,:, None, None] / self.H_z[None,:, None, None] / self.r[None,:, None, None] / self.r[None,:, None, None]
            Cl_GG     = trapz(Cl_GG_int,self.z,axis=1)[:nell_GC,:,:]

        if 'WL_GCph_XC' in self.probe:
            Cl_LG_int = W_L[:,:,:, None] * W_G[:,:, None,:] * Pk_XC[:,:, None, None] / self.H_z[None,:, None, None] / self.r[None,:, None, None] / self.r[None,:, None, None]
            Cl_LG     = trapz(Cl_LG_int,self.z,axis=1)[:nell_XC,:,:]
            Cl_GL     = np.transpose(Cl_LG,(0,2,1))

        if self.printtimes:
            t_cell = time()
            print("Cell calculated in:", t_cell-t_window)

        ####################
        # Plot Pk and Cl's #
        ####################

        # do you want to save the power spectrum?
        if self.save_PS:
            debug_file_path = os.path.join( self.data_directory, 'euclid_photometric_Pkz.npz')
            # loading the file yields P(k,z), k(ell,z), z
            np.savez(debug_file_path, Pk_WL=Pk_WL, Pk_GC=Pk_GC, k=k, z=self.z)
            print("Printed P(k,z)")

        # do you want to save the Ceeell?
        if self.save_Cell:
            debug_file_path = os.path.join(self.data_directory, 'euclid_photometric_Cls.npz')
            if 'WL_GCph_XC' in self.probe:
                np.savez(debug_file_path, ells_LL=self.l_WL, ells_GG=self.l_GC, ells_GL=self.l_XC, Cl_LL = Cl_LL, Cl_GG = Cl_GG, Cl_GL = Cl_GL)
            if 'WL' in self.probe:
                np.savez(debug_file_path, ells_LL=self.l_WL, Cl_LL = Cl_LL)
            if 'GCph' in self.probe:
                np.savez(debug_file_path, ells_GG=self.l_GC, Cl_GG = Cl_GG)

        if self.printtimes:
            t_debug = time()
            print("Debug options obtained in:" , t_debug-t_cell)

        #########
        # Noise #
        #########
        # dimensionless

        self.noise = {
           'LL': self.rms_shear**2./self.n_bar,
           'LG': 0.,
           'GL': 0.,
           'GG': 1./self.n_bar}

        # add noise to Ceeells after saving to better compare
        for i in range(self.nbin):
            if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
                Cl_LL[:,i,i] += self.noise['LL']
            if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
                Cl_GG[:,i,i] += self.noise['GG']
            if 'WL_GCph_XC' in self.probe:
                Cl_GL[:,i,i] += self.noise['GL']
                Cl_LG[:,i,i] += self.noise['LG']

        #######################
        # Create fiducial file
        #######################

        if self.fid_values_exist is False:
            # Store the values now, and exit.
            fid_file_path = os.path.join(self.data_directory, self.fiducial_file)
            fiducial_cosmo = dict()
            for key, value in data.mcmc_parameters.items():
                    fiducial_cosmo[key] = value['current']*value['scale']

            if 'WL_GCph_XC' in self.probe:
                np.savez(fid_file_path,fid_cosmo=fiducial_cosmo, probe=self.probe, ells_LL=self.l_WL, ells_GG=self.l_GC, ells_GL=self.l_XC, Cl_LL = Cl_LL, Cl_GG = Cl_GG, Cl_GL = Cl_GL)
            if 'WL' in self.probe:
                np.savez(fid_file_path,fid_cosmo=fiducial_cosmo, probe=self.probe, ells_LL=self.l_WL, Cl_LL = Cl_LL)
            if 'GCph' in self.probe:
                np.savez(fid_file_path,fid_cosmo=fiducial_cosmo, probe=self.probe, ells_GG=self.l_GC, Cl_GG = Cl_GG)

            warnings.warn(
                "Writing fiducial model in %s, for %s likelihood\n" % (
                    self.data_directory+'/'+self.fiducial_file, self.name))
            return 1j

        #############
        # Spline Cl #
        #############
        # Find C(l) for every integer l

        Cov_theory_dict = dict()
        if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
            inter_LL = interp1d(self.l_WL,Cl_LL,axis=0, kind='cubic',fill_value="extrapolate")(self.ells_WL)
            Cov_theory_dict["Cov_theory"] = inter_LL
            ells = self.ells_WL

        if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
            inter_GG = interp1d(self.l_GC,Cl_GG,axis=0, kind='cubic',fill_value="extrapolate")(self.ells_GC)
            Cov_theory_dict["Cov_theory"] = inter_GG
            ells = self.ells_GC

        if 'WL_GCph_XC' in self.probe:
            inter_GL = interp1d(self.l_XC,Cl_GL,axis=0, kind='cubic',fill_value="extrapolate")(self.ells_XC)
            inter_LG = np.transpose(inter_GL,(0,2,1))
            if self.lmax_WL > self.lmax_GC:
                Cov_theory_dict["Cov_theory"] = np.block([[inter_LL[:self.ell_jump,:,:],inter_LG],[inter_GL,inter_GG]])
                Cov_theory_dict["Cov_theory_high"] = inter_LL[self.ell_jump:,:,:]
                ells = self.ells_WL
            else:
                Cov_theory_dict["Cov_theory"] = np.block([[inter_LL,inter_LG],[inter_GL,inter_GG[:self.ell_jump,:,:]]])
                Cov_theory_dict["Cov_theory_high"] = inter_GG[self.ell_jump:,:,:]
                ells = self.ells_GC

        if self.printtimes:
            t_spline = time()
            print("Covariance obtained in:", t_spline-t_debug)

        ######################
        # Compute likelihood
        ######################

        T_Rerr_dict = dict()
        T_Rerr_dict["T_Rerr"] = np.zeros_like(Cov_theory_dict["Cov_theory"])
        if 'WL_GCph_XC' in self.probe:
            T_Rerr_dict["T_Rerr_high"] = np.zeros_like(Cov_theory_dict["Cov_theory_high"])

        pcompute_chisq =  partial(self.compute_chisq, ells = ells, Cov_observ_dict = self.Cov_observ_dict, Cov_theory_dict = Cov_theory_dict, T_Rerr_dict = T_Rerr_dict)

        eps_l = np.zeros_like(ells)
        if self.theoretical_error != False:

            ######################
            # Theoretical errors #
            ######################
            import theoretical_errors

            El_dict = theoretical_errors.get_covariance_error(cosmo, data, self, k, Pk_WL, Pk_GC, Pk_XC, W_L, W_G)
            T_Rerr_dict.update(theoretical_errors.spline_error(self, El_dict))
            eps_l = theoretical_errors.minimize_chisq(self, self.compute_chisq, ells, self.Cov_observ_dict, Cov_theory_dict, T_Rerr_dict)

        chi2 = pcompute_chisq(eps_l)

        if self.printtimes:
            t_lkl = time()
            print("Likelihood calculated in:" ,t_lkl-t_spline)
            print("Total time taken:", t_lkl-t_start)

        return -chi2/2.

    # Comopute the log likelihood for a given set of multipoles
    def compute_chisq(self, eps_l, ells, Cov_observ_dict, Cov_theory_dict, T_Rerr_dict):

        Cov_observ = Cov_observ_dict["Cov_observ"]
        Cov_theory = Cov_theory_dict["Cov_theory"]
        T_Rerr = T_Rerr_dict["T_Rerr"]

        nbin = Cov_observ.shape[1]
        ell_jump = Cov_observ.shape[0]

        shifted_Cov = Cov_theory + eps_l[:ell_jump, None, None] * T_Rerr
        dtilde_the = np.linalg.det(shifted_Cov)

        d_obs = np.linalg.det(Cov_observ)
        dtilde_mix = np.zeros_like(dtilde_the)
        for i in range(nbin):
            newCov = np.copy(shifted_Cov)
            newCov[:, i] = Cov_observ[:, :, i]
            dtilde_mix += np.linalg.det(newCov)

        N = np.ones_like(ells) * nbin

        # if the probe is 3x2pt calculate the part with no cross correlation
        if "WL_GCph_XC" in self.probe:

            Cov_observ_high = Cov_observ_dict["Cov_observ_high"]
            Cov_theory_high = Cov_theory_dict["Cov_theory_high"]
            T_Rerr_high = T_Rerr_dict["T_Rerr_high"]

            nbin = Cov_observ_high.shape[1]

            shifted_Cov_high = Cov_theory_high + eps_l[ell_jump:, None, None] * T_Rerr_high
            dtilde_the_high = np.linalg.det(shifted_Cov_high)

            d_obs_high = np.linalg.det(Cov_observ_high)
            dtilde_mix_high = np.zeros_like(dtilde_the_high)
            for i in range(nbin):
                newCov = np.copy(shifted_Cov_high)
                newCov[:, i] = Cov_observ_high[:, :, i]
                dtilde_mix_high += np.linalg.det(newCov)

            N[ell_jump:] = nbin
            dtilde_the = np.concatenate([dtilde_the, dtilde_the_high])
            d_obs = np.concatenate([d_obs, d_obs_high])
            dtilde_mix = np.concatenate([dtilde_mix, dtilde_mix_high])

        return np.sum((2 * ells + 1) * self.fsky * ((dtilde_mix / dtilde_the) + np.log(dtilde_the / d_obs) - N) + np.power(eps_l, 2))
