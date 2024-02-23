from montepython.likelihood_class import Likelihood
import os
import numpy as np
import warnings
from math import log, log10
import io_mp
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline as inter
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from scipy import interpolate
from scipy.signal import savgol_filter



class euclid_spectroscopic(Likelihood):
    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)
        self.path_euclid_pk = path
        self.data_euclid_pk = data
        self.command_line_euclid_pk = command_line

        # minimum requirements for CLASS
        self.need_cosmo_arguments(data, {"output": "mPk"})
        self.need_cosmo_arguments(data, {"z_max_pk": self.zmax})
        self.need_cosmo_arguments(
            data, {"P_k_max_1/Mpc": 51}
        )  # need high k for dewiggling

        # the entire likelihood code is using units of [1/Mpc] for k.
        
        #################
        # define redshift bins
        #################

        self.z_edges = np.array([0.90, 1.10, 1.30, 1.50, 1.80])
        self.z_mean = np.array([1.00, 1.20, 1.40, 1.65])
        self.z = np.concatenate((self.z_edges, self.z_mean))
        self.z.sort()
        self.z_central = 1.2
        self.z = np.append(self.z, self.z_central)
        # add the mean redshift of the survey as self.z[-1]

        self.nbin = len(self.z_mean)
        self.nz = len(self.z)
        self.mu_fid = np.linspace(-1, 1, self.mu_size)

        ################
        # Noise spectrum
        ################

        # If the file exists, initialize the fiducial values
        self.fid_values_exist = False

        self.H_fid = np.zeros(self.nbin, "float64")
        self.D_A_fid = np.zeros(self.nbin, "float64")
        self.sigma_v_fid = np.zeros(self.nbin, "float64")
        self.sigma_p_fid = np.zeros(self.nbin, "float64")
        self.P_obs_fid = np.zeros(
            (self.k_size, self.nbin, self.mu_size), "float64"
        )  

        if self.scale_dependent_growth_factor_f is False:
            self.f_fid = np.zeros(self.nbin, "float64")
            self.f_cb_fid = np.zeros(self.nbin, "float64")
        else:
            self.f_fid = np.zeros((self.k_size, self.nbin, self.mu_size), "float64")
            self.f_cb_fid = np.zeros((self.k_size, self.nbin, self.mu_size), "float64")

        fid_file_path = os.path.join(self.data_directory, self.fiducial_file)

        if os.path.exists(fid_file_path):
            self.fid_values_exist = True
            with open(fid_file_path, "r") as fid_file:
                line = fid_file.readline()
                while line.find("#") != -1:
                    line = fid_file.readline()
                while line.find("\n") != -1 and len(line) == 1:
                    line = fid_file.readline()
                for index_z in range(self.nbin):
                    self.H_fid[index_z] = float(line.split()[0])
                    self.D_A_fid[index_z] = float(line.split()[1])
                    line = fid_file.readline()
                self.h_fid = float(line)
                line = fid_file.readline()
                for index_k in range(self.k_size):
                    for index_z in range(self.nbin):
                        arr = np.array([float(x) for x in line.split()])
                        self.P_obs_fid[index_k, index_z, :] = arr[:]
                        line = fid_file.readline()
                self.sigma_v_fid = np.array([float(x) for x in line.split()])
                line = fid_file.readline()
                self.sigma_p_fid = np.array([float(x) for x in line.split()])
                if self.scale_dependent_growth_factor_f is False:
                    line = fid_file.readline()
                    self.f_fid = np.array([float(x) for x in line.split()])
                else:
                    for index_k in range(self.k_size):
                        for index_z in range(self.nbin):
                            line = fid_file.readline()
                            self.f_fid[index_k, index_z, :] = np.array(
                                [float(x) for x in line.split()]
                            )
                if self.scale_dependent_growth_factor_f is False:
                    line = fid_file.readline()
                    self.f_cb_fid = np.array([float(x) for x in line.split()])
                else:
                    for index_k in range(self.k_size):
                        for index_z in range(self.nbin):
                            line = fid_file.readline()
                            self.f_cb_fid[index_k, index_z, :] = np.array(
                                [float(x) for x in line.split()]
                            )
        else:
            # the fiducial file will be created in the loglkl() function below
            # therefore we need to extract the h fiducial value from the data 
            try:
                self.h_fid = data.parameters["h"][0]
            except KeyError as kk:
                print("{:s} fiducial value should be present in the .param file".format(str(kk)))
                raise

        # TODO calculate the fiducial volums (not just for Euclid fiducial)
        self.gal_density_fid = (
            np.array([6.86e-4, 5.58e-4, 4.21e-4, 2.61e-4]) * self.h_fid**3
        )
        self.P_shot_fid = 1 / (self.gal_density_fid)  
        self.V_fid = (
            np.array([7.94, 9.15, 10.05, 16.22]) * 1e9 / (self.h_fid**3)
        )  

        if self.NonLinError == "superpessimistic":
            self.nuisance += [
                "sigma_v0",
                "sigma_v1",
                "sigma_v2",
                "sigma_v3",
                "sigma_p0",
                "sigma_p1",
                "sigma_p2",
                "sigma_p3",
            ]

        return

    def loglkl(self, cosmo, data):
        np.set_printoptions(precision=16)

        # Define the k values in (1/Mpc) for the integration (from kmin to kmax), at which
        # the spectrum will be computed (and stored for the fiducial model)
        self.k_fid = np.logspace(
            log10(self.kmin * self.h_fid),
            log10(self.kmax * self.h_fid),
            num=self.k_size,
        )

        self.h = cosmo.h()

        r, H = cosmo.z_of_r(self.z_mean)  
        D_A = np.zeros(self.nbin, "float64")
        for i in range(len(D_A)):
            D_A[i] = cosmo.angular_distance(self.z_mean[i])  

        sigma_r = self.spectroscopic_error / H  
        if self.spectroscopic_error_z_dependent:
            sigma_r *= 1 + self.z_mean

        lnbsigma8 = np.array(
            [
                data.mcmc_parameters["lnbsigma8_0"]["current"]
                * data.mcmc_parameters["lnbsigma8_0"]["scale"],
                data.mcmc_parameters["lnbsigma8_1"]["current"]
                * data.mcmc_parameters["lnbsigma8_1"]["scale"],
                data.mcmc_parameters["lnbsigma8_2"]["current"]
                * data.mcmc_parameters["lnbsigma8_2"]["scale"],
                data.mcmc_parameters["lnbsigma8_3"]["current"]
                * data.mcmc_parameters["lnbsigma8_3"]["scale"],
            ]
        )
        bsigma8 = np.exp(lnbsigma8)

        #############
        # AP effect #
        #############

        self.k = np.zeros((self.k_size, self.nbin, self.mu_size), "float64")
        if not self.fid_values_exist:
            q_orth = np.ones((self.nbin,))
            q_parr = np.ones((self.nbin,))
        else:
            q_orth = self.D_A_fid / D_A  
            q_parr = H / self.H_fid  

        self.k = (
            self.k_fid[:, None, None]
            * q_orth[None, :, None]
            * np.sqrt(
                1.0
                + self.mu_fid[None, None, :] ** 2
                * (q_parr[None, :, None] ** 2 / (q_orth[None, :, None] ** 2) - 1)
            )
        )
        
        # if you want to reproduce the Euclid IST:F wrong results with the h-bug rescale here the k as in the comment
        #    self.k *= self.h / self.h_fid  # "A rescaling so nice we had to do it twice"

        self.mu = (
            self.mu_fid[None, :]
            * q_parr[:, None]
            / q_orth[:, None]
            / np.sqrt(
                1.0
                + self.mu_fid[None, :] ** 2
                * (q_parr[:, None] ** 2 / (q_orth[:, None] ** 2) - 1)
            )
        )

        ###############################
        # Compute the growth factor f #
        ###############################

        if self.scale_dependent_growth_factor_f is False:
            f = np.zeros((self.nbin), "float64")
            f_cb = np.zeros((self.nbin), "float64")
            for index_z in range(self.nbin):
                f[index_z] = cosmo.scale_independent_growth_factor_f(
                    self.z_mean[index_z]
                )
                f_cb[index_z] = cosmo.scale_independent_growth_factor_f_cb(
                    self.z_mean[index_z]
                )
        else:
            f = np.zeros((self.k_size, self.nbin, self.mu_size), "float64")
            f_cb = np.zeros((self.k_size, self.nbin, self.mu_size), "float64")
            Pk_cb_l, k, z = cosmo.get_pk_and_k_and_z(
                only_clustering_species=True, nonlinear=False
            )
            z = z[::-1]
            pk_flipped = np.flip(Pk_cb_l, axis=1).T
            D_growth_cb_zk = RectBivariateSpline(
                z, k, np.sqrt(pk_flipped / pk_flipped[0, :])
            )
            D_cb_z = np.array(
                [UnivariateSpline(z, D_growth_cb_zk(z, kk), s=0) for kk in k]
            )
            f_cb_z = np.array(
                [
                    -(1 + z) / D_cb_zk(z) * (D_cb_zk.derivative())(z)
                    for D_cb_zk in D_cb_z
                ]
            )
            f_g_cb_kz = RectBivariateSpline(z, k, f_cb_z.T)

            Pk_mm_l, _, _ = cosmo.get_pk_and_k_and_z(nonlinear=False)
            pk_flipped = np.flip(Pk_mm_l, axis=1).T
            D_growth_mm_zk = RectBivariateSpline(
                z, k, np.sqrt(pk_flipped / pk_flipped[0, :])
            )
            D_mm_z = np.array(
                [UnivariateSpline(z, D_growth_mm_zk(z, kk), s=0) for kk in k]
            )
            f_mm_z = np.array(
                [
                    -(1 + z) / D_mm_zk(z) * (D_mm_zk.derivative())(z)
                    for D_mm_zk in D_mm_z
                ]
            )
            f_g_mm_kz = RectBivariateSpline(z, k, f_mm_z.T)

            for index_z, zz in enumerate(self.z_mean):
                for index_k, kk in enumerate(self.k_fid):
                    f[index_k, index_z, :] = f_g_mm_kz(
                        zz, self.k[index_k, index_z, :], grid=False
                    )
                    f_cb[index_k, index_z, :] = f_g_cb_kz(
                        zz, self.k[index_k, index_z, :], grid=False
                    )

        if self.fid_values_exist is True:
            f_fid = self.f_fid

        # Compute the power spectrum for a given k range
        k_long_min = self.dewiggling_k_min_invMpc
        k_long_max = self.dewiggling_k_max_invMpc
        num_k_long = (
            round(log(k_long_max / k_long_min) / log(1 + self.dewiggling_dlnk)) + 1
        )
        k_long = np.zeros(num_k_long)
        k_long = k_long_min * (1 + self.dewiggling_dlnk) ** np.arange(num_k_long)
        pk_long = np.zeros((len(k_long), self.nbin), "float64")  
        pk_cb_long = np.zeros((len(k_long), self.nbin), "float64")  

        for index_k in range(len(k_long)):
            for index_z in range(self.nbin):
                kval = k_long[index_k]
                zval = self.z_mean[index_z]
                pk_long[index_k, index_z] = np.log(cosmo.pk_lin(kval, zval))
                pk_cb_long[index_k, index_z] = np.log(cosmo.pk_cb_lin(kval, zval))

        sigma8_cb_of_z = np.array(
            [cosmo.sigma_cb(R=8 / cosmo.h(), z=zi) for zi in self.z_mean]
        )
        sigma8_of_z = np.array(
            [cosmo.sigma(R=8 / cosmo.h(), z=zi) for zi in self.z_mean]
        )

        if self.use_tracer == "matter":
            f_tracer = f
            pk_tracer_long = pk_long
            sigma8_tracer_of_z = sigma8_of_z
        elif self.use_tracer == "clustering":
            f_tracer = f_cb
            pk_tracer_long = pk_cb_long
            sigma8_tracer_of_z = sigma8_cb_of_z
        else:
            print("\n")
            warnings.warn(
                "Tracer not recognised." "Please choose 'matter' or 'clustering'."
            )
            raise ValueError

        if self.dewiggle == "savgol_filter":
            # same dewiggle method as the Euclid Collaboration

            pk_tracer_nobao_long = np.empty_like(pk_tracer_long)
            window_len = int(self.savgol_width / np.log(1.0 + self.dewiggling_dlnk))

            for index_z in range(self.nbin):
                pk_tracer_nobao_long[:, index_z] = savgol_filter(
                    pk_tracer_long[:, index_z], window_len, self.savgol_order
                )
        else:
            print("\n")
            warnings.warn(
                "Dewiggle method not recognised."
                "Please choose 'savgol_filter'"
            )
            raise ValueError

        pk_lin = np.zeros(
            (self.k_size, self.nbin, self.mu_size), "float64"
        )  # in (Mpc)**3
        pk_tracer_lin = np.zeros(
            (self.k_size, self.nbin, self.mu_size), "float64"
        )  # in (Mpc)**3
        pk_tracer_nobao = np.zeros(
            (self.k_size, self.nbin, self.mu_size), "float64"
        )  # in (Mpc)**3

        pk_sigmavp = np.zeros((self.k_size, self.nbin), "float64")
        k_sigmavp = np.geomspace(0.001, 5, self.k_size)

        for index_z in range(self.nbin):
            pk_lin_spline = interpolate.interp1d(k_long, pk_long[:, index_z])
            pk_tracer_lin_spline = interpolate.interp1d(
                k_long, pk_tracer_long[:, index_z]
            )
            pk_tracer_nobao_spline = interpolate.interp1d(
                k_long, pk_tracer_nobao_long[:, index_z]
            )
            # This is open for debate if we should use P_m or P_cb for sigma_v . 
            # The idea behind our choice of P_m is, that if it should be the same as sigma_p then we should use
            # P_m as this is a gravitational effect and not related to the tracer bias. 
            # We could use diffrent sigma_v and sigma_p using P_cb and P_m respectively.
            pk_sigmavp[:, index_z] = np.exp(pk_lin_spline(k_sigmavp[:]))
            for index_mu in range(self.mu_size):
                pk_lin[:, index_z, index_mu] = np.exp(
                    pk_lin_spline(self.k[:, index_z, index_mu])
                )
                pk_tracer_lin[:, index_z, index_mu] = np.exp(
                    pk_tracer_lin_spline(self.k[:, index_z, index_mu])
                )
                pk_tracer_nobao[:, index_z, index_mu] = np.exp(
                    pk_tracer_nobao_spline(self.k[:, index_z, index_mu])
                )

        if self.NonLinError == "linear":
            # linear setting
            sigma_v = np.zeros((self.nbin), "float64")
            sigma_p = np.zeros((self.nbin), "float64")
        elif self.NonLinError == "pessimistic" or self.NonLinError == "optimistic":
            # nonlinear pessimistic and optimistic setting
            if self.fid_values_exist is False:
                sigma_v = np.sqrt(
                    1 / (6 * np.pi**2) * simpson(pk_sigmavp, x=k_sigmavp, axis=0)
                )
                sigma_p = sigma_v
            else:
                sigma_v = self.sigma_v_fid
                sigma_p = self.sigma_p_fid
        elif self.NonLinError == "superpessimistic":
            # nonlinear superpessimistic setting, varying freely sigma_p and sigma_v at each bin
            sigma_v = np.array(
                [
                    data.mcmc_parameters["sigma_v0"]["current"]
                    * data.mcmc_parameters["sigma_v0"]["scale"],
                    data.mcmc_parameters["sigma_v1"]["current"]
                    * data.mcmc_parameters["sigma_v1"]["scale"],
                    data.mcmc_parameters["sigma_v2"]["current"]
                    * data.mcmc_parameters["sigma_v2"]["scale"],
                    data.mcmc_parameters["sigma_v3"]["current"]
                    * data.mcmc_parameters["sigma_v3"]["scale"],
                ]
            )
            sigma_p = np.array(
                [
                    data.mcmc_parameters["sigma_p0"]["current"]
                    * data.mcmc_parameters["sigma_p0"]["scale"],
                    data.mcmc_parameters["sigma_p1"]["current"]
                    * data.mcmc_parameters["sigma_p1"]["scale"],
                    data.mcmc_parameters["sigma_p2"]["current"]
                    * data.mcmc_parameters["sigma_p2"]["scale"],
                    data.mcmc_parameters["sigma_p3"]["current"]
                    * data.mcmc_parameters["sigma_p3"]["scale"],
                ]
            )

        else:
            print("\n")
            warnings.warn(
                "Treatment of Non-linear error not recognised."
                "Please choose 'linear', 'pessimistic' , 'optimistic' , 'superpessimistic' "
            )
            raise ValueError

        if self.fid_values_exist is False:
            fz_fog = f
            fz_kaiser = f_tracer
            fz_dewigg = f
        else:
            fz_fog = f_fid
            fz_kaiser = f_tracer
            fz_dewigg = f_fid

        if self.scale_dependent_growth_factor_f is False:
            fz_fog = fz_fog[None, :, None]
            fz_kaiser = fz_kaiser[None, :, None]
            fz_dewigg = fz_dewigg[None, :, None]

        F_AP = q_parr * q_orth**2

        # Linear Kaiser factor F_Kaiser(k,z,mu)
        # following the presciption of arXiv:1807.04672v2 in f*sigma8 factor we will use f_cb
        F_Kaiser = (
            bsigma8[None, :, None]
            + fz_kaiser * sigma8_tracer_of_z[None, :, None] * self.mu[None, :, :] ** 2
        )

        F_FOG = 1 / (
            1 + (fz_fog * self.k * self.mu[None, :, :] * sigma_p[None, :, None]) ** 2
        )

        F_z = np.exp(-((self.k * self.mu[None, :, :] * sigma_r[None, :, None]) ** 2))

        fac = np.exp(
            -self.k**2
            * sigma_v[None, :, None] ** 2
            * (1.0 + self.mu[None, :, :] ** 2 * (-1.0 + (1 + fz_dewigg) ** 2))
        )
        P_tracer_dw = pk_tracer_lin * fac + pk_tracer_nobao * (1 - fac)

        self.P_obs = (
            F_AP[None, :, None]
            * (F_Kaiser) ** 2
            * F_FOG
            * P_tracer_dw
            / (sigma8_tracer_of_z[None, :, None] ** 2)
            * F_z
        )

        # For plotting P_m and P_nw:
        # To debug the individual components
        Pk_debug = False
        if Pk_debug == True:
            debug_file_path = os.path.join(self.data_directory, "euclid_GC_k.npy")
            with open(debug_file_path, "w") as debug_file:
                np.savetxt(debug_file, self.k[:, :, 0])
            debug_file_path = os.path.join(self.data_directory, "euclid_GC_mu.npy")
            with open(debug_file_path, "w") as debug_file:
                np.savetxt(debug_file, self.mu)
            debug_file_path = os.path.join(self.data_directory, "euclid_GC_Pk_m.npy")
            with open(debug_file_path, "w") as debug_file:
                np.savetxt(debug_file, pk_lin[:, :, 0])
            debug_file_path = os.path.join(self.data_directory, "euclid_GC_Pk_cb.npy")
            with open(debug_file_path, "w") as debug_file:
                np.savetxt(debug_file, pk_tracer_lin[:, :, 0])
            debug_file_path = os.path.join(self.data_directory, "euclid_GC_Pk_nw.npy")
            with open(debug_file_path, "w") as debug_file:
                np.savetxt(debug_file, pk_tracer_nobao[:, :, 0])
            debug_file_path = os.path.join(self.data_directory, "euclid_GC_Pk_dw.npy")
            with open(debug_file_path, "w") as debug_file:
                np.savetxt(
                    debug_file, P_tracer_dw[:, :, 0] / sigma8_cb_of_z[None, :] ** 2
                )
            debug_file_path = os.path.join(self.data_directory, "euclid_GC_Pk.npy")
            with open(debug_file_path, "w") as debug_file:
                np.savetxt(debug_file, self.P_obs[:, :, 0])

        if Pk_debug:
            debug_file_path = os.path.join(self.data_directory, "euclid_GCsp_obs.npy")
            np.savez(
                debug_file_path,
                k_grid=self.k,
                mu_grid=self.mu,
                z_grid=self.z_mean,
                P_obs=self.P_obs,
                F_AP=F_AP,
                F_Kaiser=F_Kaiser,
                F_FOG=F_FOG,
                F_z=F_z,
                P_dd=P_tracer_dw,
            )

        # Shot noise spectrum:
        self.P_shot = np.zeros((self.nbin), "float64")
        self.P_shot = np.array(
            [
                data.mcmc_parameters["P_shot0"]["current"]
                * data.mcmc_parameters["P_shot0"]["scale"],
                data.mcmc_parameters["P_shot1"]["current"]
                * data.mcmc_parameters["P_shot1"]["scale"],
                data.mcmc_parameters["P_shot2"]["current"]
                * data.mcmc_parameters["P_shot2"]["scale"],
                data.mcmc_parameters["P_shot3"]["current"]
                * data.mcmc_parameters["P_shot3"]["scale"],
            ]
        )

        self.P_obs += self.P_shot[None, :, None] + self.P_shot_fid[None, :, None]

        # If the fiducial model does not exist, recover the power spectrum and
        # store it, then exit.

        if self.fid_values_exist is False:
            fid_file_path = os.path.join(self.data_directory, self.fiducial_file)
            with open(fid_file_path, "w") as fid_file:
                fid_file.write("# Fiducial parameters")
                for key, value in io_mp.dictitems(data.mcmc_parameters):
                    fid_file.write(
                        ", %s = %.20g" % (key, value["current"] * value["scale"])
                    )
                fid_file.write("\n")
                for index_z in range(self.nbin):
                    fid_file.write("%.20g %.20g" % (H[index_z], D_A[index_z]))
                    fid_file.write("\n")
                fid_file.write("%.20g" % (cosmo.h()))
                fid_file.write("\n")
                for index_k in range(self.k_size):
                    for index_z in range(self.nbin):
                        fid_file.write(
                            " ".join(
                                ["%.20g" % x for x in self.P_obs[index_k, index_z, :]]
                            )
                        )
                        fid_file.write("\n")
                fid_file.write(" ".join(["%.20g" % x for x in sigma_v[:]]))
                fid_file.write("\n")
                fid_file.write(" ".join(["%.20g" % x for x in sigma_p[:]]))
                fid_file.write("\n")
                if self.scale_dependent_growth_factor_f is False:
                    fid_file.write(" ".join(["%.20g" % x for x in f[:]]))
                else:
                    for index_k in range(self.k_size):
                        for index_z in range(self.nbin):
                            fid_file.write(
                                " ".join(["%.20g" % x for x in f[index_k, index_z, :]])
                            )
                            fid_file.write("\n")
                if self.scale_dependent_growth_factor_f is False:
                    fid_file.write(" ".join(["%.20g" % x for x in f_cb[:]]))
                else:
                    for index_k in range(self.k_size):
                        for index_z in range(self.nbin):
                            fid_file.write(
                                " ".join(
                                    ["%.20g" % x for x in f_cb[index_k, index_z, :]]
                                )
                            )
                            fid_file.write("\n")
            print("\n")
            warnings.warn(
                "Writing fiducial model in %s, for %s likelihood\n"
                % (self.data_directory + "/" + self.fiducial_file, self.name)
            )
            return 1j

        chi2 = 0.0
        for index_z in range(self.nbin):
            mu_integrant = np.zeros((self.mu_size), "float64")
            for index_mu in range(self.mu_size):
                k_integrant = self.array_integrand(index_z, index_mu)
                mu_integrant[index_mu] = simpson(k_integrant[:], x=self.k_fid[:])
            chi2_of_z = simpson(mu_integrant[:], x=self.mu_fid[:])
            chi2 += chi2_of_z

        print("euclid_spectroscopic: chi2 =", chi2)
        return -chi2 / 2.0

    def array_integrand(self, index_z, index_mu):
        integrand = (
            self.V_fid[index_z]
            * self.k_fid[:] ** 2
            / (8 * np.pi**2)
            * (self.P_obs[:, index_z, index_mu] - self.P_obs_fid[:, index_z, index_mu])
            ** 2
            / (self.P_obs[:, index_z, index_mu] ** 2)
        )
        return integrand
