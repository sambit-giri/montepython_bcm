##################################################################################################
# Interface module between euclid_photometric likelihood and BCemu emulator of baryonic feedback #
##################################################################################################

# BCemu python package available at https://github.com/sambit-giri/BCemu
# Developped by Sambit Giri and Aurel Schneider, see arXiv:2108.08863 (physical model in arXiv:1810.08629)

# Interface with Euclid likelihood witten by J. Schwagereit and Sefa Pamuk (2023)

import BCemu
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy import interpolate
from tqdm import tqdm
from glob import glob 
import pandas as pd

def get_boost_baryonic_feedback(cosmo, data, lkl, k, z):

    print('Baryonic feedback flag:', data.bar_flag)

    if data.bar_flag==0:
        BFC_interpolator = baryonic_feedback_hydrosims(0)

    elif data.bar_flag==2:
        # baryonic feedback modifications are only applied to k>kmin_bfc
        # it is very computationally expensive to call BCemu at every z in self.z, and it is a very smooth function with z,
        # so it is only called at self.BCemu_k_bins points in k and self.BCemu_z_bins points in z and then the result is
        # splined over all z in self.z. For k>kmax_bfc = 12.5 h/Mpc, the maximum k the emulator is trained on, a constant
        # suppression in k is assumed: BFC(k,z) = BFC(12.5 h/Mpc, z).

        if not hasattr(lkl, "bfcemu"):
            lkl.bfcemu = BCemu.BCM_7param(verbose=lkl.verbose_BCemu)

        log10Mc = data.mcmc_parameters['log10Mc']['current'] * data.mcmc_parameters['log10Mc']['scale']
        mu      = data.mcmc_parameters['mu']['current']      * data.mcmc_parameters['mu']['scale']
        thej    = data.mcmc_parameters['thej']['current']    * data.mcmc_parameters['thej']['scale']
        gamma   = data.mcmc_parameters['gamma']['current']   * data.mcmc_parameters['gamma']['scale']
        delta   = data.mcmc_parameters['delta']['current']   * data.mcmc_parameters['delta']['scale']

        eta     = data.mcmc_parameters['eta']['current']     * data.mcmc_parameters['eta']['scale']
        deta    = data.mcmc_parameters['deta']['current']    * data.mcmc_parameters['deta']['scale']

        nu_Mc    = data.mcmc_parameters['nu_Mc']['current']    * data.mcmc_parameters['nu_Mc']['scale']
        nu_mu    = data.mcmc_parameters['nu_mu']['current']    * data.mcmc_parameters['nu_mu']['scale']
        nu_thej  = data.mcmc_parameters['nu_thej']['current']  * data.mcmc_parameters['nu_thej']['scale']
        nu_gamma = data.mcmc_parameters['nu_gamma']['current'] * data.mcmc_parameters['nu_gamma']['scale']
        nu_delta = data.mcmc_parameters['nu_delta']['current'] * data.mcmc_parameters['nu_delta']['scale']

        nu_eta   = data.mcmc_parameters['nu_eta']['current']   * data.mcmc_parameters['nu_eta']['scale']
        nu_deta  = data.mcmc_parameters['nu_deta']['current']  * data.mcmc_parameters['nu_deta']['scale']

        bcemu_dict ={
            'log10Mc' : log10Mc,
            'nu_Mc'   : nu_Mc,
            'mu'      : mu, #0.93,
            'nu_mu'   : nu_mu, #0.0,
            'thej'    : thej, #2.6,
            'nu_thej' : nu_thej, #0.0,
            'gamma'   : gamma, #2.25,
            'nu_gamma': nu_gamma, #0.0,
            'delta'   : delta, #6.4,
            'nu_delta': nu_delta, #0.0,
            'eta'     : eta, #0.15,
            'nu_eta'  : nu_eta, #0.0,
            'deta'    : deta, #0.14,
            'nu_deta' : nu_deta,#0.06
            }

        Ob = cosmo.Omega_b()
        Om = cosmo.Omega_m()

        fb = Ob/Om
        if fb < 0.1 or fb > 0.25:
            if lkl.verbose: print(" /!\ Skipping point because the baryon fraction is out of bounds!")
            return -1e10

        if log10Mc / 3**nu_Mc < 11 or log10Mc / 3**nu_Mc > 15 :
            if lkl.verbose: print(" /!\ Skipping point because BF parameters are out of bounds!")
            return -1e10

        kmin_bfc = 0.035
        kmax_bfc = 12.5
        k_bfc = np.logspace(np.log10(max(kmin_bfc, lkl.k_min_h_by_Mpc)), np.log10(min(kmax_bfc, lkl.k_max_h_by_Mpc)), lkl.BCemu_k_bins)
        # ^ all have units h/Mpc

        z_bfc = np.linspace(lkl.z[0], min(2, lkl.z[-1]), lkl.BCemu_z_bins)
        BFC = np.zeros((lkl.BCemu_k_bins, lkl.BCemu_z_bins))

        for iz, zi in enumerate(z_bfc):
            BFC[:,iz] = lkl.bfcemu.get_boost(zi,bcemu_dict,k_bfc,fb)

        BFC_interpolator = RectBivariateSpline(k_bfc*cosmo.h(), z_bfc, BFC)
        # ^ gets passed k in units 1/Mpc

    elif data.bar_flag in [11,12,13,15,17,18,19,20,21,22,23,24,25]:
        BFC_interpolator = baryonic_feedback_hydrosims(data.bar_flag)
    
    else:
        BFC_interpolator = None

    boost_m_nl_BCM = np.ones_like(k, 'float64')
    for iz, zi in enumerate(z):
        pknn_mask = np.where((k[:,iz]>lkl.kmin_in_inv_Mpc) & (k[:,iz]<lkl.kmax_in_inv_Mpc))
        boost_m_nl_BCM[pknn_mask, iz] = np.reshape(BFC_interpolator(np.minimum(k[pknn_mask,iz],12.5*cosmo.h()),min(zi, 2))[:,0], boost_m_nl_BCM[pknn_mask, iz].shape)

    return boost_m_nl_BCM


def baryonic_feedback_hydrosims(bar_flag):
    if bar_flag==0:
        print('DMO model')
        file_dmb = 'HydroSims/powtable_OWLS_AGN.dat'
        file_dmo = 'HydroSims/powtable_DMONLY_L100N512.dat'
        OWLS_planck_dmo = read_file(file_dmo)
        OWLS_planck_dmb = read_file(file_dmb)
        interpolator = interpolate.RectBivariateSpline(
                OWLS_planck_dmo['k']*0.70, 
                OWLS_planck_dmo['z'],
                np.ones_like(OWLS_planck_dmb['P']/OWLS_planck_dmo['P']).T, 
                kx=1, ky=1)
        
    elif bar_flag==11:
        print('OWLS model')
        file_dmb = 'HydroSims/powtable_OWLS_AGN.dat'
        file_dmo = 'HydroSims/powtable_DMONLY_L100N512.dat'
        OWLS_planck_dmo = read_file(file_dmo)
        OWLS_planck_dmb = read_file(file_dmb)
        interpolator = interpolate.RectBivariateSpline(
                OWLS_planck_dmo['k']*0.70, 
                OWLS_planck_dmo['z'],
                (OWLS_planck_dmb['P']/OWLS_planck_dmo['P']).T, 
                kx=1, ky=1)
        
    elif bar_flag==12:
        print('Cosmo OWLS 8.0 model')
        file_dmb = 'HydroSims/powtable_C-OWLS_AGN_WMAP7.dat'
        file_dmo = 'HydroSims/powtable_DMONLY_WMAP7_L400N1024.dat'
        cosmoOWLs8p0_wmap7_dmo = read_file(file_dmo)
        cosmoOWLs8p0_wmap7_dmb = read_file(file_dmb)
        interpolator = interpolate.RectBivariateSpline(
                    cosmoOWLs8p0_wmap7_dmo['k']*0.70, 
                    cosmoOWLs8p0_wmap7_dmo['z'],
                    (cosmoOWLs8p0_wmap7_dmb['P']/cosmoOWLs8p0_wmap7_dmo['P']).T, 
                    kx=1, ky=1)
        
    elif bar_flag==13:
        print('Cosmo OWLS 8.5 model')
        file_dmb = 'HydroSims/powtable_C-OWLS_AGN_Theat8.5_Planck2013.dat' #'HydroSims/powtable_C-OWLS_AGN_Theat8.5_WMAP7.dat'
        file_dmo = 'HydroSims/powtable_DMONLY_Planck2013_L400N1024.dat' #'HydroSims/powtable_DMONLY_WMAP7_L400N1024.dat'
        cosmoOWLs8p5_wmap7_dmo = read_file(file_dmo)
        cosmoOWLs8p5_wmap7_dmb = read_file(file_dmb)
        interpolator = interpolate.RectBivariateSpline(
                    cosmoOWLs8p5_wmap7_dmo['k']*0.70, 
                    cosmoOWLs8p5_wmap7_dmo['z'],
                    (cosmoOWLs8p5_wmap7_dmb['P']/cosmoOWLs8p5_wmap7_dmo['P']).T, 
                    kx=1, ky=1)
        
    elif bar_flag==15:
        print('BAHAMAS 7.8 model')
        file_dmb_planck2013 = 'HydroSims/powtable_BAHAMAS_nu0_Planck2013.dat'
        file_dmo_planck2013 = 'HydroSims/powtable_DMONLY_2fluid_nu0_Planck2013_L400N1024.dat'
        bahamas7p8_planck_dmo = read_file(file_dmo_planck2013)
        bahamas7p8_planck_dmb = read_file(file_dmb_planck2013)
        interpolator = interpolate.RectBivariateSpline(
                    bahamas7p8_planck_dmo['k']*0.67, 
                    bahamas7p8_planck_dmo['z'],
                    (bahamas7p8_planck_dmb['P']/bahamas7p8_planck_dmo['P']).T, 
                    kx=1, ky=1)
        
    elif bar_flag==17:
        print('MAGNETICUM model')
        file_dmbs = glob('HydroSims/Magneticum_Pk_box2/Pk_box2_1024_bao_*.txt')
        file_dmos = glob('HydroSims/Magneticum_Pk_box2/Pk_box2_1024_dm_*.txt')
        snap_list = np.array([ff.split('_')[-1].split('.txt')[0] for ff in file_dmbs]).astype(int)
        snap_list = snap_list[np.argsort(-snap_list)]
        snap_zs   = np.loadtxt('HydroSims/Magneticum_Pk_box2/outputs_145_sparse.txt')[:,0]**-1-1
        Sk, zs = [], []
        for s in tqdm(snap_list):
            data_dmb = np.loadtxt('HydroSims/Magneticum_Pk_box2/Pk_box2_1024_bao_{:03d}.txt'.format(s))
            data_dmo = np.loadtxt('HydroSims/Magneticum_Pk_box2/Pk_box2_1024_dm_{:03d}.txt'.format(s))
            Sk.append(data_dmb[:,1]/data_dmo[:,1])
            zs.append(snap_zs[s-1])
        Sk = np.array(Sk)
        zs = np.array(zs)
        Mag1024_info = {'z': zs, 'k': data_dmo[:,0], 'Sk': Sk}
        interpolator = interpolate.RectBivariateSpline(
                    Mag1024_info['k']*0.7,
                    Mag1024_info['z'], 
                    Mag1024_info['Sk'].T, 
                    kx=1, ky=1)

    elif bar_flag==18:
        print('Horizon-AGN model')
        HAGN_info = np.load('HydroSims/HorizonAGN_data.npz')
        interpolator = interpolate.RectBivariateSpline(
                    HAGN_info['k']*0.7, 
                    HAGN_info['z'],
                    HAGN_info['Sk'].T, 
                    kx=1, ky=1)

    elif bar_flag==19:
        print('Illustris-TNG model')
        TNG300_info = np.load('HydroSims/TNG300_data.npz')
        interpolator = interpolate.RectBivariateSpline(
                    TNG300_info['k']*0.677, 
                    TNG300_info['z'],
                    TNG300_info['Sk'].T, 
                    kx=1, ky=1)
            
    elif bar_flag==20:
        print('Illustris model')
        illustris_hlittle = 0.704
        Tb_ill1 = pd.read_csv("HydroSims/logPkRatio_illustris.dat",sep='\s+')
        illustris_info = {'z': np.array([3.5,3.49,3.28,3.08,2.90,2.73,2.44,2.1,2.0,1.82,1.74,1.6,1.41,1.21,1.04,1.0,0.79,0.7,0.6,0.4,0.35,0.2,0.0]),
                    'colz': np.array(['z350', 'z349', 'z328', 'z308', 'z290', 'z273', 'z244', 'z210',
                                        'z200', 'z182', 'z174', 'z160', 'z141', 'z121', 'z104', 'z100', 'z079',
                                        'z070', 'z060', 'z040', 'z035', 'z020', 'z000']),
                    'k': 10**Tb_ill1['logk']}
        arg_zs = np.argsort(illustris_info['z'])
        zs, colz = illustris_info['z'][arg_zs], illustris_info['colz'][arg_zs]
        Sk = []
        for colzi in tqdm(colz):
            data_Sk = 10**Tb_ill1[colzi]
            Sk.append(data_Sk)
        Sk = np.array(Sk)
        interpolator = interpolate.RectBivariateSpline(
                    illustris_info['k']*illustris_hlittle, zs,
                    Sk.T, kx=1, ky=1)
            
    elif bar_flag==21:
        print('MassiveBlackII model')
        MB2_hlittle = 0.701
        Tb_mb2 = pd.read_csv("HydroSims/logPkRatio_mb2.dat",sep='\s+')
        MB2_info = {'z': np.array([3.5,3.25,2.8,2.45,2.1,2.0,1.8,1.7,1.6,1.4,1.2,1.1,1.0,0.8,0.7,0.6,0.4,0.35,0.2,0.0625,0.0]),
                    'colz': np.array(['z350', 'z325', 'z280', 'z245', 'z210', 'z200', 'z180', 'z170',
                                        'z160', 'z140', 'z120', 'z110', 'z100', 'z080', 'z070', 'z060', 'z040',
                                        'z035', 'z020', 'z00625', 'z000']),
                    'k': 10**Tb_mb2['logk']}
        arg_zs = np.argsort(MB2_info['z'])
        zs, colz = MB2_info['z'][arg_zs], MB2_info['colz'][arg_zs]
        Sk = []
        for colzi in tqdm(colz):
            data_Sk = 10**Tb_mb2[colzi]
            Sk.append(data_Sk)
        Sk = np.array(Sk)
        interpolator = interpolate.RectBivariateSpline(
                    MB2_info['k']*MB2_hlittle, zs, 
                    Sk.T, kx=1, ky=1)
            
    elif bar_flag==22:
        print('EAGLE model')
        eagle_hlittle = 0.6777
        Tb_eagle = pd.read_csv("HydroSims/logPkRatio_eagle.dat",sep='\s+')
        eagle_info = {'z': np.array([3.53,3.02,2.48,2.24,2.01,1.74,1.49,1.26,1.0,0.74,0.5,0.27,0.0]),
                    'colz': np.array(['z353', 'z302', 'z248', 'z224', 'z201', 'z174', 'z149', 'z126', 'z100', 
                                        'z074', 'z050', 'z027', 'z000']),
                    'k': 10**Tb_eagle['logk']}
        arg_zs = np.argsort(eagle_info['z'])
        zs, colz = eagle_info['z'][arg_zs], eagle_info['colz'][arg_zs]
        Sk = []
        for colzi in tqdm(colz):
            data_Sk = 10**Tb_eagle[colzi]
            Sk.append(data_Sk)
        Sk = np.array(Sk)
        interpolator = interpolate.RectBivariateSpline(
                    eagle_info['k']*eagle_hlittle, zs, 
                    Sk.T, kx=1, ky=1)
        
    elif bar_flag==23:
        print('Illustris TNG100 model')
        TNG100_hlittle = 0.704
        Tb_TNG100 = pd.read_csv("HydroSims/logPkRatio_TNG100.dat",sep='\s+')
        TNG100_info = {'z': np.array([3.71,3.49,3.28,2.90,2.44,2.1,1.74,1.41,1.04,0.7,0.35,0.18,0.0]),
                    'colz': np.array(['z371', 'z349', 'z328', 'z290', 'z244', 'z210', 'z174', 'z141', 
                                        'z104', 'z070', 'z035', 'z018', 'z000']),
                    'k': 10**Tb_TNG100['logk']}
        arg_zs = np.argsort(TNG100_info['z'])
        zs, colz = TNG100_info['z'][arg_zs], TNG100_info['colz'][arg_zs]
        Sk = []
        for colzi in tqdm(colz):
            data_Sk = 10**Tb_TNG100[colzi]
            Sk.append(data_Sk)
        Sk = np.array(Sk)
        interpolator = interpolate.RectBivariateSpline(
                    TNG100_info['k']*TNG100_hlittle, zs, 
                    Sk.T, kx=1, ky=1)
        
    elif bar_flag==24:
        print('FLAMINGO L1_m9 model')
        FLAMINGO_L1_m9_hlittle = 0.681
        FLAMINGO_L1_m9_zs = np.array([0,0.5,1.0,1.5,2.0])
        file_Sks = [glob('HydroSims/FLAMINGO/FLAMINGO_matter_suppression_L1_m9_z{}.txt'.format(i))[0] 
                        for i in ['p'.join('{:07.3f}'.format(zi).split('.')) for zi in FLAMINGO_L1_m9_zs]]
        Sk, zs = [], []
        for ff in tqdm(file_Sks):
            data_Sk = np.loadtxt(ff)
            Sk.append(data_Sk[:,2])
            zs.append(data_Sk[:,0])
        ks = data_Sk[:,1]
        Sk = np.array(Sk)
        zs = np.array(zs)
        FLAMINGO_L1_m9_info = {'z': FLAMINGO_L1_m9_zs, 'k': ks, 'Sk': Sk}
        FLAMINGO_L1_m9_Sk1  = interpolate.RectBivariateSpline(
                    FLAMINGO_L1_m9_info['z'],
                    FLAMINGO_L1_m9_info['k'], 
                    FLAMINGO_L1_m9_info['Sk'], 
                    kx=1, ky=1)
        ks1 = 10**np.linspace(-2,np.log10(30),40)
        interpolator = interpolate.RectBivariateSpline(
                    ks1, FLAMINGO_L1_m9_info['z'], 
                    FLAMINGO_L1_m9_Sk1(FLAMINGO_L1_m9_info['z'],ks1).T, 
                    kx=1, ky=1)
            
    elif bar_flag==25:
        print('FLAMINGO Jet model')
        FLAMINGO_Jet_hlittle = 0.681
        FLAMINGO_Jet_zs = np.array([0,0.5,1.0,1.5,2.0])
        file_Sks = [glob('HydroSims/FLAMINGO/FLAMINGO_matter_suppression_Jet_z{}.txt'.format(i))[0] 
                        for i in ['p'.join('{:07.3f}'.format(zi).split('.')) for zi in FLAMINGO_Jet_zs]]
        Sk, zs = [], []
        for ff in tqdm(file_Sks):
            data_Sk = np.loadtxt(ff)
            Sk.append(data_Sk[:,2])
            zs.append(data_Sk[:,0])
        ks = data_Sk[:,1]
        Sk = np.array(Sk)
        zs = np.array(zs)
        FLAMINGO_Jet_info = {'z': FLAMINGO_Jet_zs, 'k': ks, 'Sk': Sk}
        FLAMINGO_Jet_Sk1  = interpolate.RectBivariateSpline(
                    FLAMINGO_Jet_info['z'],
                    FLAMINGO_Jet_info['k'], 
                    FLAMINGO_Jet_info['Sk'], 
                    kx=1, ky=1)
        ks1 = 10**np.linspace(-2,np.log10(30),40)
        interpolator = interpolate.RectBivariateSpline(
                    ks1, FLAMINGO_Jet_info['z'], 
                    FLAMINGO_Jet_Sk1(FLAMINGO_Jet_info['z'],ks1).T, 
                    kx=1, ky=1)

    return interpolator

def read_file(filename, z=None):
    # from tqdm import tqdm
    rd = np.loadtxt(filename)
    if z is not None:
        assert z in np.unique(rd[:,0])
        z_args = np.argwhere(rd[:,0]==z).squeeze()
        return {'k': rd[z_args,1], 'P': rd[z_args,2]}
    else:
        zs = np.unique(rd[:,0])
        ks = np.unique(rd[:,1])
        Ps = []
        for z in tqdm(zs):
            out = read_file(filename, z=z)
            Ps.append(out['P'])
        return {'k': ks, 'z': zs, 'P': np.array(Ps)}
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ks = 10**np.linspace(-1,1,20)
    S_owls = baryonic_feedback_hydrosims(11)
    S_Cowl8p0 = baryonic_feedback_hydrosims(12)
    S_Cowl8p5 = baryonic_feedback_hydrosims(13)
    S_bahm = baryonic_feedback_hydrosims(15)
    S_L1m9 = baryonic_feedback_hydrosims(24)

    fig, axs = plt.subplots(1,3,figsize=(12,4.5))
    for ii,zi in enumerate([0,1,2]):
        axs[ii].semilogx(ks, S_owls(ks,zi).flatten(), ls='-', label='OWLS')
        axs[ii].semilogx(ks, S_Cowl8p0(ks,zi).flatten(), ls='-', label='cosmo-OWLS 8.0')
        axs[ii].semilogx(ks, S_Cowl8p5(ks,zi).flatten(), ls='-', label='cosmo-OWLS 8.5')
        axs[ii].semilogx(ks, S_bahm(ks,zi).flatten(), ls='-', label='BAHAMAS 7.8')
        axs[ii].semilogx(ks, S_L1m9(ks,zi).flatten(), ls='-', label='FLAMINGO L1_m9')
    axs[0].legend()
    for ax in axs:
        ax.axis([0.08,12,0.7,1.1])
        ax.set_xlabel(f'$k [h/\mathrm{{Mpc}}]$')
        ax.set_ylabel(f'$S(k)$')
    plt.tight_layout()
    plt.show()

