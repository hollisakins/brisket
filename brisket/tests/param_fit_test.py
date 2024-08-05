
fit = dict(
    ID': 756434,
    method': 'MultiNest', # 'lm' or 'MultiNest' or 'nautilus'
    n_live': 100, 
    n_posterior': 500, 
    out_wav_units': 'um', 
    out_sed_units': 'uJy', # nJy, uJy, mJy, Jy, or ergscma

)
    },
    'photometry':{
        'file': '/data/COSMOS-Web/catalogs/COSMOS-Web_supercatalog_pm_v0.7.fits',
        'phot_columns': ['f_auto_f115w','f_auto_f150w','f_auto_f277w','f_auto_f444w'],
        'phot_err_columns': ['e_auto_f115w','e_auto_f150w','e_auto_f277w','e_auto_f444w'],
        'filt_list': ['f115w','f150w','f277w','f444w'],
        'phot_units': 'uJy',
    },
    # [fit.spectroscopy]
    #     file = 'spectra/msa_id_0_spec_x1d.fits' # spectra are typically stored in an individual file per-object
    #     # BRISKET will look in the 0th and 1st FITS extensions for a BinTableHDU
    #     wav_column = 'wave' # column name for spectrum wavelengths
    #     wav_units = "um" # units of wavelengths
    #     spec_column = 'flux' # column name for spectrum fluxes
    #     spec_err_column = 'fluxerr' # column name for spectrum flux errors
    #     spec_units = "ergscma" # nJy, uJy, mJy, Jy, or ergscma
    #     R_curve = 'prism' # optional -- convolve model with "line spread function" defined by a resolution curve
    #     f_LSF = 1.5 # optional -- scaling parameter for the line spread function




[base] # basic physical parameters that apply to the entire model
    redshift = {low=3, high=8, prior='Gaussian', mu=5, sigma=0.5}
    igm      = 'Inoue14' # Madau95, Inoue14
    test     = 5
    # damping  = "n" # need specify NH? 
    # MWdust   = "n" # need specify coords?

[galaxy1] # galaxy SED model, including stellar/nebular emission and dust attenuation/emission via energy balance
    model    = 'BC03'
    # imf      = "Chabrier"
    logMstar = {low=8, high=10} # log of stellar mass in Msun
    metallicity = 0.1 # metallicity in Z/Zsun

    # sfh      = "delayed" 
    # age      = 0.2
    # tau      = 0.1
    
    # sfh       = "continuity" # or "bursty-continuity"
    # bin_edges = [0, 10, 50]
    # n_bins    = 6
    # z_max     = 20

    sfh       = 'constant' # constant SF in one bin
    age_min   = 0.0 # Gyr
    age_max   = 0.1 # Gyr
    
    [galaxy1.nebular] 
        logU = -2
        fesc = 0.5

    [galaxy1.dust_atten] 
        type = "Calzetti" # "Calzetti", "Cardelli", "SMC", "Salim", "CF00"
        Av   = [1,5]
        eta  = 1
        logfscat = [-3,-1]
        
# [agn]
#     # type = "full"
#     # logLbol = 45
#     type = 'dblplw'
#     Muv = -21
#     beta1 = -2
#     beta2 = -2
#     wav1 = 3880
#     [agn.nebular]
#         type = "qsogen" # "manual", "sdss", "Temple21"
#         eline_type = 0
#         scale_eline = 0
#         scale_halpha = 0
#         scale_lya = 0
#         scale_nlr = 0 
#         scale_oiii = 0

#     [agn.dust_atten]
#         type = "QSO" # "Calzetti", "Cardelli", "SMC", "Salim", "CF00", "QSO"
#         Av = 3
#         # logfscat = -2
        

    # [Xray]
}