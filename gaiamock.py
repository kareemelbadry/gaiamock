import numpy as np
import matplotlib.pyplot as plt
import os 
import ctypes 
from astropy.table import Table
import healpy as hp
import joblib

def al_uncertainty_per_ccd_interp(G):
    '''
    This gives the uncertainty *per CCD* (not per FOV transit), taken from Fig 3 of https://arxiv.org/abs/2206.05439
    This is the "EDR3 adjusted" line from that Figure, which is already inflated compared to the formal uncertainties.
    '''
    G_vals =    [ 4,    5,   6,     7,   8.2,  8.4, 10,    11,    12,  13,    14,   15,   16,   17,   18,   19,  20]
    sigma_eta = [0.4, 0.35, 0.15, 0.17, 0.23, 0.13,0.13, 0.135, 0.125, 0.13, 0.15, 0.23, 0.36, 0.63, 1.05, 2.05, 4.1]
    return np.interp(G, G_vals, sigma_eta)


def read_in_C_functions():
    '''
    this function reads in compiled functions from kepler_solve_astrometry.so 
    '''
    script_dir = os.path.dirname(__file__)
    compiled_path = os.path.join(script_dir, 'kepler_solve_astrometry.so')
    if os.path.exists(compiled_path):
        c_funcs = ctypes.CDLL(compiled_path)
    else:
        raise ValueError('You need to compile kepler_solve_astrometry.c!')
    return c_funcs
    
def get_astrometric_chi2(t_ast_yr, psi, plx_factor, ast_obs, ast_err, P, phi_p, ecc, c_funcs, reject_outlier=False):
    '''
    this function takes arrays of astrometric data (t_ast_yr, psi, plx_factor, ast_obs, ast_err), and a set of (P, phi_p, ecc), solves for the best-fit linear parameters, predicts the epoch astrometry, and calculates the chi2. It also returns the best-fit vector of linear parameters. This uses the compiled c function from kepler_solve_astrometry.so. 
    c_funcs comes from read_in_C_functions()
    returns chi2 and an array of 9 linear parameters
    '''
    t_ast_yr_double = t_ast_yr.astype(np.double)
    psi_double = psi.astype(np.double)
    plx_factor_double = plx_factor.astype(np.double)
    ast_obs_double = ast_obs.astype(np.double)
    ast_err_double = ast_err.astype(np.double)
    chi2_array = np.empty(10, dtype = np.double)
    
    if reject_outlier:
        c_funcs.get_chi2_astrometry_reject_outliers(ctypes.c_int(len(t_ast_yr)), ctypes.c_void_p(t_ast_yr_double.ctypes.data), ctypes.c_void_p(psi_double.ctypes.data), ctypes.c_void_p(plx_factor_double.ctypes.data), ctypes.c_void_p(ast_obs_double.ctypes.data), ctypes.c_void_p(ast_err_double.ctypes.data), ctypes.c_double(P), ctypes.c_double(phi_p), ctypes.c_double(ecc), ctypes.c_void_p(chi2_array.ctypes.data))
    else:
        c_funcs.get_chi2_astrometry(ctypes.c_int(len(t_ast_yr)), ctypes.c_void_p(t_ast_yr_double.ctypes.data), ctypes.c_void_p(psi_double.ctypes.data), ctypes.c_void_p(plx_factor_double.ctypes.data), ctypes.c_void_p(ast_obs_double.ctypes.data), ctypes.c_void_p(ast_err_double.ctypes.data), ctypes.c_double(P), ctypes.c_double(phi_p), ctypes.c_double(ecc), ctypes.c_void_p(chi2_array.ctypes.data))
        
        
    chi2 = chi2_array[0]
    mu = chi2_array[1:] # ra_off, pmra, dec_off, pmdec, plx, B, G, A, F
    
    return chi2, mu
    

def get_astrometric_residuals_12par(t_ast_yr, psi, plx_factor, ast_obs, ast_err, theta_array, c_funcs, reject_outlier=False):
    '''
    this function takes arrays of astrometric data (t_ast_yr, psi, plx_factor, ast_obs, ast_err), and a 12-set of astrometric parameters, predicts the epoch astrometry, and calculates the array of uncertainty-scaled residuals. This uses the compiled c function from kepler_solve_astrometry.so. 
    theta_array = (ra_off, dec_off, parallax, pmra, pmdec, period, ecc, phi_p, A, B, F, G); should be a numpy array.
    c_funcs comes from read_in_C_functions()
    '''
    t_ast_yr_double = t_ast_yr.astype(np.double)
    psi_double = psi.astype(np.double)
    plx_factor_double = plx_factor.astype(np.double)
    ast_obs_double = ast_obs.astype(np.double)
    ast_err_double = ast_err.astype(np.double)
    theta_array_double = theta_array.astype(np.double)
    
    resid_array = np.zeros(len(t_ast_yr), dtype = np.double)
    
    c_funcs.get_residual_array_12par_solution(ctypes.c_int(len(t_ast_yr)), ctypes.c_void_p(t_ast_yr_double.ctypes.data), 
        ctypes.c_void_p(psi_double.ctypes.data), ctypes.c_void_p(plx_factor_double.ctypes.data), ctypes.c_void_p(ast_obs_double.ctypes.data), ctypes.c_void_p(ast_err_double.ctypes.data),  ctypes.c_void_p(theta_array_double.ctypes.data), ctypes.c_void_p(resid_array.ctypes.data))
        
    if reject_outlier:
        resid_array[np.argmax(np.abs(resid_array))] = 0
    return np.array(resid_array)


def fit_orbital_solution_nonlinear(t_ast_yr, psi, plx_factor, ast_obs, ast_err, c_funcs, 
    L = np.array([10, 0, 0]), U = np.array([1e4, 2*np.pi, 0.99]), reject_outlier=False):
    '''
    this function takes arrays of astrometric data (t_ast_yr, psi, plx_factor, ast_obs, ast_err) and solves for the best-fit tuple of nonlinear parameters (P, phi_p, e) via adaptive simulated annealing.  This uses the compiled c function from kepler_solve_astrometry.so. 
    c_funcs comes from read_in_C_functions()
    L and U are arrays giving the lower and upper limits on each of the 3 nonlinear parameters. 
    returns (P, phi_p, e)
    '''

    t_ast_yr_double = t_ast_yr.astype(np.double)
    psi_double = psi.astype(np.double)
    plx_factor_double = plx_factor.astype(np.double)
    ast_obs_double = ast_obs.astype(np.double)
    ast_err_double = ast_err.astype(np.double)
    L_double = L.astype(np.double)
    U_double = U.astype(np.double)
    
    results_array = np.empty(3, dtype = np.double)
    
    if reject_outlier:
        c_funcs.run_astfit_reject_outlier(ctypes.c_void_p(t_ast_yr_double.ctypes.data), ctypes.c_void_p(psi_double.ctypes.data), ctypes.c_void_p(plx_factor_double.ctypes.data), ctypes.c_void_p(ast_obs_double.ctypes.data), ctypes.c_void_p(ast_err_double.ctypes.data), ctypes.c_void_p(L_double.ctypes.data), ctypes.c_void_p(U_double.ctypes.data),  ctypes.c_void_p(results_array.ctypes.data), ctypes.c_int(len(t_ast_yr)))    
    else:
        c_funcs.run_astfit(ctypes.c_void_p(t_ast_yr_double.ctypes.data), ctypes.c_void_p(psi_double.ctypes.data), ctypes.c_void_p(plx_factor_double.ctypes.data), ctypes.c_void_p(ast_obs_double.ctypes.data), ctypes.c_void_p(ast_err_double.ctypes.data), ctypes.c_void_p(L_double.ctypes.data), ctypes.c_void_p(U_double.ctypes.data),  ctypes.c_void_p(results_array.ctypes.data), ctypes.c_int(len(t_ast_yr)))    
    return results_array
    

def solve_kepler_eqn_on_array(M, ecc, c_funcs, xtol = 1e-10):
    '''
    numerically solve Kepler's equation for an array of mean anomaly values
    M: array, mean anomaly in radians
    ecc: eccentricity, float
    c_funcs: from read_in_C_functions()
    xtol: tolerance beyond which to stop
    '''
    M_double = M.astype(np.double)
    results_array = np.zeros(len(M), dtype = np.double)    
    c_funcs.solve_Kepler_equation_array(ctypes.c_int(len(M)), ctypes.c_void_p(M_double.ctypes.data), ctypes.c_double(ecc), ctypes.c_double(xtol), ctypes.c_void_p(results_array.ctypes.data))

    return results_array
    
def predict_reduced_chi2_unbinned_data(chi2_red_binned, n_param, N_points, Nbin=8):
    '''
    this function corrects for the fact that reduced chi2 for a poor fit increases when the data is binned. 
    chi2_red_binned: the reduced chi2, i.e. chi^2/(N_data - n_param), calculated from the binned data
    n_param: the number of free parameters in the model
    N_points: the number of points after binning. The number before binning is N_points*Nbin
    Nbin: how many observations are combined to make one data point. For our purposes, the number of CCDs 
    '''
    return (N_points*Nbin - N_points + chi2_red_binned*(N_points - n_param) )/(N_points*Nbin - n_param)
    
def predict_F2_unbinned_data(chi2_red_binned, n_param, N_points, Nbin=8):
    '''
    
    '''
    chi2_red_unbinned = predict_reduced_chi2_unbinned_data(chi2_red_binned = chi2_red_binned, n_param = n_param, N_points = N_points, Nbin=Nbin)
    nu_unbinned = N_points*Nbin - n_param # unbinned
    return np.sqrt(9*nu_unbinned/2)*(chi2_red_unbinned**(1/3) + 2/(9*nu_unbinned) -1  )

def check_ruwe(t_ast_yr, psi, plx_factor, ast_obs, ast_err, binned = True):
    '''
    This function takes a set of astrometric data (t_ast_yr, psi, plx_factor, ast_obs, ast_err) and fits a 5-parameter solution. It inflates the uncertainties according to the goodness of fit and returns the 5-parameter UWE, best-fit parameters, and uncertainties. 
    When calculating ruwe and parallax uncertainty inflation factors, we need to account for the fact that we binned
        (averaging 8 ccds per FOV transit), because binning does not conserve reduced chi^2. 
    '''
    Cinv = np.diag(1/ast_err**2)    
    M = np.vstack([np.sin(psi), t_ast_yr*np.sin(psi), np.cos(psi), t_ast_yr*np.cos(psi), plx_factor]).T 
    mu = np.linalg.solve(M.T @ Cinv @ M, M.T @ Cinv @ ast_obs)  
    Lambda_pred = np.dot(M, mu)
    resids = ast_obs - Lambda_pred
    Nobs, nu, nu_unbinned = len(ast_obs), len(ast_obs) - 5, len(ast_obs)*8 - 5  
    chi2_red_binned = np.sum(resids**2/ast_err**2)/nu
    chi2_red_unbinned = predict_reduced_chi2_unbinned_data(chi2_red_binned = chi2_red_binned, n_param = 5, N_points = Nobs, Nbin=8)
    
    if binned:
        ruwe = np.sqrt(chi2_red_unbinned)
        cc = np.sqrt(chi2_red_unbinned/((1-2/(9*nu_unbinned))**3 ))
    else:
        ruwe = np.sqrt(chi2_red_binned)
        cc = np.sqrt(chi2_red_binned/((1-2/(9*nu))**3 ))
        
    cov_matrix = np.linalg.inv(M.T @ Cinv @ M)
    sigma_mu = cc*np.sqrt(np.diag(cov_matrix))
    
    return ruwe, mu, sigma_mu

def check_7par(t_ast_yr, psi, plx_factor, ast_obs, ast_err, binned = True):
    '''
    This function takes a set of astrometric data (t_ast_yr, psi, plx_factor, ast_obs, ast_err) and fits a 7-parameter acceleration solution. It inflates the uncertainties according to the goodness of fit and returns the best-fit parameters and uncertainties and F2 and significance associated with the solution. 
    '''
    Cinv = np.diag(1/ast_err**2)    
    M = np.vstack([np.sin(psi), t_ast_yr*np.sin(psi), 1/2*t_ast_yr**2*np.sin(psi), np.cos(psi), t_ast_yr*np.cos(psi), 1/2*t_ast_yr**2*np.cos(psi), plx_factor]).T 
    mu = np.linalg.solve(M.T @ Cinv @ M, M.T @ Cinv @ ast_obs)  # ra, pmra, pmra_dot, dec, pmdec, pmdec_dot, plx
    
    Lambda_pred = np.dot(M, mu)
    resids = ast_obs - Lambda_pred
    Nobs, nu, nu_unbinned = len(ast_obs), len(ast_obs) - 7, len(ast_obs)*8 - 7
    
    chi2_red_binned = np.sum(resids**2/ast_err**2)/nu
    chi2_red_unbinned = predict_reduced_chi2_unbinned_data(chi2_red_binned = chi2_red_binned, n_param = 7, N_points = Nobs, Nbin=8)
    if binned:
        F2 = predict_F2_unbinned_data(chi2_red_binned = chi2_red_binned, n_param = 7, N_points = Nobs, Nbin=8)
        cc = np.sqrt(chi2_red_unbinned/((1-2/(9*nu_unbinned))**3 ))
    else:
        F2 = np.sqrt(9*nu/2)*(chi2_red_binned**(1/3) + 2/(9*nu) -1  )
        cc = np.sqrt(chi2_red_binned/((1-2/(9*nu))**3 ))
    
    cov_matrix = np.linalg.inv(M.T @ Cinv @ M)
    sigma_mu = cc*np.sqrt(np.diag(cov_matrix))
    cov25 = cov_matrix[2][5]*cc**2
    
    p1, p2, sig1, sig2 = mu[2], mu[5], sigma_mu[2], sigma_mu[5]
    rho12 = cov25/(sig1*sig2)
    s = 1/(sig1*sig2)*np.sqrt((p1**2*sig2**2 + p2**2*sig1**2 - 2*p1*p2*rho12*sig1*sig2)/(1-rho12**2))
    return F2, s, mu, sigma_mu
 
def check_9par(t_ast_yr, psi, plx_factor, ast_obs, ast_err, binned=True):
    '''
    This function takes a set of astrometric data (t_ast_yr, psi, plx_factor, ast_obs, ast_err) and fits a 9-parameter acceleration solution. It inflates the uncertainties according to the goodness of fit and returns the best-fit parameters and uncertainties and F2 and significance associated with the solution. 
    '''
    Cinv = np.diag(1/ast_err**2)    
    M = np.vstack([np.sin(psi), t_ast_yr*np.sin(psi), 1/2*t_ast_yr**2*np.sin(psi),  1/6*t_ast_yr**3*np.sin(psi), np.cos(psi), t_ast_yr*np.cos(psi), 1/2*t_ast_yr**2*np.cos(psi), 1/6*t_ast_yr**3*np.cos(psi), plx_factor]).T 
    mu = np.linalg.solve(M.T @ Cinv @ M, M.T @ Cinv @ ast_obs)  # ra, pmra, pmra_dot, pmra_ddot, dec, pmdec, pmdec_dot, pmdec_ddot, plx
    
    Lambda_pred = np.dot(M, mu)
    resids = ast_obs - Lambda_pred
    
    Nobs, nu, nu_unbinned = len(ast_obs), len(ast_obs) - 9, len(ast_obs)*8 - 9
    chi2_red_binned = np.sum(resids**2/ast_err**2)/nu
    chi2_red_unbinned = predict_reduced_chi2_unbinned_data(chi2_red_binned = chi2_red_binned, n_param = 9, N_points = Nobs, Nbin=8)
    
    if binned:
        F2 = predict_F2_unbinned_data(chi2_red_binned = chi2_red_binned, n_param = 9, N_points = Nobs, Nbin=8)
        cc = np.sqrt(chi2_red_unbinned/((1-2/(9*nu_unbinned))**3 ))
    else:
        F2 = np.sqrt(9*nu/2)*(chi2_red_binned**(1/3) + 2/(9*nu) -1  )
        cc = np.sqrt(chi2_red_binned/((1-2/(9*nu))**3 ))
 
    cov_matrix = np.linalg.inv(M.T @ Cinv @ M)
    sigma_mu = cc*np.sqrt(np.diag(cov_matrix))
    cov37 = cov_matrix[3][7]*cc**2
    
    p1, p2, sig1, sig2 = mu[3], mu[7], sigma_mu[3], sigma_mu[7]
    rho12 = cov37/(sig1*sig2)
    s = 1/(sig1*sig2)*np.sqrt((p1**2*sig2**2 + p2**2*sig1**2 - 2*p1*p2*rho12*sig1*sig2)/(1-rho12**2))
    return F2, s, mu, sigma_mu


def al_bias_binary(delta_eta, q, f, u = 90):
    '''
    This function predicts the epoch astrometry for a binary assuming that the 1D centroid is at the peak of the combined AL flux profile, following the model from Lindegren+2022
    q = m2/m1 is the flux ratio 
    f = F2/F1 is the light ratio
    u is the effective angular resolution in mas. 
    delta_eta = rho*cos(psi-theta) = (-y*cos(psi) - x*sin(psi)) 
        where rho is the angular separation between the two stars, psi is the scan angle, and theta is position angle.
    '''

    def solve_for_x(ff, xi, tol = 1e-6, niter_max=100):
        '''
        ff is flux ratio, xi is angular separation in units of angular resolution. 
        tol is a tolerance to monitor convergence. 
        niter_max is the maximum number of iterations
        '''
        x = 0.0
        for i in range(niter_max):
            thisx = ff * xi / (ff + np.exp(0.5 * xi**2 - xi * x))
            if abs(thisx - x) < tol:
                break
            x = thisx
        return x

    # the first two cases reduce to the same thing, but it's better to separate them for numerical stability.
    if np.abs(delta_eta/u) <= 0.1:
        deta = (f/(1+f) - q/(1+q))*delta_eta
    elif np.abs(delta_eta/u) > 0.1 and np.abs(delta_eta/u) <= 3-f:
        B = solve_for_x(ff=f, xi=delta_eta/u)
        deta = u*B - q/(1+q)*delta_eta
    elif np.abs(delta_eta/u) > 3-f:
        deta = -q/(1+q)*delta_eta
    return deta
    
def rescale_times_astrometry(jd, data_release):
    '''
    calculating the time in years relative to the reference epoch, which is different for each data release. 
    jd is the time in days
    data_release is dr3, dr4, or dr5
    '''
    if data_release == 'dr3':
        t_ast_day = jd - 2457389.0
    elif data_release == 'dr4':
        t_ast_day = jd - 2457936.875
    elif data_release == 'dr5':
        t_ast_day = jd - 2458818.5
    else: 
        raise ValueError('invalid data_release!')
    t_ast_yr = t_ast_day/365.25
    return t_ast_yr
    
def predict_astrometry_luminous_binary(ra, dec, parallax, pmra, pmdec, m1, m2, period, Tp, ecc, omega, inc, w, phot_g_mean_mag, f, data_release, c_funcs, do_blending_noise = False):
    '''
    this function predicts the epoch-level astrometry for a binary as it would be observed by Gaia. 
    ra and dec (degrees): the coordinates of the source at the reference time (which is different for dr3/dr4/dr5)
    parallax (mas): the true parallax (i.e., 1/d)
    pmra, pmdec: true proper motions in mas/yr
    m1: mass of the star more luminous in the G-band, in Msun
    m2: mass of the other star, in Msun
    period: orbital period in days
    Tp: periastron time in days
    ecc: eccentricity
    omega: "big Omega" in radians
    inc: inclination in radians, defined so that 0 or pi is face-on, and pi/2 is edge-on. 
    w: "little omega" in radians
    phot_g_mean_mag: G-band magnitude 
    f: flux ratio, F2/F1, in the G-band. 
    data_release: 'dr3', 'dr4', or 'dr5'
    c_funcs: from read_in_C_functions()
    '''
    
    t = get_gost_one_position(ra, dec, data_release=data_release)
    
    # reject a random 10%
    t = t[np.random.uniform(0, 1, len(t)) > 0.1]
    psi, plx_factor, jds = fetch_table_element(['scanAngle[rad]', 'parallaxFactorAlongScan', 'ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]'], t)
    t_ast_yr = rescale_times_astrometry(jd = jds, data_release = data_release)
    
    N_ccd_avg = 8
    epoch_err_per_transit = al_uncertainty_per_ccd_interp(G = phot_g_mean_mag)/np.sqrt(N_ccd_avg)
    
    if phot_g_mean_mag < 13:
        extra_noise = np.random.uniform(0, 0.04)
    else: 
        extra_noise = 0

    EE = solve_kepler_eqn_on_array(M = 2*np.pi/period * (t_ast_yr*365.25 - Tp), ecc = ecc, c_funcs = c_funcs)
    a_mas = get_a_mas(period, m1, m2, parallax)
    A_pred = a_mas*( np.cos(w)*np.cos(omega) - np.sin(w)*np.sin(omega)*np.cos(inc) )
    B_pred = a_mas*( np.cos(w)*np.sin(omega) + np.sin(w)*np.cos(omega)*np.cos(inc) )
    F_pred = -a_mas*( np.sin(w)*np.cos(omega) + np.cos(w)*np.sin(omega)*np.cos(inc) )
    G_pred = -a_mas*( np.sin(w)*np.sin(omega) - np.cos(w)*np.cos(omega)*np.cos(inc) )
    cpsi, spsi = np.cos(psi), np.sin(psi)
    
    X = np.cos(EE) - ecc
    Y = np.sqrt(1-ecc**2)*np.sin(EE)
    
    x, y = B_pred*X + G_pred*Y, A_pred*X + F_pred*Y   
    delta_eta = (-y*cpsi - x*spsi) 
    bias = np.array([al_bias_binary(delta_eta = delta_eta[i], q=m2/m1, f=f) for i in range(len(psi))])
    Lambda_com = pmra*t_ast_yr*spsi + pmdec*t_ast_yr*cpsi + parallax*plx_factor # barycenter motion
    Lambda_pred = Lambda_com + bias # binary motion

    Lambda_pred += epoch_err_per_transit*np.random.randn(len(psi)) # modeled noise
    Lambda_pred += extra_noise*np.random.randn(len(psi)) # unmodeled noise
    
    # extra noise for partially resolved sources
    if do_blending_noise:
        blending_noise = 0.5*np.random.randn(len(psi))
        blending_noise[np.abs(delta_eta) < 45] = 0 # 0 if \Delta \eta < resolution/2
        Lambda_pred += blending_noise
    
    return t_ast_yr, psi, plx_factor, Lambda_pred, epoch_err_per_transit*np.ones(len(Lambda_pred))


def predict_astrometry_binary_in_terms_of_a0(ra, dec, parallax, pmra, pmdec, period, Tp, ecc, omega, inc, w, a0_mas, phot_g_mean_mag, data_release, c_funcs):
    '''
    this function predicts the epoch-level astrometry for a binary as it would be observed by Gaia, in terms of a0 rather than m1 and m2 and f. It is only valid in the limit where the separation of the two stars is less than about 45 mas, so that the photocenter approximation works well. 
    
    ra and dec (degrees): the coordinates of the source at the reference time (which is different for dr3/dr4/dr5)
    parallax (mas): the true parallax (i.e., 1/d)
    pmra, pmdec: true proper motions in mas/yr
    a0_mas: the photocenter semimajor axis 
    period: orbital period in days
    Tp: periastron time in days
    ecc: eccentricity
    omega: "big Omega" in radians
    inc: inclination in radians, defined so that 0 or pi is face-on, and pi/2 is edge-on. 
    w: "little omega" in radians
    phot_g_mean_mag: G-band magnitude 
    data_release: 'dr3', 'dr4', or 'dr5'
    c_funcs: from read_in_C_functions()
    '''
    
    t = get_gost_one_position(ra, dec, data_release=data_release)
    
    # reject a random 10%
    t = t[np.random.uniform(0, 1, len(t)) > 0.1]
    psi, plx_factor, jds = fetch_table_element(['scanAngle[rad]', 'parallaxFactorAlongScan', 'ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]'], t)
    t_ast_yr = rescale_times_astrometry(jd = jds, data_release = data_release)
    
    N_ccd_avg = 8
    epoch_err_per_transit = al_uncertainty_per_ccd_interp(G = phot_g_mean_mag)/np.sqrt(N_ccd_avg)
    
    if phot_g_mean_mag < 13:
        extra_noise = np.random.uniform(0, 0.04)
    else: 
        extra_noise = 0

    EE = solve_kepler_eqn_on_array(M = 2*np.pi/period * (t_ast_yr*365.25 - Tp), ecc = ecc, c_funcs = c_funcs)
 
    A_pred = a0_mas*( np.cos(w)*np.cos(omega) - np.sin(w)*np.sin(omega)*np.cos(inc) )
    B_pred = a0_mas*( np.cos(w)*np.sin(omega) + np.sin(w)*np.cos(omega)*np.cos(inc) )
    F_pred = -a0_mas*( np.sin(w)*np.cos(omega) + np.cos(w)*np.sin(omega)*np.cos(inc) )
    G_pred = -a0_mas*( np.sin(w)*np.sin(omega) - np.cos(w)*np.cos(omega)*np.cos(inc) )
    
    X = np.cos(EE) - ecc
    Y = np.sqrt(1-ecc**2)*np.sin(EE)
    
    x, y = B_pred*X + G_pred*Y, A_pred*X + F_pred*Y   
    delta_eta = (y*np.cos(psi) + x*np.sin(psi)) 
    
    Lambda_com = pmra*t_ast_yr*np.sin(psi) + pmdec*t_ast_yr*np.cos(psi) + parallax*plx_factor # barycenter motion
    Lambda_pred = Lambda_com + delta_eta # binary motion

    Lambda_pred += epoch_err_per_transit*np.random.randn(len(psi)) # modeled noise
    Lambda_pred += extra_noise*np.random.randn(len(psi)) # unmodeled noise
    
    return t_ast_yr, psi, plx_factor, Lambda_pred, epoch_err_per_transit*np.ones(len(Lambda_pred))


def get_a_mas(period, m1, m2, parallax):
    '''
    calculate the projected semi-major axis of a binary (not the photocenter)
    period: orbital period in days
    m1: mass of star 1 in Msun
    m2: mass of star 2 in Msun
    parallax: true parallax (i.e., 1/d) in mas. 
    '''
    G, Msun, AU = 6.6743e-11, 1.98840987069805e+30, 1.4959787e+11
    a_au = (((period*86400)**2 * G * (m1*Msun + m2*Msun)/(4*np.pi**2))**(1/3.))/AU
    return a_au*parallax
    
def get_a0_mas(period, m1, m2, parallax, f):
    '''
    calculate the projected photocenter semi-major axis of a binary 
    period: orbital period in days
    m1: mass of star 1 in Msun
    m2: mass of star 2 in Msun
    parallax: true parallax (i.e., 1/d) in mas. 
    f = F2/F1 is flux ratio in G band. 
    
    '''
    G, Msun, AU = 6.6743e-11, 1.98840987069805e+30, 1.4959787e+11
    a_au = (((period*86400)**2 * G * (m1*Msun + m2*Msun)/(4*np.pi**2))**(1/3.))/AU
    a_mas, q = a_au*parallax,  m2/m1
    a0_mas = a_mas*(q/(1+q) - f/(1+f))
    return a0_mas
    
def fetch_table_element(colname, table):
    '''
    retrieve a column or set of columns from an astropy table as an array or set of arrays. Lose the masks or units. 
    '''
    if type(colname) == str:
        if type(table[colname].data.data) == memoryview:
            dat_ = table[colname].data
        else:
            dat_ = table[colname].data.data
    elif type(colname) == list:
        dat_ = []
        for col in colname:
            dat_.append(fetch_table_element(col, table))
    return dat_


def get_gost_one_position(ra, dec, data_release):
    '''
    this function finds the scan times and angles for a given sky position and data release by searching for the nearest position in a pre-downloaded set of 49152 sky positions (healpix level 64). For this to work, you need to have downloaded healpix_scans.zip and unzipped it into gaiamock/healpix_scans. 
    ra and dec: coordinates of interest in degrees (floats)
    data_release: 'dr3', 'dr4', or 'dr5'
    '''
    num =  hp.ang2pix(64, np.radians(90.0 - np.array(dec)), np.radians(np.array(ra)), nest=False)
    tab = Table.read(os.path.join(os.path.dirname(__file__), 'healpix_scans/healpix_64_%d.fits' % num))
    jd = fetch_table_element('ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]', tab)
    
    if data_release == 'dr4':
        m = (jd > 2456891.5) & (jd < 2458868.5)
    elif data_release == 'dr3': 
        m = (jd > 2456891.5) & (jd < 2457902)
    elif data_release == 'dr5':
        m = np.ones(len(tab), dtype=bool)
    else:
        raise ValueError('invalid data_release!')
    return tab[m]

def plot_residuals(t_ast_yr, psi, plx_factor, ast_obs, ast_err, theta_array, c_funcs):
    '''
    this function takes a set of epoch astrometry (as described by t_ast_yr, psi, plx_factor, ast_obs, and ast_err), and a set of nonlinear parameters theta_array = (Porb, phi_p, ecc), and predicts the epoch astrometry. It also calculates the best-fit 5 parameter solution for the same astrometry. Finally, it plots the epoch astrometry residuals as a function of time for both solutions. 
    '''
 
    # and also do one without an orbit
    Cinv = np.diag(1/ast_err**2)    
    M = np.vstack([np.sin(psi), t_ast_yr*np.sin(psi), np.cos(psi), t_ast_yr*np.cos(psi), plx_factor]).T 
    mu = np.linalg.solve(M.T @ Cinv @ M, M.T @ Cinv @ ast_obs) #  ra, pmra, dec, pmdec, parallax
    Lambda_pred = np.dot(M, mu)
    
    nrow, width, height_scale = 2, 6, 1
    xlim = [np.min(t_ast_yr)-0.2, np.max(t_ast_yr)+0.2]
    f, ax = plt.subplots(nrow, 1, figsize = (width, 1+3*nrow*height_scale))
    plt.subplots_adjust(hspace = 0)
    for i in range(nrow):
        ax[i].set_xlim(xlim)
        ax[i].tick_params(labelsize = 18)
        if i != nrow - 1:
            ax[i].set_xticklabels([])
    ax[0].errorbar(t_ast_yr, ast_obs - Lambda_pred, yerr=ast_err, fmt='k.')
    print('single star chi2: %.2f'  % (np.sum( (ast_obs - Lambda_pred)**2 / ast_err**2 )) )
    
    period, Tp, ecc = theta_array[0], theta_array[1]*theta_array[0]/(2*np.pi), theta_array[2]
    EE = solve_kepler_eqn_on_array(M = 2*np.pi/period * (t_ast_yr*365.25 - Tp), ecc = ecc, c_funcs = c_funcs)
    X = np.cos(EE) - ecc
    Y = np.sqrt(1-ecc**2)*np.sin(EE) 
    M = np.vstack([np.sin(psi), t_ast_yr*np.sin(psi), np.cos(psi), t_ast_yr*np.cos(psi), plx_factor, X*np.sin(psi), Y*np.sin(psi), X*np.cos(psi), Y*np.cos(psi)]).T  
    mu = np.linalg.solve(M.T @ Cinv @ M, M.T @ Cinv @ ast_obs) 
    Lambda_pred = np.dot(M, mu) 
        
    print('binary star chi2: %.2f'  % (np.sum( (ast_obs - Lambda_pred)**2 / ast_err**2 )) )
    ax[1].errorbar(t_ast_yr, ast_obs - Lambda_pred, yerr=ast_err, fmt='k.')
    ax[1].set_xlabel('time (years)', fontsize=20)
    ax[0].set_ylabel('residual (5 par)', fontsize=20)
    ax[1].set_ylabel('residual (12 par)', fontsize=20)

def plot_residuals_9par(t_ast_yr, psi, plx_factor, ast_obs, ast_err, theta_array, c_funcs):
    '''
    this function takes a set of epoch astrometry (as described by t_ast_yr, psi, plx_factor, ast_obs, and ast_err), and a set of linear parameters for a 9-parameter solution, theta_array = (ra, pmra, pmra_dot, pmra_ddot, dec, pmdec, pmdec_dot, pmdec_ddot, plx), and predicts the epoch astrometry. It also calculates the best-fit 5 parameter solution for the same astrometry. Finally, it plots the epoch astrometry residuals as a function of time for both solutions. 
    '''
    # 5 parameter solution
    Cinv = np.diag(1/ast_err**2)    
    M = np.vstack([np.sin(psi), t_ast_yr*np.sin(psi), np.cos(psi), t_ast_yr*np.cos(psi), plx_factor]).T 
    mu = np.linalg.solve(M.T @ Cinv @ M, M.T @ Cinv @ ast_obs) #  ra, pmra, dec, pmdec, parallax
    Lambda_pred = np.dot(M, mu)
    
    nrow, width, height_scale = 2, 6, 1
    xlim = [np.min(t_ast_yr)-0.2, np.max(t_ast_yr)+0.2]
    f, ax = plt.subplots(nrow, 1, figsize = (width, 1+3*nrow*height_scale))
    plt.subplots_adjust(hspace = 0)
    for i in range(nrow):
        ax[i].set_xlim(xlim)
        ax[i].tick_params(labelsize = 18)
        if i != nrow - 1:
            ax[i].set_xticklabels([])
    ax[0].errorbar(t_ast_yr, ast_obs - Lambda_pred, yerr=ast_err, fmt='k.')
    print('single star chi2: %.2f'  % (np.sum( (ast_obs - Lambda_pred)**2 / ast_err**2 )) )
    
    # 9 parameter solution
    M = np.vstack([np.sin(psi), t_ast_yr*np.sin(psi), 1/2*t_ast_yr**2*np.sin(psi),  1/6*t_ast_yr**3*np.sin(psi), np.cos(psi), t_ast_yr*np.cos(psi), 1/2*t_ast_yr**2*np.cos(psi), 1/6*t_ast_yr**3*np.cos(psi), plx_factor]).T 
    Lambda_pred = np.dot(M, theta_array)
    resids = ast_obs - Lambda_pred
    chi2 = np.sum(resids**2/ast_err**2)
 
    print('9 parameter chi2: %.2f' % chi2 )
    ax[1].errorbar(t_ast_yr, ast_obs - Lambda_pred, yerr=ast_err, fmt='k.')
    ax[1].set_xlabel('time (years)', fontsize=20)
    ax[0].set_ylabel('residual (5 par)', fontsize=20)
    ax[1].set_ylabel('residual (9 par)', fontsize=20)


def plot_residuals_7par(t_ast_yr, psi, plx_factor, ast_obs, ast_err, theta_array, c_funcs):
    '''
    this function takes a set of epoch astrometry (as described by t_ast_yr, psi, plx_factor, ast_obs, and ast_err), and a set of linear parameters for a 7-parameter solution, theta_array = (ra, pmra, pmra_dot, pmra_ddot, dec, pmdec, pmdec_dot, pmdec_ddot, plx), and predicts the epoch astrometry. It also calculates the best-fit 5 parameter solution for the same astrometry. Finally, it plots the epoch astrometry residuals as a function of time for both solutions. 
    '''
    # 5 parameter solution
    Cinv = np.diag(1/ast_err**2)    
    M = np.vstack([np.sin(psi), t_ast_yr*np.sin(psi), np.cos(psi), t_ast_yr*np.cos(psi), plx_factor]).T 
    mu = np.linalg.solve(M.T @ Cinv @ M, M.T @ Cinv @ ast_obs) #  ra, pmra, dec, pmdec, parallax
    Lambda_pred = np.dot(M, mu)
    
    nrow, width, height_scale = 2, 6, 1
    xlim = [np.min(t_ast_yr)-0.2, np.max(t_ast_yr)+0.2]
    f, ax = plt.subplots(nrow, 1, figsize = (width, 1+3*nrow*height_scale))
    plt.subplots_adjust(hspace = 0)
    for i in range(nrow):
        ax[i].set_xlim(xlim)
        ax[i].tick_params(labelsize = 18)
        if i != nrow - 1:
            ax[i].set_xticklabels([])
    ax[0].errorbar(t_ast_yr, ast_obs - Lambda_pred, yerr=ast_err, fmt='k.')
    print('single star chi2: %.2f'  % (np.sum( (ast_obs - Lambda_pred)**2 / ast_err**2 )) )
    
    # 7 parameter solution
    M = np.vstack([np.sin(psi), t_ast_yr*np.sin(psi), 1/2*t_ast_yr**2*np.sin(psi), np.cos(psi), t_ast_yr*np.cos(psi), 1/2*t_ast_yr**2*np.cos(psi), plx_factor]).T 
    Lambda_pred = np.dot(M, theta_array)
    resids = ast_obs - Lambda_pred
    chi2 = np.sum(resids**2/ast_err**2)
 
    print('7-parameter chi2: %.2f' % chi2 )
    ax[1].errorbar(t_ast_yr, ast_obs - Lambda_pred, yerr=ast_err, fmt='k.')
    ax[1].set_xlabel('time (years)', fontsize=20)
    ax[0].set_ylabel('residual (5 par)', fontsize=20)
    ax[1].set_ylabel('residual (7 par)', fontsize=20)

    
def get_uncertainties_at_best_fit_binary_solution(t_ast_yr, psi, plx_factor, ast_obs, ast_err, p0, c_funcs, binned=True, reject_outlier=False):
    '''
    This function calculates the Hessian matrix, approximated from the Jacobian, at a coordinate in 12-dimensional parameter space p0. 
    p0 = (ra_offset, dec_offset, parallax, pmra, pmdec, period, ecc, phi_p, A, B, F, G)
    t_ast_yr, psi, plx_factor, ast_obs, and ast_err are the epoch astrometry 
    c_funcs is from read_in_C_functions()
    
    returns:
        uncertainties: 12-parameter uncertainties 
        a0, sigma_a0: photocenter ellipse and uncertainty, calculated following Halbwachs+2023
        inc_deg: inclination in degrees, calculated following Halbwachs+2023
    '''
    
    def resid_func(theta):
        '''helper function for Jacobian'''
        return get_astrometric_residuals_12par(t_ast_yr, psi, plx_factor, ast_obs, ast_err, theta_array = np.array(theta), c_funcs = c_funcs, reject_outlier=reject_outlier)
            
    def jacobian(params, epsilon=1e-8):
        '''
        numerical Jacobian. epsilon is step size. 
        '''
        J = np.zeros((len(t_ast_yr), len(params)))
        for i in range(len(params)):
            params_up = params.copy()
            params_down = params.copy()
            params_up[i] += epsilon
            params_down[i] -= epsilon
            J[:, i] = (resid_func(params_up) - resid_func(params_down)) / (2 * epsilon)
        return J
        
    J = jacobian(p0)    
    cov_x = np.linalg.inv(np.dot(J.T, J))  

    if not np.sum(~np.isfinite(cov_x)):
        
        if reject_outlier:
            nu, Nobs, nu_unbinned = len(ast_obs)-12-1, len(ast_obs)-1, (len(ast_obs)-1)*8 - 12
        else:
            nu, Nobs, nu_unbinned = len(ast_obs) - 12, len(ast_obs), len(ast_obs)*8 - 12  
        chi2_red_binned = np.sum(resid_func(p0)**2)/nu
        chi2_red_unbinned = predict_reduced_chi2_unbinned_data(chi2_red_binned = chi2_red_binned, n_param = 12, N_points = Nobs, Nbin=8)
        if binned:
            cc = np.sqrt(chi2_red_unbinned/((1 - 2/(9*nu_unbinned))**3 ))
        else:
            cc = np.sqrt(chi2_red_binned/((1 - 2/(9*nu))**3 ))
        uncertainties = cc*np.sqrt(np.diag(cov_x))

        ra_off, dec_off, parallax, pmra, pmdec, period, ecc, phi_p, A, B, F, G = p0
        sig_ra_off, sig_dec_off, sig_parallax, sig_pmra, sig_pmdec, sig_period, sig_ecc, sig_phi_p, sig_A, sig_B, sig_F, sig_G = uncertainties
        
        # calculate a0 and errors according to halbwachs
        u = 0.5 * (A**2 + B**2 + F**2 + G**2)
        v = A * G - B * F
        a0 = np.sqrt(u + np.sqrt(u**2 - v**2))
        inc_deg = 180/np.pi*np.arccos(v/(a0*a0))
        
        t_A = A + (A * u - G * v) / np.sqrt(u**2 - v**2)
        t_B = B + (B * u + F * v) / np.sqrt(u**2 - v**2)
        t_F = F + (F * u + B * v) / np.sqrt(u**2 - v**2)
        t_G = G + (G * u - A * v) / np.sqrt(u**2 - v**2)
        
        covAB, covAF, covAG = cov_x[8, 9]*cc**2, cov_x[8,10]*cc**2, cov_x[8, 11]*cc**2
        covBF, covBG, covFG = cov_x[9,10]*cc**2, cov_x[9,11]*cc**2, cov_x[10,11]*cc**2
        
        under_radical = t_A**2 * sig_A**2 + t_B**2 * sig_B**2 + t_F**2 * sig_F**2 + t_G**2 * sig_G**2 + 2 * t_A * t_B * covAB + 2 * t_A * t_F * covAF + 2 * t_A * t_G * covAG + 2 * t_B * t_F * covBF + 2 * t_B * t_G * covBG + 2 * t_F * t_G * covFG 
        
        if under_radical > 0:
            sigma_a0 = 1 / (2 * a0) * np.sqrt(under_radical)
        else:
            sigma_a0 = a0*1000


    else:
        uncertainties = np.ones(len(p0))*1000
        a0, sigma_a0, inc_deg = 0.01, 100, 0
    
    return uncertainties, a0, sigma_a0, inc_deg


def fit_full_astrometric_cascade(t_ast_yr, psi, plx_factor, ast_obs, ast_err, c_funcs, verbose=False, show_residuals=False, binned = True, ruwe_min = 1.4, skip_acceleration=False, reject_outlier=False, P_min = 10):
    '''
    this function takes 1D astrometry and fits it with a cascade of astrometric models.  
    t_ast_yr, psi, plx_factor, ast_obs, ast_err: arrays of astrometric measurements and related metadata
    c_funcs: from read_in_C_functions()
    verbose: whether to print results of fitting. 
    if show_residuals, plot the residuals of the best-fit 5-parameter solution and the best-fit orbital solution. This will only happen if an orbital solution is actually calculated (i.e., we get to that stage in the cascade.)
    reject_outlier: if true, ignore the point with the worst chi2 when fitting the orbital solution
    '''    
    Nret = 23 # number of arguments to return 
    N_visibility_periods = int(np.sum( np.diff(t_ast_yr*365.25) > 4) + 1)
    if (N_visibility_periods < 12) or (len(ast_obs) < 13): 
        if verbose:
            print('not enough visibility periods!')
        return Nret*[0]
    
    # check 5-parameter solution 
    ruwe, mu, sigma_mu = check_ruwe(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, binned=binned)
    if ruwe < ruwe_min:
        res =  Nret*[-1] # set most return arguments to -1, but save a few parameters for convenience 
        res[1] = ruwe 
        res[2] = mu[-1]
        res[3] = sigma_mu[-1]
        
        if verbose:
            print('UWE < 1.4: returning only 5-parameter solution.')
        return res
        
    if c_funcs is None:
        c_funcs = read_in_C_functions()
    
    # mu is  ra, pmra, pmra_dot, pmra_ddot, dec, pmdec, pmdec_dot, pmdec_ddot, plx
    F2_9par, s_9par, mu, sigma_mu = check_9par(t_ast_yr, psi, plx_factor, ast_obs, ast_err, binned=binned)
    plx_over_err9 = mu[-1]/sigma_mu[-1]
    if (F2_9par < 25) and (s_9par > 12) and (plx_over_err9 > 2.1*s_9par**1.05) and (not skip_acceleration):
        res =  Nret*[-9]  # set most return arguments to -9, but save a few parameters for convenience 
        res[1] = s_9par 
        res[2], res[3] = mu[-1], sigma_mu[-1] # parallax
        res[4], res[5] = mu[2], sigma_mu[2] # pmra_dot
        res[6], res[7] = mu[3], sigma_mu[3] # pmra_ddot
        res[8], res[9] = mu[6], sigma_mu[6] # pmdec_dot
        res[10], res[11] = mu[7], sigma_mu[7] # pmdec_ddot
        res[12] = ruwe 
        res[13] = F2_9par 

        if verbose:
            print('9 parameter solution accepted! Not trying anything else.')
            print('s9: %.1f, plx_over_err9: %.1f, F2_9: %.1f' % (s_9par, plx_over_err9, F2_9par))
        if show_residuals:
            plot_residuals_9par(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, theta_array = mu, c_funcs = c_funcs)
        return res
    
    # mu is ra, pmra, pmra_dot, dec, pmdec, pmdec_dot, plx
    F2_7par, s_7par, mu, sigma_mu = check_7par(t_ast_yr, psi, plx_factor, ast_obs, ast_err, binned=binned)
    plx_over_err7 = mu[-1]/sigma_mu[-1]
    if (F2_7par < 25) and (s_7par > 12) and (plx_over_err7 > 1.2*s_7par**1.05) and (not skip_acceleration):
        res =  Nret*[-7]
        res[1] = s_7par
        res[2], res[3] = mu[-1], sigma_mu[-1] # parallax
        res[4], res[5] = mu[2], sigma_mu[2] # pmra_dot
        res[6], res[7] = mu[5], sigma_mu[5] # pmdec_dot
        res[8] = ruwe
        res[9] = F2_7par
        
        if verbose:
            print('7 parameter solution accepted! Not trying anything else.')
        if show_residuals:
            plot_residuals_7par(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, theta_array = mu, c_funcs = c_funcs)
        return res

    res = fit_orbital_solution_nonlinear(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, c_funcs = c_funcs, L = np.array([P_min, 0, 0]), reject_outlier=reject_outlier)
        
    if verbose:
        print('found best-fit nonlinear parameters:', res)
    
    # get the linear parameters 
    period, phi_p, ecc = res
    chi2, mu_linear = get_astrometric_chi2(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, P = period, phi_p = phi_p, ecc = ecc, c_funcs=c_funcs, reject_outlier=reject_outlier)
    ra_off, pmra, dec_off, pmdec, plx, B, G, A, F = mu_linear
    p0 = [ra_off, dec_off, plx, pmra, pmdec, period, ecc, phi_p, A, B, F, G]
    
    # get some uncertainties 
    errors, a0_mas, sigma_a0_mas, inc_deg = get_uncertainties_at_best_fit_binary_solution(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, p0 = p0, c_funcs = c_funcs, binned=binned, reject_outlier=reject_outlier)
    sig_ra,sig_dec,sig_parallax,sig_pmra,sig_pmdec,sig_period,sig_ecc,sig_phi_p,sig_A,sig_B, sig_F,sig_G = errors
    
    if reject_outlier:
        Nobs, nu, nu_unbinned = len(ast_obs)-1, len(ast_obs)-12-1, (len(ast_obs)-1)*8 - 12
    else:
        Nobs, nu, nu_unbinned = len(ast_obs), len(ast_obs) - 12, len(ast_obs)*8 - 12
    chi2_red_binned = chi2/nu
    
    if binned:
        F2 = predict_F2_unbinned_data(chi2_red_binned = chi2_red_binned, n_param = 12, N_points = Nobs, Nbin=8)
    else:
        F2 = np.sqrt(9*nu/2)*(chi2_red_binned**(1/3) + 2/(9*nu) - 1)
    a0_over_err, parallax_over_error = a0_mas/sigma_a0_mas, plx/sig_parallax
    
    if show_residuals:
        plot_residuals(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, theta_array = res, c_funcs = c_funcs)
    
    if verbose: 
        if F2 < 25:
            print('goodness_of_fit (F2) is low enough to pass DR3 cuts! F2: %.1f' % F2)
        else:
            print('goodness_of_fit (F2) is too high to pass DR3 cuts! F2: %.1f' % F2)

        if (a0_over_err > 158/np.sqrt(period)) and (a0_over_err > 5):
            print('a0_over_err is high enough to pass DR3 cuts! a0_over_err: %.1f' % a0_over_err)
        else:
            print('a0_over_err is NOT high enough to pass DR3 cuts! a0_over_err: %.1f' % a0_over_err)

        if parallax_over_error > 20000/period:
            print('parallax over error is high enough to pass DR3 cuts! parallax_over_error: %.1f' % parallax_over_error)
        else:
            print('parallax over error is NOT high enough to pass DR3 cuts! parallax_over_error: %.1f' % parallax_over_error)
        if (sig_ecc < 0.079*np.log(period)-0.244):
            print('eccentricity error is low enough to pass DR3 cuts! ecc_error: %.2f' % sig_ecc)
        else:
            print('eccentricity error is too high to pass DR3 cuts! ecc_error: %.2f' % sig_ecc)
    
    # lots of stuff that can be useful to return
    return_array = [plx, sig_parallax, A, sig_A, B, sig_B, F, sig_F, G, sig_G, period, sig_period, phi_p, sig_phi_p, ecc, sig_ecc, inc_deg, a0_mas, sigma_a0_mas, N_visibility_periods, len(t_ast_yr), F2, ruwe]
    return return_array


def run_full_astrometric_cascade(ra, dec, parallax, pmra, pmdec, m1, m2, period, Tp, ecc, omega, inc_deg, w, phot_g_mean_mag, f, data_release, c_funcs, verbose=False, show_residuals=False, ruwe_min = 1.4, skip_acceleration=False):
    '''
    this function generates the mock 1D astrometry for a binary and then fits it with a cascade of astrometric models.  
    ra, dec: coordinates, in degrees
    parallax: true parallax in mas
    pmra, pmdec, true proper motions in mas/yr
    m1: mass of star 1 in Msun
    m2: mass of star 2 in Msun
    period: orbital period in days
    Tp: periastron time in days
    ecc: eccentricity
    omega: "big Omega" in radians
    inc_deg: inclination in degrees, defined so that 0 or 180 is face-on, and 90 is edge-on. 
    w: "little omega" in radians
    epoch_err_mas: this is the uncertainty in AL displacement per FOV transit (not per CCD)
    f: flux ratio, F2/F1, in the G-band. 
    data_release: 'dr3', 'dr4', or 'dr5'
    c_funcs: from read_in_C_functions()
    verbose: whether to print results of fitting. 
    if show_residuals, plot the residuals of the best-fit 5-parameter solution and the best-fit orbital solution. This will only happen if an orbital solution is actually calculated (i.e., we get to that stage in the cascade.)
    '''
    if c_funcs is None:
        c_funcs = read_in_C_functions()

    t_ast_yr, psi, plx_factor, ast_obs, ast_err = predict_astrometry_luminous_binary(ra = ra, dec = dec, parallax = parallax, pmra = pmra, pmdec = pmdec, m1 = m1, m2 = m2, period = period, Tp = Tp, ecc = ecc, omega = omega, inc = inc_deg*np.pi/180, w=w, phot_g_mean_mag = phot_g_mean_mag, f=f, data_release=data_release, c_funcs=c_funcs)
    
    Nret = 23 # number of arguments to return 
    N_visibility_periods = int(np.sum( np.diff(t_ast_yr*365.25) > 4) + 1)
    if (N_visibility_periods < 12) or (len(ast_obs) < 13): 
        if verbose:
            print('not enough visibility periods!')
        return Nret*[0]
        
    res = fit_full_astrometric_cascade(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, c_funcs = c_funcs, verbose = verbose, show_residuals = show_residuals, ruwe_min=ruwe_min,skip_acceleration=skip_acceleration ) 
    
    # potentially blended, so rerun 
    if (period > 1e4) and (res[-2] < 25) & (res[-6]/res[-5] > 5): 
        t_ast_yr, psi, plx_factor, ast_obs, ast_err = predict_astrometry_luminous_binary(ra = ra, dec = dec, parallax = parallax, pmra = pmra, pmdec = pmdec, m1 = m1, m2 = m2, period = period, Tp = Tp, ecc = ecc, omega = omega, inc = inc_deg*np.pi/180, w=w, phot_g_mean_mag = phot_g_mean_mag, f=f, data_release=data_release, c_funcs=c_funcs, do_blending_noise = True)
        Nret = 23 # number of arguments to return 
        N_visibility_periods = int(np.sum( np.diff(t_ast_yr*365.25) > 4) + 1)
        if (N_visibility_periods < 12) or (len(ast_obs) < 13): 
            if verbose:
                print('not enough visibility periods!')
            return Nret*[0]
        res = fit_full_astrometric_cascade(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, c_funcs = c_funcs, verbose = verbose, show_residuals = show_residuals, ruwe_min=ruwe_min, skip_acceleration=skip_acceleration) 

    return res

def xyz_to_galactic(x, y, z):
    '''
    This is a helper function to translate x, y, z (in pc) to l and b (in degrees). 
    It is mostly needed because the mwdust packaged expects galactic coordinates.
    We assume the sun is at (0, 0, +20.8), and the x axis points from the position of the Sun projected to the Galactic midplane to the Galactic center.
    '''
    z_sun = 20.8 # Sun's height above the galactic plane in pc
    d_pc = np.sqrt(x**2 + y**2 + (z-z_sun)**2)
    b = np.arcsin((z-z_sun)/d_pc)
    cosl, sinl= x/d_pc/np.cos(b), y/d_pc/np.cos(b)
    l = np.arcsin(sinl)
    l[cosl < 0] = np.pi - l[cosl < 0]
    l[(cosl >= 0)*(sinl < 0)] += 2*np.pi
    return l*180/np.pi, b*180/np.pi

def xyz_to_radec(x, y, z):
    '''
    This is a helper function to translate x, y, z (in pc) to ra and dec (in degrees). 
    We assume the sun is at (0, 0, +20.8), and the x axis points from the position of the Sun projected to the Galactic midplane to the Galactic center.
    '''
    z_sun = 20.8  # Sun's height above the galactic plane in pc
    d_pc = np.sqrt(x**2 + y**2 + (z-z_sun)**2)
    cosb, sinb = np.sqrt(d_pc**2 - (z-z_sun)**2)/d_pc, (z-z_sun)/d_pc, 
    cosl, sinl= x/d_pc/cosb, y/d_pc/cosb
    T = np.array([[-0.05487554,  0.49410945, -0.86766614],
       [-0.8734371 , -0.44482959, -0.19807639],
       [-0.48383499,  0.74698225,  0.45598379]])
    eqXYZ = np.dot(T, np.array([cosb*cosl, cosb*sinl, sinb]))
    dec, ra = np.arcsin(eqXYZ[2]), np.arctan2(eqXYZ[1], eqXYZ[0])
    ra[ra < 0] += 2*np.pi
    return ra*180/np.pi, dec*180/np.pi

def draw_from_exponential_disk(N, hz_pc = 300, hR_pc = 2500):
    '''
    this function draws 3D positions from a disk with exponential z and R density profiles with scale height hz_pc and hR_pc. 
    N: how many samples to draw
    returns 3D positions in a coordinate system with the sun at (0, 0, 20.8)
    '''
    z_sun = 20.8
    phi = np.random.uniform(0, 1, N)*2*np.pi
    z = np.random.exponential(scale = hz_pc, size = N)*np.random.choice([-1, 1], size=N)
    
    xx = np.linspace(0, hR_pc*10, 10000)
    cdf = -np.exp(-xx/hR_pc)*(xx/hR_pc + 1) + 1
    R = np.interp(np.random.uniform(0, 1, N), cdf, xx) # this draws from a r*np.exp(-r/h_R) distribution
    x_gal = np.cos(phi)*R + 8.122e3
    y_gal = np.sin(phi)*R
    return x_gal, y_gal, z

def generate_coordinates_at_a_given_distance_exponential_disk(d_min, d_max, N_stars = 1000, hz_pc = 300, hR_pc=2500):
    '''
    this function generates the coordinates (ra, dec, distance) of a set of N_stars sources with distances ranging from d_min to d_max (in pc)
    If d_max is much smaller than hz, it assumes the stellar density is uniform. If it's larger than hz/10 but smaller than hR/3, plane-parallel, an exponential disk is assumed. Finally, if it's larger than hR/3, a disk with exponential density profile in both z and R is assumed. 
    
    This is a helper function for simulate_many_realizations_of_a_single_binary().
    
    returns:
        ra, dec: coordinates in degrees
        d_pc: distance from the sun in pc
        x, y, z: 3D xyz coordinates 
    '''
    z_sun = 20.8  # Sun's height above the galactic plane in pc
    mult_factor = 10 # how many extra stars to generate (will be increased if not enough)
    
    N_found = 0
    while N_found < N_stars:    
        NN = mult_factor*N_stars
        
        if d_max < hz_pc/10: # uniform
            x, y, z = np.random.uniform(-d_max, d_max, size = NN), np.random.uniform(-d_max, d_max, size = NN), np.random.uniform(-d_max, d_max, size = NN) + z_sun
        elif (d_max > hz_pc/10) and (d_max < hR_pc/3): # plane-parallal disk
            x, y = np.random.uniform(-d_max, d_max, size = NN), np.random.uniform(-d_max, d_max, size = NN)
            z = np.random.exponential(scale = hz_pc, size = NN)*np.random.choice([-1, 1], size=NN)
        elif (d_max > hR_pc/3):
            x, y, z = draw_from_exponential_disk(N = NN, hz_pc = hz_pc, hR_pc = hR_pc)
        else:
            raise ValueError('the combination of hR and hz you provided is not appropriate for any of our approximations!')    
                
        d = np.sqrt(x**2 + y**2 + (z-z_sun)**2)
        ok = (d < d_max) & (d > d_min)
        N_found = np.sum(ok)
        mult_factor *= 2
    
    ra, dec = xyz_to_radec(x[ok], y[ok], z[ok])
    d_pc = d[ok]
    ra, dec, d_pc = ra[:N_stars], dec[:N_stars], d_pc[:N_stars]
    return ra, dec, d_pc, x[ok][:N_stars], y[ok][:N_stars], z[ok][:N_stars]
    
def simulate_many_realizations_of_a_single_binary(d_min, d_max, period, Mg_tot, f, m1, m2, ecc, N_realizations = 100, data_release='dr3', do_dust = True, ruwe_min=1.4, skip_acceleration = False):
    '''
    this function generates N_realizations realizations of a binary within a given distance range, for a fixed Porb, absolute magnitude, flux ratio, and eccentricity. Sky positions and orientations will be different for each realization. It generates epoch astrometry for each realization, and then fits that astrometry with the standard astrometric cascade. Finally, it reports what fraction of all realizations resulted in an orbital solution that passes all the DR3 cuts.  
    '''
    ra, dec, d_pc, x,y,z = generate_coordinates_at_a_given_distance_exponential_disk(d_min = d_min, d_max = d_max, N_stars = N_realizations, hz_pc = 300)
    l_deg, b_deg = xyz_to_galactic(x = x, y = y, z = z) 
    
    if do_dust:
        import mwdust
        combined19_ebv = mwdust.Combined19()
        ebv = combined19_ebv(l_deg, b_deg, d_pc/1000)
        A_G = 2.80*ebv
    else: 
        if d_min > 100:
            print('you are ignoring dust. This is probably not a good idea at this distance. ')
        A_G = np.zeros(N_realizations)
 
    Tp = np.random.uniform(0, 1, N_realizations)*period
    omega =  np.random.uniform(0, 2*np.pi, N_realizations) 
    w = np.random.uniform(0, 2*np.pi, N_realizations)
    inc_deg = np.degrees(np.arccos( np.random.uniform(-1, 1, N_realizations)))
    phot_g_mean_mag = Mg_tot + 5*np.log10(d_pc/10) + A_G

    
    def search_mock_binary_worker(i):
        result = run_full_astrometric_cascade(ra = ra[i], dec = dec[i], parallax = 1000/d_pc[i], pmra = 0, pmdec = 0, m1 = m1, m2 = m2, period = period, Tp = Tp[i], ecc = ecc, omega = omega[i], inc_deg = inc_deg[i], w = w[i], phot_g_mean_mag = phot_g_mean_mag[i], f = f, data_release = data_release, c_funcs = None, verbose=False, show_residuals=False, ruwe_min=ruwe_min, skip_acceleration=skip_acceleration)        
        return result
    
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=joblib.cpu_count())(delayed(search_mock_binary_worker)(x) for x in range(N_realizations))

    plx, sig_parallax, A, sig_A, B, sig_B, F, sig_F, G, sig_G, fit_period, sig_period, phi_p, sig_phi_p, fit_ecc, sig_ecc, fit_inc_deg, a0_mas, sigma_a0_mas, N_visibility_periods, N_obs, F2, ruwe = np.array(res).T

    plx_over_err, a0_over_err = plx/sig_parallax, a0_mas/sigma_a0_mas
    
    accepted, ok = np.zeros(len(plx), dtype=bool), fit_period > 0
    passed_cuts = (a0_over_err[ok] > 5) & (plx_over_err[ok] > 20000/fit_period[ok]) & (a0_over_err[ok] > 158/np.sqrt(fit_period[ok])) & (sig_ecc[ok] < 0.079*np.log(fit_period[ok])-0.244) 
    accepted[ok] = passed_cuts

    print('%d out of %d solutions had insufficient visibility periods' % (np.sum(plx == 0), N_realizations) )    
    print('%d out of %d solutions had ruwe < 1.4' % (np.sum(plx == -1), N_realizations) )
    print('%d out of %d solutions got 9-parameter solutions' % (np.sum(plx == -9), N_realizations) )
    print('%d out of %d solutions got 7-parameter solutions' % (np.sum(plx == -7), N_realizations) )
    print('%d out of %d solutions passed all cuts and got an orbital solution!' % (np.sum(accepted), N_realizations) )
    print('%d out of %d solutions got to orbital solutions but failed at least one cut. ' % (np.sum(~accepted & (plx > 0)), N_realizations) )
        
    return ra, dec, d_pc, phot_g_mean_mag, Tp, omega, w, fit_inc_deg, accepted

def predict_radial_velocities(t_rvs_day, period, Tp, ecc, w, K, gamma, c_funcs):
    '''
    this function predicts radial velocities at times t_rvs_day.
    period: orbital period in days
    Tp: periastron time in days
    ecc: eccentricity
    w: "little omega" in radians
    K: RV semi-amplitude in km/s
    gamma: center of mass RV in km/s
    c_funcs: from read_in_C_functions()
    '''
    t_rvs_double = t_rvs_day.astype(np.double)
    results_array = np.zeros(len(t_rvs_day), dtype = np.double)   
     
    c_funcs.predict_radial_velocties(ctypes.c_int(len(t_rvs_day)), ctypes.c_void_p(t_rvs_double.ctypes.data), ctypes.c_void_p(results_array.ctypes.data), ctypes.c_double(period), ctypes.c_double(2*np.pi*Tp/period), ctypes.c_double(ecc), ctypes.c_double(w), ctypes.c_double(K), ctypes.c_double(gamma))
    return results_array
    
    
def predict_astrometry_and_rvs_simultaneously(t_ast_yr, psi, plx_factor, t_rvs_yr, period, Tp, ecc, m1, m2, f, parallax, pmra, pmdec, omega, w, inc_deg, gamma, c_funcs):
    '''
    This function predicts both astrometry and RV curves for the same binary. They can be sampled at different times. 
    t_ast_year: times at which the RVs are sampled, in years, relative to the reference epoch
    psi:  scan angles of the astrometry, in radians
    plx_factor: parallax factors of the astrometry
    t_rvs_day: times at which RVs are measured, in years, relative to THE SAME reference epoch as t_ast_yr
    period: days
    Tp: periastron time, days
    ecc: eccentricity
    m1: mass of star 1, for which we are predicting the RVs, Msun
    m1: mass of star 2,  Msun
    f: flux ratio, F2/F1
    parallax = 1/distance; the true parallax in mas
    pmra, pmdec = proper motions in mas/yr
    omega: "big omega" in radians
    w: "little omega" in radians
    inc_deg: inclination, in degrees
    gamma: center-of-mass RV, km/s
    c_funcs: from read_in_C_functions()
    '''
    EE = solve_kepler_eqn_on_array(M = 2*np.pi/period * (t_ast_yr*365.25 - Tp), ecc = ecc, c_funcs = c_funcs)
    a_mas = get_a_mas(period, m1, m2, parallax)
    inc = inc_deg*np.pi/180
    A_pred = a_mas*( np.cos(w)*np.cos(omega) - np.sin(w)*np.sin(omega)*np.cos(inc) )
    B_pred = a_mas*( np.cos(w)*np.sin(omega) + np.sin(w)*np.cos(omega)*np.cos(inc) )
    F_pred = -a_mas*( np.sin(w)*np.cos(omega) + np.cos(w)*np.sin(omega)*np.cos(inc) )
    G_pred = -a_mas*( np.sin(w)*np.sin(omega) - np.cos(w)*np.cos(omega)*np.cos(inc) )
    cpsi, spsi = np.cos(psi), np.sin(psi)
    
    X = np.cos(EE) - ecc
    Y = np.sqrt(1-ecc**2)*np.sin(EE)
    
    x, y = B_pred*X + G_pred*Y, A_pred*X + F_pred*Y   
    delta_eta = (-y*cpsi - x*spsi) 
    bias = np.array([al_bias_binary(delta_eta = delta_eta[i], q=m2/m1, f=f) for i in range(len(psi))])
    Lambda_pred = pmra*t_ast_yr*spsi + pmdec*t_ast_yr*cpsi + parallax*plx_factor + bias
    
    
    G, Msun = 6.6743e-11, 1.9884098e+30 # SI units
    K1_kms = 0.001*(2*np.pi*G*(m2*Msun) * (m2/(m1 + m2))**2 / (period*86400 *  (1 - ecc**2)**(3/2)))**(1/3) * np.sin(inc_deg*np.pi/180)
    rv_pred = predict_radial_velocities(t_rvs_day = t_rvs_yr*365.25, period = period, Tp = Tp, ecc = ecc, w = w, K = K1_kms, gamma = gamma, c_funcs = c_funcs)
    
    return Lambda_pred, rv_pred
    
def predict_astrometry_single_source(ra, dec, parallax, pmra, pmdec, phot_g_mean_mag, data_release, c_funcs):
    '''
    this function predicts the epoch-level astrometry for single source. 
    ra and dec (degrees): the coordinates of the source at the reference time (which is different for dr3/dr4/dr5)
    parallax (mas): the true parallax (i.e., 1/d)
    pmra, pmdec: true proper motions in mas/yr
    phot_g_mean_mag: G-band magnitude
    f: flux ratio, F2/F1, in the G-band. 
    c_funcs: from read_in_C_functions()
    '''
    
    t = get_gost_one_position(ra, dec, data_release=data_release)
    
    # reject a random 10%
    t = t[np.random.uniform(0, 1, len(t)) > 0.1]
    psi, plx_factor, jds = fetch_table_element(['scanAngle[rad]', 'parallaxFactorAlongScan', 'ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]'], t)
    
    t_ast_yr = rescale_times_astrometry(jd = jds, data_release = data_release)

    N_ccd_avg = 8
    epoch_err_per_transit = al_uncertainty_per_ccd_interp(G = phot_g_mean_mag)/np.sqrt(N_ccd_avg)
    
    if phot_g_mean_mag < 13:
        extra_noise = np.random.uniform(0, 0.04)
    else: 
        extra_noise = 0
    
    Lambda_pred = pmra*t_ast_yr*np.sin(psi) + pmdec*t_ast_yr*np.cos(psi) + parallax*plx_factor 
    Lambda_pred += epoch_err_per_transit*np.random.randn(len(psi)) # modeled noise
    Lambda_pred += extra_noise*np.random.randn(len(psi)) # unmodeled noise

    return t_ast_yr, psi, plx_factor, Lambda_pred, epoch_err_per_transit*np.ones(len(Lambda_pred))

 
def photocenter_orbit_2d_from_thiele_innes(t_ast_yr, parallax, period, ecc, Tp, A, B, F, G,c_funcs):
    '''
    this function calculates the part of a binary's astrometric motion that is due to the orbit, without parallax and proper motion. 
    t_ast_yr: times relative to reference epoch
    parallax: 1/distance; only used to calculate angular size of the orbit
    period: orbital period in days
    ecc: eccentricity 
    A,B,F,G: Thiele-Innnes elements, in mas
    c_funcs: from read_in_C_functions()
    '''
    EE = solve_kepler_eqn_on_array(M = 2*np.pi/period * (t_ast_yr*365.25 - Tp), ecc = ecc, c_funcs = c_funcs)
    X, Y = np.cos(EE) - ecc,  np.sqrt(1-ecc**2)*np.sin(EE)
    x, y = B*X + G*Y, A*X + F*Y       
    return x, y
    
def plot_2d_orbit_and_residuals(t_ast_yr, psi, plx_factor, ast_obs, ast_err, period, ecc, Tp, delta_ra, delta_dec, parallax, pmra, pmdec, A, B, F, G, data_release, c_funcs, ax=None):
    '''
    This function plots the 2D projected orbit of a binary together with the epoch astrometry
    The approach it uses closely follows the pystromety package: https://github.com/Johannes-Sahlmann/pystrometry
    t_ast_year: times at which the astrometric measurements are made, in years, relative to the reference epoch
    psi:  scan angles of the astrometry, in radians
    plx_factor: parallax factors of the astrometry
    ast_obs, ast_err: epoch astrometry measurements in mas
    period: orbital period in days
    ecc: eccentricity 
    Tp: periastron time relative to reference epoch
    delta_ra, delta_dec: position at reference epoch relative to a reference position 
    parallax, pmra, pmdec: in mas and mas/yr
    A, B, F, G: Thiele-Innes coefficients in mas
    data_release: 'dr3' or 'dr4' or 'dr4'
    c_funcs: from read_in_C_functions()
    if ax is not None, incorporate this in another figure 
    '''
    x_periastron, y_periastron = photocenter_orbit_2d_from_thiele_innes(t_ast_yr = np.array([Tp/365.25]), parallax = parallax, period = period, Tp=Tp, ecc = ecc, A=A, B=B, F=F, G=G, c_funcs = c_funcs)
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    ax.plot([0, x_periastron[0]], [0, y_periastron[0]], marker='.', ls='-', lw=0.5, color='0.5')
    ax.plot(x_periastron, y_periastron, marker='s', color='0.5', mfc='0.5', zorder=10)

    t_grid = np.linspace(np.min(t_ast_yr), np.min(t_ast_yr) + period/365.25, 1000)
    x_curve, y_curve = photocenter_orbit_2d_from_thiele_innes(t_ast_yr = t_grid, parallax = parallax, period = period, Tp=Tp, ecc = ecc, A=A, B=B, F=F, G=G, c_funcs = c_funcs)
    ax.plot(x_curve, y_curve, ls='-', lw=1, color='b')
    

    x_epoch, y_epoch = photocenter_orbit_2d_from_thiele_innes(t_ast_yr = t_ast_yr, parallax = parallax, period = period, Tp=Tp, ecc = ecc, A=A, B=B, F=F, G=G, c_funcs = c_funcs)

    orbit_exact = x_epoch*np.sin(psi) + y_epoch*np.cos(psi)
    Lambda_com = (delta_ra + pmra*t_ast_yr)*np.sin(psi) + (delta_dec + pmdec*t_ast_yr)*np.cos(psi) + parallax*plx_factor
    residuals = ast_obs - (orbit_exact + Lambda_com)
    residual_ra, residual_dec = np.sin(psi)*residuals, np.cos(psi)*residuals
    
    ax.plot(x_epoch + residual_ra, y_epoch + residual_dec, marker='o', color='k', ms=4, mfc='k', mec='k', ls='')
                        
    # plot epoch-level error-bars
    for jj in range(len(residuals)):
        x1 = x_epoch[jj] + np.sin(psi)[jj] * (residuals[jj] + ast_err[jj])
        x2 = x_epoch[jj] + np.sin(psi)[jj] * (residuals[jj] - ast_err[jj])
        y1 = y_epoch[jj] + np.cos(psi)[jj] * (residuals[jj] + ast_err[jj])
        y2 = y_epoch[jj] + np.cos(psi)[jj] * (residuals[jj] - ast_err[jj])
        ax.plot([x1, x2], [y1, y2], 'k-', lw=1)
    ax.set_xlabel(r'$\rm \Delta\alpha\cos\delta \,\,[mas]$', fontsize=20)
    ax.set_ylabel(r'$\rm \Delta\delta\,\,[mas]$', fontsize=20)
    ax.invert_xaxis()
    
def get_Campbell_elements(A, B, F, G):
    '''
    Translate between Campbell elements and Thiele-Innes coefficients. Equations from the appendix of Halbwachs+2023. 
    Equations for uncertainties can also be found there but are more complicated and not implemented here. 
    A, B, F, G are Thiele-Innes elements in mas, provided as scalars or arrays.
    Adapted from NSSTools 
    '''
    # Compute wp - Omega and wm - Omega
    wp_minus_Omega = np.arctan2(B - F, A + G)  # Argument of periapsis + ascending node
    wm_minus_Omega = np.arctan2(-B - F, A - G)  # Argument of periapsis - ascending node

    # Initial estimates for w and Omega
    w = (wp_minus_Omega + wm_minus_Omega) / 2.0  # Argument of periapsis
    Omega = (wp_minus_Omega - wm_minus_Omega) / 2.0  # Longitude of ascending node

    # Ensure Omega is between 0 and pi
    Omega = np.where(Omega < 0, Omega + np.pi, Omega)  # Adjust Omega by adding pi
    w = np.where(Omega < 0, w + np.pi, w)  # Adjust w accordingly

    # Calculate tan^2(i/2) using two formulas
    tan2_i_AG = np.abs((A + G) * np.cos(wm_minus_Omega))
    tan2_i_BF = np.abs((F - B) * np.sin(wm_minus_Omega))

    # Choose the formula with the larger denominator for stability
    use_tan2_i_AG = tan2_i_AG > tan2_i_BF
    inclination = np.where(
        use_tan2_i_AG,
        2.0 * np.arctan2(np.sqrt(np.abs((A - G) * np.cos(wp_minus_Omega))), np.sqrt(tan2_i_AG)),
        2.0 * np.arctan2(np.sqrt(np.abs((B + F) * np.sin(wp_minus_Omega))), np.sqrt(tan2_i_BF))
    )

    # Compute semi-major axis
    u = (A**2 + B**2 + F**2 + G**2) / 2.0
    v = A * G - B * F
    sqrt_u2_minus_v2 = np.sqrt((u + v) * (u - v))
    a0 = np.sqrt(u + sqrt_u2_minus_v2)

    # Ensure w is between 0 and 2*pi
    w = np.where(w > 2 * np.pi, w - 2 * np.pi, w)
    w = np.where(w < 0, w + 2 * np.pi, w)

    # Convert to scalars if inputs are scalars
    if np.isscalar(A) and np.isscalar(B) and np.isscalar(F) and np.isscalar(G):
        return float(a0), float(Omega), float(w), float(inclination)
    return a0, Omega, w, inclination
        
def get_companion_mass_from_mass_function(M1, a0_mas, period, parallax, fluxratio, tol=1e-6, max_iter=1000):
    '''
    This function calculates M2 from M1 and the parameters of the astrometric orbit, assuming a flux ratio. 
    It solves the transcendental equation iteratively using Newton's method.
    M1: assumed mass of component 1 in Msun
    a0_mas: photocenter semimajor axis in mas 
    period: orbital period in days
    parallax: 1/distance, in mas
    fluxratio: F2/F1, with 0 corresponding to a dark companion
    tol: tolerance for convergence
    max_iter: maximum number of iterations. 
    '''
    
    fm = (a0_mas/parallax)**3/(period/365.25)**2
    A = fluxratio / (1 + fluxratio)

    def f(x):
        return (M1 + x) * (x / (M1 + x) - A)**3 - fm
    def df_dx(x):
        term1 = (x / (M1 + x) - A)**3
        term2 = 3 * (x / (M1 + x) - A)**2 * (1 / (M1 + x) - x / (M1 + x)**2)
        return term1 + (M1 + x) * term2

    x = M1 / 2  # Start with half the mass of the primary as an initial guess
    for _ in range(max_iter):
        f_x, df_x = f(x), df_dx(x)
        if np.abs(f_x) < tol:
            return x
        if df_x == 0:
            raise ValueError("Derivative is zero. Newton's method fails.")
        x -= f_x / df_x

    # If no solution is found within max_iter iterations
    raise ValueError("Solution did not converge")
    
def get_astrometric_likelihoods_worker(t_ast_yr, psi, plx_factor, ast_obs, ast_err, samples):
    '''
    call this function through multiprocessing
    this function calculates the likelihood of a large number of samples (from generate_prior_samples()). It is useful
    for rejection sampling in the low-SNR regime. 
    t_ast_yr, psi, plx_factor, ast_obs, ast_err: arrays of epoch astrometry
    samples: (P, ecc, phi_p) arrays from generate_prior_samples()
    '''
    c_funcs = read_in_C_functions()
    
    P, ecc, phi_p = samples
    t_ast_yr_double = t_ast_yr.astype(np.double)
    psi_double = psi.astype(np.double)
    plx_factor_double = plx_factor.astype(np.double)
    ast_obs_double = ast_obs.astype(np.double)
    ast_err_double = ast_err.astype(np.double)
    P_double = P.astype(np.double)
    phi_p_double = phi_p.astype(np.double)
    ecc_double = ecc.astype(np.double)

    lnL_array = np.empty(len(P), dtype = np.double)
        
    c_funcs.get_likelihoods_astrometry(ctypes.c_int(len(t_ast_yr)), ctypes.c_int(len(P)), ctypes.c_void_p(t_ast_yr_double.ctypes.data),ctypes.c_void_p(psi_double.ctypes.data), ctypes.c_void_p(plx_factor_double.ctypes.data), ctypes.c_void_p(ast_obs_double.ctypes.data), ctypes.c_void_p(ast_err_double.ctypes.data), ctypes.c_void_p(P_double.ctypes.data), ctypes.c_void_p(phi_p_double.ctypes.data), ctypes.c_void_p(ecc_double.ctypes.data),  ctypes.c_void_p(lnL_array.ctypes.data))
    return lnL_array


def generate_prior_samples(N_samps, P_range = [10, 10000]):
    '''
    This function produces samples of (P, ecc, phi_p), for which we can calculate likelihoods for rejection sampling. 
    uniform in frequency (1/P)
    uniform in ecc
    uniform in phi0 = 2*pi*Tp/P
    N_samps: how many samples to generate
    P_range: over what range to generate samples, uniformly spaced in frequency (not in period)
    '''
    P = 1/np.random.uniform(1/P_range[0], 1/P_range[1], N_samps)
    ecc = np.random.uniform(0, 1, N_samps)
    phi_p = np.random.uniform(0, 2*np.pi, N_samps)
    #Tp = phi0*P/(2*np.pi)
    return (P, ecc, phi_p)
    

def get_astrometric_likelihoods(t_ast_yr, psi, plx_factor, ast_obs, ast_err, samples):
    '''
    this function evaluates the likelihood for a set of astrometric measurements characterized by 
        (t_ast_yr, psi, plx_factor, ast_obs, ast_err) on a set of sample (P, e, phi_0)
    
    on my laptop, this evaluates 10^7 likelihood samples in about 45 seconds 
    returns L = np.exp(lnL)
    '''
    from joblib import Parallel, delayed 
    edges = np.linspace(0, len(samples[0]), joblib.cpu_count()+1).astype(int)
    
    def run_this_j(j):
        these_samples = [samples[i][edges[j]:edges[j+1]] for i in range(3)]
        return get_astrometric_likelihoods_worker(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, samples = these_samples)
        
    res = Parallel(n_jobs=joblib.cpu_count())(delayed(run_this_j)(x) for x in range(joblib.cpu_count()))
    L = np.exp(np.concatenate(res))
    return L
 
    