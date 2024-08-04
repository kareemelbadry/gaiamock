import numpy as np
import matplotlib.pyplot as plt
import os 
import ctypes 
from astropy.table import Table
import healpy as hp
    

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
    
def get_astrometric_chi2(t_ast_yr, psi, plx_factor, ast_obs, ast_err, P, phi_p, ecc, c_funcs):
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
    
    c_funcs.get_chi2_astrometry(ctypes.c_int(len(t_ast_yr)), ctypes.c_void_p(t_ast_yr_double.ctypes.data), 
        ctypes.c_void_p(psi_double.ctypes.data), ctypes.c_void_p(plx_factor_double.ctypes.data), ctypes.c_void_p(ast_obs_double.ctypes.data), ctypes.c_void_p(ast_err_double.ctypes.data), 
         ctypes.c_double(P), ctypes.c_double(phi_p), ctypes.c_double(ecc), ctypes.c_void_p(chi2_array.ctypes.data))
    chi2 = chi2_array[0]
    mu = chi2_array[1:] # ra_off, pmra, dec_off, pmdec, plx, B, G, A, F
    
    return chi2, mu
    

def get_astrometric_residuals_12par(t_ast_yr, psi, plx_factor, ast_obs, ast_err, theta_array, c_funcs):
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

    return np.array(resid_array)


def fit_orbital_solution_nonlinear(t_ast_yr, psi, plx_factor, ast_obs, ast_err, c_funcs, 
    L = np.array([10, 0, 0]), U = np.array([1e4, 2*np.pi, 0.99])):
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
    


def check_ruwe(t_ast_yr, psi, plx_factor, ast_obs, ast_err):
    '''
    This function takes a set of astrometric data (t_ast_yr, psi, plx_factor, ast_obs, ast_err) and fits a 5-parameter solution. It inflates the uncertainties according to the goodness of fit and returns the 5-parameter UWE, best-fit parameters, and uncertainties. 
    '''
    Cinv = np.diag(1/ast_err**2)    
    M = np.vstack([np.sin(psi), t_ast_yr*np.sin(psi), np.cos(psi), t_ast_yr*np.cos(psi), plx_factor]).T 
    mu = np.linalg.solve(M.T @ Cinv @ M, M.T @ Cinv @ ast_obs)  
    Lambda_pred = np.dot(M, mu)
    resids = ast_obs - Lambda_pred
    chi2 = np.sum(resids**2/ast_err**2)
    ruwe = np.sqrt(chi2/(len(psi) - 5))
    
    nu = len(ast_obs) - 5
    cc = np.sqrt(chi2/(nu*(1-2/(9*nu))**3 )) # uncertainy inflation factor
    
    cov_matrix = np.linalg.inv(M.T @ Cinv @ M)
    sigma_mu = cc*np.sqrt(np.diag(cov_matrix))
    
    return ruwe, mu, sigma_mu

def check_7par(t_ast_yr, psi, plx_factor, ast_obs, ast_err):
    '''
    This function takes a set of astrometric data (t_ast_yr, psi, plx_factor, ast_obs, ast_err) and fits a 7-parameter acceleration solution. It inflates the uncertainties according to the goodness of fit and returns the best-fit parameters and uncertainties and F2 and significance associated with the solution. 
    '''
    Cinv = np.diag(1/ast_err**2)    
    M = np.vstack([np.sin(psi), t_ast_yr*np.sin(psi), 1/2*t_ast_yr**2*np.sin(psi), np.cos(psi), t_ast_yr*np.cos(psi), 1/2*t_ast_yr**2*np.cos(psi), plx_factor]).T 
    mu = np.linalg.solve(M.T @ Cinv @ M, M.T @ Cinv @ ast_obs)  # ra, pmra, pmra_dot, dec, pmdec, pmdec_dot, plx
    
    Lambda_pred = np.dot(M, mu)
    resids = ast_obs - Lambda_pred
    chi2 = np.sum(resids**2/ast_err**2)
    nu = len(ast_obs) - 7 
    F2 = np.sqrt(9*nu/2)*( (chi2/nu)**(1/3) + 2/(9*nu) -1 )
    cc = ((F2*np.sqrt(2/(9*nu))+1 - 2/(9*nu))/(1-2/(9*nu)) )**(3/2) 
    
    
    cov_matrix = np.linalg.inv(M.T @ Cinv @ M)
    sigma_mu = cc*np.sqrt(np.diag(cov_matrix))
    p1, p2, sig1, sig2 = mu[2], mu[5], sigma_mu[2], sigma_mu[5]
    rho12 = cov_matrix[2][5]/(sig1*sig2)
    s = 1/(sig1*sig2)*np.sqrt((p1**2*sig2**2 + p2**2*sig1**2 -2*p1*p2*rho12*sig1*sig2)/(1-rho12**2))
    return F2, s, mu, sigma_mu
    
def check_9par(t_ast_yr, psi, plx_factor, ast_obs, ast_err):
    '''
    This function takes a set of astrometric data (t_ast_yr, psi, plx_factor, ast_obs, ast_err) and fits a 9-parameter acceleration solution. It inflates the uncertainties according to the goodness of fit and returns the best-fit parameters and uncertainties and F2 and significance associated with the solution. 
    '''
    Cinv = np.diag(1/ast_err**2)    
    M = np.vstack([np.sin(psi), t_ast_yr*np.sin(psi), 1/2*t_ast_yr**2*np.sin(psi),  1/6*t_ast_yr**3*np.sin(psi), np.cos(psi), t_ast_yr*np.cos(psi), 1/2*t_ast_yr**2*np.cos(psi), 1/6*t_ast_yr**3*np.cos(psi), plx_factor]).T 
    mu = np.linalg.solve(M.T @ Cinv @ M, M.T @ Cinv @ ast_obs)  # ra, pmra, pmra_dot, pmra_ddot, dec, pmdec, pmdec_dot, pmdec_ddot, plx
    
    Lambda_pred = np.dot(M, mu)
    resids = ast_obs - Lambda_pred
    chi2 = np.sum(resids**2/ast_err**2)
    nu = len(ast_obs) - 9
    F2 = np.sqrt(9*nu/2)*( (chi2/nu)**(1/3) + 2/(9*nu) -1 )
    cc = ((F2*np.sqrt(2/(9*nu))+1 - 2/(9*nu))/(1-2/(9*nu)) )**(3/2) 
    
    cov_matrix = np.linalg.inv(M.T @ Cinv @ M)
    sigma_mu = cc*np.sqrt(np.diag(cov_matrix))
    p1, p2, sig1, sig2 = mu[3], mu[6], sigma_mu[3], sigma_mu[6]
    rho12 = cov_matrix[3][6]/(sig1*sig2)
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
    
def predict_astrometry_luminous_binary(ra, dec, parallax, pmra, pmdec, m1, m2, period, Tp, ecc, omega, inc, w, epoch_err_mas, f, data_release, c_funcs, extra_noise = 0.0):
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
    epoch_err_mas: this is the uncertainty in AL displacement per FOV transit (not per CCD)
    f: flux ratio, F2/F1, in the G-band. 
    data_release: 'dr3', 'dr4', or 'dr5'
    c_funcs: from read_in_C_functions()
    extra_noise: additional noise to optionally be added to the epoch astrometry, without inflating formal uncertainties.
    '''
    
    t = get_gost_one_position(ra, dec, data_release=data_release)
    
    # reject a random 10%
    t = t[np.random.uniform(0, 1, len(t)) > 0.1]
    psi, plx_factor, jds = fetch_table_element(['scanAngle[rad]', 'parallaxFactorAlongScan', 'ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]'], t)
    
    if data_release == 'dr4':
        t_ast_day = jds - 2457936.875
    elif data_release == 'dr3':
        t_ast_day = jds - 2457389.0
    elif data_release == 'dr5':
        t_ast_day = jds - 2458818.5
    else: 
        raise ValueError('invalid data_release!')

    t_ast_yr = t_ast_day/365.25
    EE = solve_kepler_eqn_on_array(M = 2*np.pi/period * (t_ast_day - Tp), ecc = ecc, c_funcs = c_funcs)
    a_mas = get_a_mas(period, m1, m2, parallax)
    A_pred = a_mas*( np.cos(w)*np.cos(omega) - np.sin(w)*np.sin(omega)*np.cos(inc) )
    B_pred = a_mas*( np.cos(w)*np.sin(omega) + np.sin(w)*np.cos(omega)*np.cos(inc) )
    F_pred = -a_mas*( np.sin(w)*np.cos(omega) + np.cos(w)*np.sin(omega)*np.cos(inc) )
    G_pred = -a_mas*( np.sin(w)*np.sin(omega) - np.cos(w)*np.cos(omega)*np.cos(inc) )
    cpsi, spsi = np.cos(psi), np.sin(psi)
    
    X = np.cos(EE) - ecc
    Y = np.sqrt(1-ecc**2)*np.sin(EE)
    
    x, y = B_pred*X + G_pred*Y, A_pred*X + F_pred*Y   
    rho = np.sqrt(x**2 + y**2) 
    delta_eta = (-y*cpsi - x*spsi) 
    bias = np.array([al_bias_binary(delta_eta = delta_eta[i], q=m2/m1, f=f) for i in range(len(psi))])
    Lambda_com = pmra*t_ast_yr*spsi + pmdec*t_ast_yr*cpsi + parallax*plx_factor # barycenter motion
    Lambda_pred = Lambda_com + bias # binary motion

    Lambda_pred += epoch_err_mas*np.random.randn(len(psi)) # modeled noise
    Lambda_pred += extra_noise*np.random.randn(len(psi)) # unmodeled noise

    return t_ast_yr, psi, plx_factor, Lambda_pred, epoch_err_mas*np.ones(len(Lambda_pred))

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
    
    chi2 = np.sum((ast_obs - Lambda_pred)**2/ast_err**2 )
    nu = len(ast_obs) - 12 # dof
    F2 = np.sqrt(9*nu/2)*( (chi2/nu)**(1/3) + 2/(9*nu) -1 )
    
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
    nu = len(ast_obs) - 9
    F2 = np.sqrt(9*nu/2)*( (chi2/nu)**(1/3) + 2/(9*nu) -1 )    
    
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
    nu = len(ast_obs) - 7 
    F2 = np.sqrt(9*nu/2)*( (chi2/nu)**(1/3) + 2/(9*nu) -1 )   
    
    print('7-parameter chi2: %.2f' % chi2 )
    ax[1].errorbar(t_ast_yr, ast_obs - Lambda_pred, yerr=ast_err, fmt='k.')
    ax[1].set_xlabel('time (years)', fontsize=20)
    ax[0].set_ylabel('residual (5 par)', fontsize=20)
    ax[1].set_ylabel('residual (7 par)', fontsize=20)

    
def get_uncertainties_at_best_fit_binary_solution(t_ast_yr, psi, plx_factor, ast_obs, ast_err, p0, c_funcs):
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
        return get_astrometric_residuals_12par(t_ast_yr, psi, plx_factor, ast_obs, ast_err, theta_array = np.array(theta), c_funcs = c_funcs)
            
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
        chi2 = np.sum(resid_func(p0)**2)
        nu = 12
        cc = np.sqrt(chi2/(nu*(1-2/(9*nu))**3 )) # uncertainy inflation factor
        
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
        
        covAB, covAF, covAG = cov_x[8, 9], cov_x[8,10], cov_x[8, 11]
        covBF, covBG, covFG = cov_x[9,10], cov_x[9,11], cov_x[10,11]
        
        sigma_a0 = 1 / (2 * a0) * np.sqrt(
            t_A**2 * sig_A**2 + t_B**2 * sig_B**2 + t_F**2 * sig_F**2 + t_G**2 * sig_G**2 +
            2 * t_A * t_B * covAB + 2 * t_A * t_F * covAF + 2 * t_A * t_G * covAG +
            2 * t_B * t_F * covBF + 2 * t_B * t_G * covBG + 2 * t_F * t_G * covFG
        )
    
    else:
        uncertainties = np.ones(len(p0))*1000
        a0, sigma_a0, inc_deg = 0.01, 100, 0
    
    return uncertainties, a0, sigma_a0, inc_deg

def run_full_astrometric_cascade(ra, dec, parallax, pmra, pmdec, m1, m2, period, Tp, ecc, omega, inc_deg, w, phot_g_mean_mag, f, data_release, c_funcs, verbose=False, show_residuals=False):
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
    N_ccd_avg = 8
    epoch_err_per_transit = al_uncertainty_per_ccd_interp(G = phot_g_mean_mag)/np.sqrt(N_ccd_avg)
    
    if phot_g_mean_mag < 13:
        extra_noise = np.random.uniform(0, 0.04)
    else: 
        extra_noise = 0
    
    t_ast_yr, psi, plx_factor, ast_obs, ast_err = predict_astrometry_luminous_binary(ra = ra, dec = dec, parallax = parallax, pmra = pmra, pmdec = pmdec, m1 = m1, m2 = m2, period = period, Tp = Tp, ecc = ecc, omega = omega, inc = inc_deg*np.pi/180, w=w, epoch_err_mas = epoch_err_per_transit, f=f, data_release=data_release, c_funcs=c_funcs, extra_noise=extra_noise)
    
    Nret = 22 # number of arguments to return 
    N_visibility_periods = int(np.sum( np.diff(t_ast_yr*365.25) > 4) + 1)
    if (N_visibility_periods < 12) or (len(ast_obs) < 13): 
        if verbose:
            print('not enough visibility periods!')
        return Nret*[0]
    
    # check 5-parameter solution 
    ruwe, mu, sigma_mu = check_ruwe(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err)
    if ruwe < 1.4:
        res =  Nret*[-1] # set most return arguments to -1, but save a few parameters for convenience 
        res[1] = ruwe 
        res[2] = mu[-1]
        res[3] = sigma_mu[-1]
        
        if verbose:
            print('UWE < 1.4: returning only 5-parameter solution.')
        return res
    
    F2_9par, s_9par, mu, sigma_mu = check_9par(t_ast_yr, psi, plx_factor, ast_obs, ast_err)
    plx_over_err9 = mu[-1]/sigma_mu[-1]
    if (F2_9par < 25) and (s_9par > 12) and (plx_over_err9 > 2.1*s_9par**1.05):
        res =  Nret*[-9]  # set most return arguments to -9, but save a few parameters for convenience 
        res[1] = s_9par
        res[2] = mu[-1]
        res[3] = sigma_mu[-1]
        
        if verbose:
            print('9 parameter solution accepted! Not trying anything else.')
            print('s9: %.1f, plx_over_err9: %.1f, F2_9: %.1f' % (s_9par, plx_over_err9, F2_9par))
        if show_residuals:
            plot_residuals_9par(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, theta_array = mu, c_funcs = c_funcs)
        return res
        
    
    F2_7par, s_7par, mu, sigma_mu = check_7par(t_ast_yr, psi, plx_factor, ast_obs, ast_err)
    plx_over_err7 = mu[-1]/sigma_mu[-1]
    if (F2_7par < 25) and (s_7par > 12) and (plx_over_err7 > 1.2*s_7par**1.05):
        res =  Nret*[-7]
        res[1] = s_7par
        res[2] = mu[-1]
        res[3] = sigma_mu[-1]
        
        if verbose:
            print('7 parameter solution accepted! Not trying anything else.')
        if show_residuals:
            plot_residuals_7par(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, theta_array = mu, c_funcs = c_funcs)
        return res

    res = fit_orbital_solution_nonlinear(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, c_funcs = c_funcs)
    if verbose:
        print('found best-fit nonlinear parameters:', res)
    
    # get the linear parameters 
    period, phi_p, ecc = res
    chi2, mu_linear = get_astrometric_chi2(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, P = period, phi_p = phi_p, ecc = ecc, c_funcs=c_funcs)
    ra_off, pmra, dec_off, pmdec, plx, B, G, A, F = mu_linear
    p0 = [ra_off, dec_off, plx, pmra, pmdec, period, ecc, phi_p, A, B, F, G]
    
    # get some uncertainties 
    errors, a0_mas, sigma_a0_mas, inc_deg = get_uncertainties_at_best_fit_binary_solution(t_ast_yr = t_ast_yr, psi = psi, plx_factor = plx_factor, ast_obs = ast_obs, ast_err = ast_err, p0 = p0, c_funcs = c_funcs)
    sig_ra,sig_dec,sig_parallax,sig_pmra,sig_pmdec,sig_period,sig_ecc,sig_phi_p,sig_A,sig_B, sig_F,sig_G = errors
    F2 = np.sqrt(9*(len(ast_obs) - 12)/2)*( (chi2/(len(ast_obs) - 12))**(1/3) + 2/(9*(len(ast_obs) - 12)) -1 )
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
    return_array = [plx, sig_parallax, A, sig_A, B, sig_B, F, sig_F, G, sig_G, period, sig_period, phi_p, sig_phi_p, ecc, sig_ecc, inc_deg, a0_mas, sigma_a0_mas, N_visibility_periods, len(t_ast_yr), F2]
    return return_array
    