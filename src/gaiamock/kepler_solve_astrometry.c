/* Helper functions for quickly solving Kepler's equation and some other utilities for fitting astrometric binaries. 
This is compiled automatically (see setup.py) when gaiamock is installed with pip, and gsl needs to be linked for it to be compiled.
To compile it by hand instead, run gcc inside this directory. On my Macbook, I installed gsl using homebrew and then compiled with:
 gcc -shared -o kepler_solve_astrometry.so kepler_solve_astrometry.c -I/opt/homebrew/Cellar/gsl/2.8/include  -L/opt/homebrew/Cellar/gsl/2.8/lib -lgsl -lgslcblas -lm 

On my local cluster, I compiled with:
 gcc -shared -o kepler_solve_astrometry.so kepler_solve_astrometry.c -lgsl -lgslcblas -lm -fPIC */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_errno.h>
#define PI 3.14159265358979

#ifndef ASTFIT_N_PERIOD
#define ASTFIT_N_PERIOD 180
#endif
#ifndef ASTFIT_N_SHORT_PERIOD
#define ASTFIT_N_SHORT_PERIOD 560
#endif
#ifndef ASTFIT_N_ECC_PHASE
#define ASTFIT_N_ECC_PHASE 36
#endif
#ifndef ASTFIT_MAX_PERIODS_KEEP
#define ASTFIT_MAX_PERIODS_KEEP 20
#endif
#ifndef ASTFIT_MAX_STARTS
#define ASTFIT_MAX_STARTS 48
#endif
#ifndef ASTFIT_NM_MAX_ITER
#define ASTFIT_NM_MAX_ITER 250
#endif
#ifndef ASTFIT_NM_FSPREAD_TOL
#define ASTFIT_NM_FSPREAD_TOL 1e-4
#endif
#ifndef ASTFIT_NM_XSPREAD_TOL
#define ASTFIT_NM_XSPREAD_TOL 3e-6
#endif

#define ASTFIT_FULL_N_PERIOD 260
#define ASTFIT_FULL_N_SHORT_PERIOD 800
#define ASTFIT_FULL_N_ECC_PHASE 48
#define ASTFIT_FULL_MAX_PERIODS_KEEP 24
#define ASTFIT_FULL_MAX_STARTS 64
#define ASTFIT_FULL_NM_MAX_ITER 400
#define ASTFIT_FULL_NM_FSPREAD_TOL 1e-6
#define ASTFIT_FULL_NM_XSPREAD_TOL 1e-7


/* This is a helper function for solving Kepler's equation. From http://alpheratz.net/dynamics/twobody/KeplerIterations_summary.pdf */
double eps3(double ecc, double M, double x){
    double t1; double t2; double t3; double t4; double t5; double t6;
    t1 = cos(x);
    t2 = -1+ecc*t1;
    t3 = sin(x);
    t4 = ecc*t3;
    t5 = -x + t4 + M;
    t6 = t5/(0.5*t5*t4/t2 + t2);
    return t5/((0.5*t3 - (1.0/6.0)*t1*t6)*ecc*t6+t2);
} 

/* This is another helper function for finding a good first guess when solving Kepler's equation. From http://alpheratz.net/dynamics/twobody/KeplerIterations_summary.pdf */
double KeplerStart3(double ecc, double M){
    double t33; double t35; double t34;
    t34 = ecc*ecc;
    t35 = ecc*t34;
    t33 = cos(M);
    return M+(-0.5*t35+ecc+(t34+1.5*t33*t35)*t33)*sin(M);
}

/* This function solves Kepler's equation for a mean anomally Mi */ 
double solve_Kepler_equation(double Mi, double ecc, double xtol){
    double eps = 1;
    double EE0 = 0;
    double Mnorm = fmod(Mi, 2*PI);
    if (Mnorm < 0.0) {
        Mnorm += 2*PI;
    }
    double EE = KeplerStart3(ecc, Mnorm);
    int n_iter = 0;
    while(eps > xtol && n_iter < 100){
        EE0 = EE;
        EE = EE0 - eps3(ecc, Mnorm, EE0);
        eps = fabs(EE0 - EE);
        n_iter++;
    }
    return EE;
}

/* This function applies the solve_Kepler_equation() function above to an array of mean anomalies. It populates the results into E_array, an array of eccentric anomalies */
void solve_Kepler_equation_array(int n_obs, double *M_array, double ecc, double xtol, double *E_array){
    for(int i = 0; i < n_obs; i++){
        E_array[i] = solve_Kepler_equation(M_array[i], ecc, xtol);
    }
}

/* This function predicts radial velocities at an array of mjds (length n). "RVs" is an input array that will be modified by the function. */
void predict_radial_velocties(int n, double *mjds, double *RVs, double P, double phi_p, double ecc, 
    double omega, double K, double gamm){
    double xtol = 1e-10;
    double fi; 
    double Ei;
    double Mi;
    for(int i = 0; i < n; i++) {
        Mi = 2*PI*mjds[i]/P - phi_p;        
        Ei = solve_Kepler_equation(Mi, ecc, xtol);
        fi = 2*atan2(sqrt(1+ecc)*sin(0.5*Ei), sqrt(1-ecc)*cos(0.5*Ei));
        RVs[i] = K*(cos(fi + omega) + ecc*cos(omega)) + gamm;
    }
}

/* This is a beta function, which we occasionally use for an eccentricity prior */

static inline double log_beta_prior(double x, double a, double b) {
    if (x <= 0.0 || x >= 1.0) return -INFINITY;

    const double logB = lgamma(a) + lgamma(b) - lgamma(a + b);
    return (a - 1.0) * log(x) + (b - 1.0) * log1p(-x) - logB;
}


/* This function takes a length-12 theta array:
theta_array = (ra_off, dec_off, parallax, pmra, pmdec, period, ecc, phi_p, A, B, F, G) 
it calculates the predicted epoch astrometry at a times t_ast_yr, with scan angles psi andparallax factors plx_factor, 
and then calculates the chi^2 given the observed epoch astrometry ast_obs and ast_err. The chi^2 is inserted into an array chi2_array.  
*/
void get_chi2_12par_solution(int n_obs, double *t_ast_yr, double *psi, double *plx_factor, double *ast_obs, double *ast_err, double *theta_array, double *chi2_array) {
    double xtol = 1e-10;
    double Ei;
    double Mi;
    double X;
    double Y; 
    double chi2 = 0;
    double Lambda_pred;
    
    for(int j = 0; j < n_obs; j++){
        Mi = 2*PI*t_ast_yr[j]*365.25/theta_array[5] - theta_array[7];
        Ei = solve_Kepler_equation(Mi, theta_array[6], xtol);
        
        X = cos(Ei) - theta_array[6];
        Y = sqrt(1-theta_array[6]*theta_array[6])*sin(Ei);
        
        Lambda_pred = (theta_array[0] + theta_array[3]*t_ast_yr[j] + theta_array[9]*X + theta_array[11]*Y)*sin(psi[j]) + (theta_array[1] + theta_array[4]*t_ast_yr[j] + theta_array[8]*X + theta_array[10]*Y)*cos(psi[j]) + plx_factor[j]*theta_array[2]; 
        chi2 += (Lambda_pred - ast_obs[j])*(Lambda_pred - ast_obs[j])/(ast_err[j]*ast_err[j]);
    }
    chi2_array[0] = chi2;
}

/* This function takes a length-12 theta array
theta_array = (ra_off, dec_off, parallax, pmra, pmdec, period, ecc, phi_p, A, B, F, G)  
it calculates the predicted epoch astrometry at a times t_ast_yr, with scan angles psi and parallax factors plx_factor, 
and then fills in an array of uncertainty-scaled residuals, (ast_pred - ast_obs)/ast_err. This is used for calculating the Jacobian. 
 */ 
void get_residual_array_12par_solution(int n_obs, double *t_ast_yr, double *psi, double *plx_factor, double *ast_obs, double *ast_err, double *theta_array, double *residual_array) {
    double xtol = 1e-10;
    double Ei;
    double Mi;
    double X;
    double Y; 
    double Lambda_pred;
    
    for(int j = 0; j < n_obs; j++){
        Mi = 2*PI*t_ast_yr[j]*365.25/theta_array[5] - theta_array[7];
        Ei = solve_Kepler_equation(Mi, theta_array[6], xtol);
        
        X = cos(Ei) - theta_array[6];
        Y = sqrt(1-theta_array[6]*theta_array[6])*sin(Ei);
        
        Lambda_pred = (theta_array[0] + theta_array[3]*t_ast_yr[j] + theta_array[9]*X + theta_array[11]*Y)*sin(psi[j]) + (theta_array[1] + theta_array[4]*t_ast_yr[j] + theta_array[8]*X + theta_array[10]*Y)*cos(psi[j]) + plx_factor[j]*theta_array[2]; 
        residual_array[j] = (Lambda_pred - ast_obs[j])/ast_err[j];
    }
}


/* This function takes a length-12 theta array
theta_array = (ra_off, dec_off, parallax, pmra, pmdec, period, ecc, phi_p, w, Omega, a0_mas, inc)  
it calculates the predicted epoch astrometry at a times t_ast_yr, with scan angles psi and parallax factors plx_factor, 
and then fills in an array of uncertainty-scaled residuals, (ast_pred - ast_obs)/ast_err. This is used for calculating the Jacobian. w, Omega, and inc are all in radians; a0 is in mas. 
 */ 
void get_residual_array_12par_solution_campbell(int n_obs, double *t_ast_yr, double *psi, double *plx_factor, double *ast_obs, double *ast_err, double *theta_array, double *residual_array) {
    double xtol = 1e-10;
    double Ei;
    double Mi;
    double X;
    double Y; 
    double Lambda_pred;
    
    double A = theta_array[10] * (cos(theta_array[8]) * cos(theta_array[9]) - sin(theta_array[8]) * sin(theta_array[9]) * cos(theta_array[11]));
    double B = theta_array[10] * (cos(theta_array[8]) * sin(theta_array[9]) + sin(theta_array[8]) * cos(theta_array[9]) * cos(theta_array[11]));
    double F = -theta_array[10] * (sin(theta_array[8]) * cos(theta_array[9]) + cos(theta_array[8]) * sin(theta_array[9]) * cos(theta_array[11]));
    double G = -theta_array[10] * (sin(theta_array[8]) * sin(theta_array[9]) - cos(theta_array[8]) * cos(theta_array[9]) * cos(theta_array[11]));
    
    
    for(int j = 0; j < n_obs; j++){
        Mi = 2*PI*t_ast_yr[j]*365.25/theta_array[5] - theta_array[7];
        Ei = solve_Kepler_equation(Mi, theta_array[6], xtol);
        
        X = cos(Ei) - theta_array[6];
        Y = sqrt(1-theta_array[6]*theta_array[6])*sin(Ei);
        
        Lambda_pred = (theta_array[0] + theta_array[3]*t_ast_yr[j] + B*X + G*Y)*sin(psi[j]) + (theta_array[1] + theta_array[4]*t_ast_yr[j] + A*X + F*Y)*cos(psi[j]) + plx_factor[j]*theta_array[2]; 
        residual_array[j] = (Lambda_pred - ast_obs[j])/ast_err[j];
    }
}


/*  This function takes a single set of (P, phi_p, ecc). It then solves for the best-fit linear parameters and predicts the epoch astrometry at time t_ast_yr, and calculates the corresponding chi2. n_obs is length of the observations array.
the first element of chi2_array will be the chi2. The next 9 elements are the best-fit linear parameters for this (P, phi_p, ecc) */
void get_chi2_astrometry(int n_obs, double *t_ast_yr, double *psi, double *plx_factor, double *ast_obs, double *ast_err, double P, double phi_p, double ecc, double *chi2_array){
    double xtol = 1e-10;
    double Ei;
    double Mi;
    double X[n_obs];
    double Y[n_obs]; 
    
    gsl_matrix *M = gsl_matrix_calloc(n_obs, 9);
    gsl_matrix *Mw = gsl_matrix_calloc(n_obs, 9);
    gsl_matrix *MtCinvM = gsl_matrix_alloc(9, 9); // Result of Mt * Cinv * M
    gsl_vector *MtCinvY = gsl_vector_alloc(9); // Result of Mt * Cinv * ast_obs
    gsl_vector *mu = gsl_vector_alloc(9); // Solution vector
    gsl_vector *Lambda_pred = gsl_vector_alloc(n_obs); // Predicted lambda
    gsl_vector *weighted_ast_obs = gsl_vector_alloc(n_obs);
    
    for(int j = 0; j < n_obs; j++){
        Mi = 2*PI*t_ast_yr[j]*365.25/P - phi_p;
        Ei = solve_Kepler_equation(Mi, ecc, xtol);
        
        X[j] = cos(Ei) - ecc;
        Y[j] = sqrt(1-ecc*ecc)*sin(Ei);
    }
      
    for (int j = 0; j < n_obs; j++) {
        gsl_matrix_set(M, j, 0, sin(psi[j]) );
        gsl_matrix_set(M, j, 1, t_ast_yr[j] * sin(psi[j]) );
        gsl_matrix_set(M, j, 2, cos(psi[j]) );
        gsl_matrix_set(M, j, 3, t_ast_yr[j] * cos(psi[j]));
        gsl_matrix_set(M, j, 4, plx_factor[j]);
        gsl_matrix_set(M, j, 5, X[j] * sin(psi[j]) );
        gsl_matrix_set(M, j, 6, Y[j] * sin(psi[j]) );
        gsl_matrix_set(M, j, 7, X[j] * cos(psi[j]));
        gsl_matrix_set(M, j, 8, Y[j] * cos(psi[j]));
        double inv_err = 1.0 / ast_err[j];
        for (int k = 0; k < 9; k++) {
            gsl_matrix_set(Mw, j, k, gsl_matrix_get(M, j, k) * inv_err);
        }
        gsl_vector_set(weighted_ast_obs, j, ast_obs[j] * inv_err);
    }

    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, Mw, Mw, 0.0, MtCinvM);
    gsl_blas_dgemv(CblasTrans, 1.0, Mw, weighted_ast_obs, 0.0, MtCinvY);
    gsl_linalg_HH_solve(MtCinvM, MtCinvY, mu);
    gsl_blas_dgemv(CblasNoTrans, 1.0, M, mu, 0.0, Lambda_pred);
    

    double chi2 = 0.0; 
    for (int j = 0; j < n_obs; j++) {
        double pred = gsl_vector_get(Lambda_pred, j);
        double obs = ast_obs[j];
        double err = ast_err[j];
        chi2 += ((pred - obs) * (pred - obs)) / (err * err);
    }
    chi2_array[0] = chi2;
     
    for (int j = 1; j < 10; j++) {
        chi2_array[j] = gsl_vector_get(mu, j-1);
    }


    // Cleanup
    gsl_matrix_free(M);
    gsl_matrix_free(Mw);
    gsl_matrix_free(MtCinvM);
    gsl_vector_free(MtCinvY);
    gsl_vector_free(mu);
    gsl_vector_free(Lambda_pred);
    gsl_vector_free(weighted_ast_obs);
}


/*  This function is identical to get_chi2_astrometry(), except that it includes a prior on parallax. It is useful if you already know the distance you your binary, e.g. because it's in a cluster. The prior is parallax = pi0 \pm sig_pi. The array will be chi_array = chi2, ra_off, pmra, dec_off, pmdec, plx, B, G, A, F */
void get_chi2_astrometry_parallax_prior(int n_obs, double *t_ast_yr, double *psi, double *plx_factor, double *ast_obs, double *ast_err, double P, double phi_p, double ecc, double pi0, double sig_pi, double *chi2_array){
    double xtol = 1e-10;
    double Ei;
    double Mi;
    double X[n_obs];
    double Y[n_obs]; 
    double ivar[n_obs];
    
    gsl_matrix *M = gsl_matrix_calloc(n_obs, 9);
    gsl_matrix *Mt = gsl_matrix_alloc(9, n_obs);
    
    gsl_matrix *Cinv = gsl_matrix_calloc(n_obs, n_obs);
    gsl_matrix *MtCinvM = gsl_matrix_alloc(9, 9); // Result of Mt * Cinv * M
    gsl_vector *MtCinvY = gsl_vector_alloc(9); // Result of Mt * Cinv * ast_obs
    gsl_vector *mu = gsl_vector_alloc(9); // Solution vector
    gsl_vector *Lambda_pred = gsl_vector_alloc(n_obs); // Predicted lambda
    gsl_matrix *temp = gsl_matrix_alloc(9, n_obs); // Temporary matrix for intermediate calculations

    // Compute ivar and fill Cinv
    for (int j = 0; j < n_obs; j++) {
        ivar[j] = 1.0 / (ast_err[j] * ast_err[j]); // Compute inverse variance
        gsl_matrix_set(Cinv, j, j, ivar[j]); // Set diagonal elements of Cinv
    }    
    
    for(int j = 0; j < n_obs; j++){
        Mi = 2*PI*t_ast_yr[j]*365.25/P - phi_p;
        Ei = solve_Kepler_equation(Mi, ecc, xtol);
        
        X[j] = cos(Ei) - ecc;
        Y[j] = sqrt(1-ecc*ecc)*sin(Ei);
    }
      
    for (int j = 0; j < n_obs; j++) {
        gsl_matrix_set(M, j, 0, sin(psi[j]) );
        gsl_matrix_set(M, j, 1, t_ast_yr[j] * sin(psi[j]) );
        gsl_matrix_set(M, j, 2, cos(psi[j]) );
        gsl_matrix_set(M, j, 3, t_ast_yr[j] * cos(psi[j]));
        gsl_matrix_set(M, j, 4, plx_factor[j]);
        gsl_matrix_set(M, j, 5, X[j] * sin(psi[j]) );
        gsl_matrix_set(M, j, 6, Y[j] * sin(psi[j]) );
        gsl_matrix_set(M, j, 7, X[j] * cos(psi[j]));
        gsl_matrix_set(M, j, 8, Y[j] * cos(psi[j]));
    }
    gsl_matrix_transpose_memcpy(Mt, M);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Mt, Cinv, 0.0, temp);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, temp, M, 0.0, MtCinvM);
    
    gsl_vector *Cinv_ast_obs = gsl_vector_alloc(n_obs); // Allocate vector for the result of Cinv @ ast_obs
    gsl_vector_view ast_obs_view = gsl_vector_view_array(ast_obs, n_obs); // Create a vector view of ast_obs
    gsl_blas_dgemv(CblasNoTrans, 1.0, Cinv, &ast_obs_view.vector, 0.0, Cinv_ast_obs);
    gsl_blas_dgemv(CblasNoTrans, 1.0, Mt, Cinv_ast_obs, 0.0, MtCinvY);
    
    // add the parallax prior
    if (sig_pi > 0) {
        const int k = 4;                      // parallax coefficient index (column 4 is plx_factor)
        const double w = 1.0/(sig_pi*sig_pi);
        gsl_matrix_set(MtCinvM, k, k, gsl_matrix_get(MtCinvM, k, k) + w);
        gsl_vector_set(MtCinvY, k, gsl_vector_get(MtCinvY, k) + w*pi0);
    }
    
    gsl_linalg_HH_solve(MtCinvM, MtCinvY, mu);
    gsl_blas_dgemv(CblasNoTrans, 1.0, M, mu, 0.0, Lambda_pred);
    

    double chi2 = 0.0; 
    for (int j = 0; j < n_obs; j++) {
        double pred = gsl_vector_get(Lambda_pred, j);
        double obs = ast_obs[j];
        double err = ast_err[j];
        chi2 += ((pred - obs) * (pred - obs)) / (err * err);
    }
    
    // add a penalty for the parallax prior 
    if (sig_pi > 0) {
        double pi_hat = gsl_vector_get(mu, 4);
        chi2 += (pi_hat - pi0)*(pi_hat - pi0)/(sig_pi*sig_pi);
    }
    
    chi2_array[0] = chi2;
     
    for (int j = 1; j < 10; j++) {
        chi2_array[j] = gsl_vector_get(mu, j-1);
    }


    // Cleanup
    gsl_matrix_free(M);
    gsl_matrix_free(Mt);
    gsl_matrix_free(Cinv);
    gsl_matrix_free(MtCinvM);
    gsl_vector_free(MtCinvY);
    gsl_vector_free(mu);
    gsl_vector_free(Lambda_pred);
    gsl_matrix_free(temp);
    gsl_vector_free(Cinv_ast_obs);
}


/*  This function takes a single set of (P, phi_p, ecc). It then solves for the best-fit linear parameters and predicts the epoch astrometry at time t_ast_yr, and calculates the corresponding chi2. n_obs is length of the observations array.
the first element of chi2_array will be the chi2. The next 9 elements are the best-fit linear parameters for this (P, phi_p, ecc). The last 9 are are the uncertainties on the best-fit linear parameter (only accounting for the uncertainty associated with the matrix inversion.) */
void get_chi2_astrometry_and_uncertainties(int n_obs, double *t_ast_yr, double *psi, double *plx_factor, double *ast_obs, double *ast_err, double P, double phi_p, double ecc, double *chi2_array){
    double xtol = 1e-10;
    double Ei;
    double Mi;
    double X[n_obs];
    double Y[n_obs]; 
    double ivar[n_obs];
    
    gsl_matrix *M = gsl_matrix_calloc(n_obs, 9);
    gsl_matrix *Mt = gsl_matrix_alloc(9, n_obs);
    
    gsl_matrix *Cinv = gsl_matrix_calloc(n_obs, n_obs);
    gsl_matrix *MtCinvM = gsl_matrix_alloc(9, 9); // Result of Mt * Cinv * M
    gsl_vector *MtCinvY = gsl_vector_alloc(9); // Result of Mt * Cinv * ast_obs
    gsl_vector *mu = gsl_vector_alloc(9); // Solution vector
    gsl_vector *Lambda_pred = gsl_vector_alloc(n_obs); // Predicted lambda
    gsl_matrix *temp = gsl_matrix_alloc(9, n_obs); // Temporary matrix for intermediate calculations
    

    // Compute ivar and fill Cinv
    for (int j = 0; j < n_obs; j++) {
        ivar[j] = 1.0 / (ast_err[j] * ast_err[j]); // Compute inverse variance
        gsl_matrix_set(Cinv, j, j, ivar[j]); // Set diagonal elements of Cinv
    }    
    
    for(int j = 0; j < n_obs; j++){
        Mi = 2*PI*t_ast_yr[j]*365.25/P - phi_p;
        Ei = solve_Kepler_equation(Mi, ecc, xtol);
        
        X[j] = cos(Ei) - ecc;
        Y[j] = sqrt(1-ecc*ecc)*sin(Ei);
    }
    
    for (int j = 0; j < n_obs; j++) {
        gsl_matrix_set(M, j, 0, sin(psi[j]) );
        gsl_matrix_set(M, j, 1, t_ast_yr[j] * sin(psi[j]) );
        gsl_matrix_set(M, j, 2, cos(psi[j]) );
        gsl_matrix_set(M, j, 3, t_ast_yr[j] * cos(psi[j]));
        gsl_matrix_set(M, j, 4, plx_factor[j]);
        gsl_matrix_set(M, j, 5, X[j] * sin(psi[j]) );
        gsl_matrix_set(M, j, 6, Y[j] * sin(psi[j]) );
        gsl_matrix_set(M, j, 7, X[j] * cos(psi[j]));
        gsl_matrix_set(M, j, 8, Y[j] * cos(psi[j]));
    }
    gsl_matrix_transpose_memcpy(Mt, M);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Mt, Cinv, 0.0, temp);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, temp, M, 0.0, MtCinvM);
    
    gsl_vector *Cinv_ast_obs = gsl_vector_alloc(n_obs); // Allocate vector for the result of Cinv @ ast_obs
    gsl_vector_view ast_obs_view = gsl_vector_view_array(ast_obs, n_obs); // Create a vector view of ast_obs
    gsl_blas_dgemv(CblasNoTrans, 1.0, Cinv, &ast_obs_view.vector, 0.0, Cinv_ast_obs);
    gsl_blas_dgemv(CblasNoTrans, 1.0, Mt, Cinv_ast_obs, 0.0, MtCinvY);
    gsl_linalg_HH_solve(MtCinvM, MtCinvY, mu);
    gsl_blas_dgemv(CblasNoTrans, 1.0, M, mu, 0.0, Lambda_pred);
    
    
    gsl_matrix *cov_matrix = gsl_matrix_alloc(9, 9);
    gsl_permutation *perm = gsl_permutation_alloc(9);
    int signum;

    gsl_linalg_LU_decomp(MtCinvM, perm, &signum);
    gsl_linalg_LU_invert(MtCinvM, perm, cov_matrix);



    double chi2 = 0.0; 
    for (int j = 0; j < n_obs; j++) {
        double pred = gsl_vector_get(Lambda_pred, j);
        double obs = ast_obs[j];
        double err = ast_err[j];
        chi2 += ((pred - obs) * (pred - obs)) / (err * err);
    }
    chi2_array[0] = chi2;
    
    // now fill in the array with the best-fit parameters and uncertainties. 
    for (int j = 1; j < 10; j++) {
        chi2_array[j] = gsl_vector_get(mu, j-1);
    }
    
    for (int j = 10; j < 19; j++) {
        chi2_array[j] =  sqrt(gsl_matrix_get(cov_matrix, j-10, j-10));
    }

    // Cleanup
    gsl_matrix_free(M);
    gsl_matrix_free(Mt);
    gsl_matrix_free(Cinv);
    gsl_matrix_free(MtCinvM);
    gsl_vector_free(MtCinvY);
    gsl_vector_free(mu);
    gsl_vector_free(Lambda_pred);
    gsl_matrix_free(temp);
    gsl_vector_free(Cinv_ast_obs);
    gsl_matrix_free(cov_matrix);
    gsl_permutation_free(perm);
}




// Function to generate a random double between 0 and 1, needed for the optimization routine. 
double random_double() {
    return (double)rand() / (double)RAND_MAX;
}

void get_chi2_astrometry_reject_outliers(int n_obs, double *t_ast_yr, double *psi, double *plx_factor, double *ast_obs, double *ast_err, double P, double phi_p, double ecc, double *chi2_array);

double astfit_objective_logp(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, int n_obs, int reject_outlier, double* L, double* U, double y[3]) {
    if (y[0] < log(L[0]) || y[0] > log(U[0]) || y[1] < L[1] || y[1] > U[1] || y[2] < L[2] || y[2] > U[2]) {
        return INFINITY;
    }
    double chi2_array[10];
    double period = exp(y[0]);
    if (reject_outlier) {
        get_chi2_astrometry_reject_outliers(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err, period, y[1], y[2], chi2_array);
    } else {
        get_chi2_astrometry(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err, period, y[1], y[2], chi2_array);
    }
    return chi2_array[0];
}

void clamp_astfit_logp_vertex(double y[3], double* L, double* U) {
    double log_p_min = log(L[0]);
    double log_p_max = log(U[0]);
    if (y[0] < log_p_min) y[0] = log_p_min;
    if (y[0] > log_p_max) y[0] = log_p_max;
    if (y[1] < L[1]) y[1] = L[1];
    if (y[1] > U[1]) y[1] = U[1];
    if (y[2] < L[2]) y[2] = L[2];
    if (y[2] > U[2]) y[2] = U[2];
}

void sort_simplex4(double simplex[4][3], double f[4]) {
    for (int i = 0; i < 3; i++) {
        for (int j = i + 1; j < 4; j++) {
            if (f[j] < f[i]) {
                double tf = f[i];
                f[i] = f[j];
                f[j] = tf;
                for (int k = 0; k < 3; k++) {
                    double tx = simplex[i][k];
                    simplex[i][k] = simplex[j][k];
                    simplex[j][k] = tx;
                }
            }
        }
    }
}

void polish_astfit_start_nelder_mead_config(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, int n_obs, int reject_outlier, double* x, int max_iter, double fspread_tol, double xspread_tol) {
    int n = 3;
    int m = 4;
    double simplex[4][3];
    double f[4];
    double steps[3] = {0.05, 0.25, 0.08};
    double alpha = 1.0;
    double gamma = 2.0;
    double rho = 0.5;
    double sigma = 0.5;

    simplex[0][0] = log(x[0]);
    simplex[0][1] = x[1];
    simplex[0][2] = x[2];
    clamp_astfit_logp_vertex(simplex[0], L, U);

    for (int i = 1; i < m; i++) {
        for (int k = 0; k < n; k++) {
            simplex[i][k] = simplex[0][k];
        }
        simplex[i][i - 1] += steps[i - 1];
        clamp_astfit_logp_vertex(simplex[i], L, U);
        if (simplex[i][i - 1] == simplex[0][i - 1]) {
            simplex[i][i - 1] -= steps[i - 1];
            clamp_astfit_logp_vertex(simplex[i], L, U);
        }
    }

    for (int i = 0; i < m; i++) {
        f[i] = astfit_objective_logp(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_obs, reject_outlier, L, U, simplex[i]);
    }

    for (int iter = 0; iter < max_iter; iter++) {
        sort_simplex4(simplex, f);
        double fspread = fabs(f[3] - f[0]);
        double xspread = 0.0;
        for (int i = 1; i < m; i++) {
            for (int k = 0; k < n; k++) {
                double dx = fabs(simplex[i][k] - simplex[0][k]);
                if (dx > xspread) xspread = dx;
            }
        }
        if (fspread < fspread_tol && xspread < xspread_tol) {
            break;
        }

        double centroid[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < 3; i++) {
            for (int k = 0; k < n; k++) {
                centroid[k] += simplex[i][k] / 3.0;
            }
        }

        double xr[3];
        for (int k = 0; k < n; k++) {
            xr[k] = centroid[k] + alpha * (centroid[k] - simplex[3][k]);
        }
        clamp_astfit_logp_vertex(xr, L, U);
        double fr = astfit_objective_logp(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_obs, reject_outlier, L, U, xr);

        if (fr < f[0]) {
            double xe[3];
            for (int k = 0; k < n; k++) {
                xe[k] = centroid[k] + gamma * (xr[k] - centroid[k]);
            }
            clamp_astfit_logp_vertex(xe, L, U);
            double fe = astfit_objective_logp(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_obs, reject_outlier, L, U, xe);
            if (fe < fr) {
                for (int k = 0; k < n; k++) simplex[3][k] = xe[k];
                f[3] = fe;
            } else {
                for (int k = 0; k < n; k++) simplex[3][k] = xr[k];
                f[3] = fr;
            }
        } else if (fr < f[2]) {
            for (int k = 0; k < n; k++) simplex[3][k] = xr[k];
            f[3] = fr;
        } else {
            double xc[3];
            if (fr < f[3]) {
                for (int k = 0; k < n; k++) {
                    xc[k] = centroid[k] + rho * (xr[k] - centroid[k]);
                }
            } else {
                for (int k = 0; k < n; k++) {
                    xc[k] = centroid[k] + rho * (simplex[3][k] - centroid[k]);
                }
            }
            clamp_astfit_logp_vertex(xc, L, U);
            double fc = astfit_objective_logp(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_obs, reject_outlier, L, U, xc);
            if (fc < f[3]) {
                for (int k = 0; k < n; k++) simplex[3][k] = xc[k];
                f[3] = fc;
            } else {
                for (int i = 1; i < m; i++) {
                    for (int k = 0; k < n; k++) {
                        simplex[i][k] = simplex[0][k] + sigma * (simplex[i][k] - simplex[0][k]);
                    }
                    clamp_astfit_logp_vertex(simplex[i], L, U);
                    f[i] = astfit_objective_logp(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_obs, reject_outlier, L, U, simplex[i]);
                }
            }
        }
    }

    sort_simplex4(simplex, f);
    x[0] = exp(simplex[0][0]);
    x[1] = simplex[0][1];
    x[2] = simplex[0][2];
}

void polish_astfit_start_nelder_mead(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, int n_obs, int reject_outlier, double* x) {
    polish_astfit_start_nelder_mead_config(t_ast_yr, psi, plx_factor, ast_obs, ast_err, L, U, n_obs, reject_outlier, x, ASTFIT_NM_MAX_ITER, ASTFIT_NM_FSPREAD_TOL, ASTFIT_NM_XSPREAD_TOL);
}

void polish_astfit_start_nelder_mead_full(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, int n_obs, int reject_outlier, double* x) {
    polish_astfit_start_nelder_mead_config(t_ast_yr, psi, plx_factor, ast_obs, ast_err, L, U, n_obs, reject_outlier, x, ASTFIT_FULL_NM_MAX_ITER, ASTFIT_FULL_NM_FSPREAD_TOL, ASTFIT_FULL_NM_XSPREAD_TOL);
}

typedef struct {
    double chi2;
    double period;
    double phi_p;
    double ecc;
} AstfitCandidate;

int compare_astfit_candidate(const void* a, const void* b) {
    double da = ((const AstfitCandidate*)a)->chi2;
    double db = ((const AstfitCandidate*)b)->chi2;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

double astfit_phase_distance(double a, double b) {
    double d = fabs(a - b);
    while (d > 2.0 * PI) {
        d -= 2.0 * PI;
    }
    if (d > PI) {
        d = 2.0 * PI - d;
    }
    return d;
}

void insert_astfit_candidate(AstfitCandidate* candidates, int* n_candidates, int max_candidates, double chi2, double period, double phi_p, double ecc) {
    if (!isfinite(chi2)) {
        return;
    }
    if (*n_candidates >= max_candidates && chi2 >= candidates[*n_candidates - 1].chi2) {
        return;
    }

    int pos = *n_candidates;
    if (pos < max_candidates) {
        *n_candidates += 1;
    } else {
        pos = max_candidates - 1;
    }

    while (pos > 0 && chi2 < candidates[pos - 1].chi2) {
        candidates[pos] = candidates[pos - 1];
        pos--;
    }

    candidates[pos].chi2 = chi2;
    candidates[pos].period = period;
    candidates[pos].phi_p = phi_p;
    candidates[pos].ecc = ecc;
}

int astfit_period_is_distinct(double period, double* periods, int n_periods, double log_separation) {
    for (int i = 0; i < n_periods; i++) {
        if (fabs(log(period / periods[i])) < log_separation) {
            return 0;
        }
    }
    return 1;
}

int astfit_start_is_distinct(AstfitCandidate candidate, AstfitCandidate* starts, int n_starts) {
    for (int i = 0; i < n_starts; i++) {
        if (
            fabs(log(candidate.period / starts[i].period)) < 0.003 &&
            astfit_phase_distance(candidate.phi_p, starts[i].phi_p) < 0.05 &&
            fabs(candidate.ecc - starts[i].ecc) < 0.05
        ) {
            return 0;
        }
    }
    return 1;
}

double eval_astfit_candidate(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, int n_obs, int reject_outlier, double period, double phi_p, double ecc) {
    double chi2_array[10];
    if (reject_outlier) {
        get_chi2_astrometry_reject_outliers(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err, period, phi_p, ecc, chi2_array);
    } else {
        get_chi2_astrometry(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err, period, phi_p, ecc, chi2_array);
    }
    return chi2_array[0];
}

void scan_circular_period_grid(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, int n_obs, int reject_outlier, double p_min, double p_max, int n_period, int n_phase, AstfitCandidate* rows, int* n_rows) {
    double log_p_min = log(p_min);
    double log_p_max = log(p_max);
    for (int ip = 0; ip < n_period; ip++) {
        double frac = (n_period == 1) ? 0.0 : ((double)ip / (double)(n_period - 1));
        double period = exp(log_p_min + frac * (log_p_max - log_p_min));
        AstfitCandidate best;
        best.chi2 = INFINITY;
        best.period = period;
        best.phi_p = L[1];
        best.ecc = 0.0;
        for (int iph = 0; iph < n_phase; iph++) {
            double phi_p = L[1] + ((double)iph / (double)n_phase) * (U[1] - L[1]);
            double chi2 = eval_astfit_candidate(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_obs, reject_outlier, period, phi_p, 0.0);
            if (chi2 < best.chi2) {
                best.chi2 = chi2;
                best.phi_p = phi_p;
            }
        }
        rows[*n_rows] = best;
        *n_rows += 1;
    }
}

double linear_circular_periodogram_chi2_skip(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, int n_obs, double period, int skip_idx, int* worst_idx) {
    gsl_matrix* normal = gsl_matrix_calloc(9, 9);
    gsl_vector* rhs = gsl_vector_calloc(9);
    gsl_vector* mu = gsl_vector_alloc(9);
    gsl_permutation* perm = gsl_permutation_alloc(9);

    for (int j = 0; j < n_obs; j++) {
        if (j == skip_idx) {
            continue;
        }
        double s = sin(psi[j]);
        double c = cos(psi[j]);
        double phase = 2.0 * PI * t_ast_yr[j] * 365.25 / period;
        double cp = cos(phase);
        double sp = sin(phase);
        double col[9] = {
            s,
            t_ast_yr[j] * s,
            c,
            t_ast_yr[j] * c,
            plx_factor[j],
            cp * s,
            sp * s,
            cp * c,
            sp * c
        };
        double w = 1.0 / (ast_err[j] * ast_err[j]);
        for (int a = 0; a < 9; a++) {
            gsl_vector_set(rhs, a, gsl_vector_get(rhs, a) + w * col[a] * ast_obs[j]);
            for (int b = 0; b < 9; b++) {
                gsl_matrix_set(normal, a, b, gsl_matrix_get(normal, a, b) + w * col[a] * col[b]);
            }
        }
    }

    int signum;
    gsl_error_handler_t* old_handler = gsl_set_error_handler_off();
    int status = gsl_linalg_LU_decomp(normal, perm, &signum);
    if (status == GSL_SUCCESS) {
        status = gsl_linalg_LU_solve(normal, perm, rhs, mu);
    }
    gsl_set_error_handler(old_handler);

    double chi2 = INFINITY;
    if (status == GSL_SUCCESS) {
        chi2 = 0.0;
        double worst_chi2 = -1.0;
        int local_worst_idx = -1;
        for (int j = 0; j < n_obs; j++) {
            if (j == skip_idx) {
                continue;
            }
            double s = sin(psi[j]);
            double c = cos(psi[j]);
            double phase = 2.0 * PI * t_ast_yr[j] * 365.25 / period;
            double cp = cos(phase);
            double sp = sin(phase);
            double col[9] = {
                s,
                t_ast_yr[j] * s,
                c,
                t_ast_yr[j] * c,
                plx_factor[j],
                cp * s,
                sp * s,
                cp * c,
                sp * c
            };
            double pred = 0.0;
            for (int a = 0; a < 9; a++) {
                pred += col[a] * gsl_vector_get(mu, a);
            }
            double resid = (ast_obs[j] - pred) / ast_err[j];
            double chi2_term = resid * resid;
            chi2 += chi2_term;
            if (chi2_term > worst_chi2) {
                worst_chi2 = chi2_term;
                local_worst_idx = j;
            }
        }
        if (worst_idx != NULL) {
            *worst_idx = local_worst_idx;
        }
    }

    gsl_permutation_free(perm);
    gsl_vector_free(mu);
    gsl_vector_free(rhs);
    gsl_matrix_free(normal);
    return chi2;
}

double linear_circular_periodogram_chi2(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, int n_obs, double period, int reject_outlier) {
    if (reject_outlier && n_obs > 13) {
        int worst_idx = -1;
        double chi2 = linear_circular_periodogram_chi2_skip(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_obs, period, -1, &worst_idx);
        if (isfinite(chi2) && worst_idx >= 0) {
            return linear_circular_periodogram_chi2_skip(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_obs, period, worst_idx, NULL);
        }
        return chi2;
    }
    return linear_circular_periodogram_chi2_skip(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_obs, period, -1, NULL);
}

void scan_circular_periodogram_grid(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double p_min, double p_max, int n_obs, int reject_outlier, int n_period, AstfitCandidate* rows, int* n_rows) {
    double log_p_min = log(p_min);
    double log_p_max = log(p_max);
    for (int ip = 0; ip < n_period; ip++) {
        double frac = (n_period == 1) ? 0.0 : ((double)ip / (double)(n_period - 1));
        double period = exp(log_p_min + frac * (log_p_max - log_p_min));
        double chi2 = linear_circular_periodogram_chi2(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_obs, period, reject_outlier);
        rows[*n_rows].chi2 = chi2;
        rows[*n_rows].period = period;
        rows[*n_rows].phi_p = 0.0;
        rows[*n_rows].ecc = 0.0;
        *n_rows += 1;
    }
}

/* Deterministic C-only nonlinear search for (P, phi_p, e): broad/short circular period grids,
   eccentric phase scans around the best distinct periods, then Nelder-Mead polish of the best starts.
   results[0:4] = P, phi_p, e, chi2. */
void run_astfit_grid_multistart(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, double* results, int n_obs, int reject_outlier) {
    int n_period = ASTFIT_N_PERIOD;
    int n_phase = 8;
    int n_short_period = ASTFIT_N_SHORT_PERIOD;
    int n_ecc_phase = ASTFIT_N_ECC_PHASE;
    int max_periods = 32;
    int max_candidates = 256;
    int max_periods_keep = ASTFIT_MAX_PERIODS_KEEP;
    int max_starts = ASTFIT_MAX_STARTS;
    double short_period_max = fmin(U[0], 300.0);
    double ecc_grid[6] = {0.0, 0.2, 0.4, 0.6, 0.8, 0.95};

    int max_rows = n_period + ((short_period_max > L[0]) ? n_short_period : 0) + 8;
    AstfitCandidate* circular_rows = (AstfitCandidate*)malloc(max_rows * sizeof(AstfitCandidate));
    int n_rows = 0;
    scan_circular_period_grid(t_ast_yr, psi, plx_factor, ast_obs, ast_err, L, U, n_obs, reject_outlier, L[0], U[0], n_period, n_phase, circular_rows, &n_rows);
    if (short_period_max > L[0]) {
        scan_circular_period_grid(t_ast_yr, psi, plx_factor, ast_obs, ast_err, L, U, n_obs, reject_outlier, L[0], short_period_max, n_short_period, n_phase, circular_rows, &n_rows);
    }
    qsort(circular_rows, n_rows, sizeof(AstfitCandidate), compare_astfit_candidate);

    double* periods = (double*)malloc(max_periods * sizeof(double));
    int n_periods = 0;
    for (int i = 0; i < n_rows && n_periods < max_periods_keep; i++) {
        if (astfit_period_is_distinct(circular_rows[i].period, periods, n_periods, 0.025)) {
            periods[n_periods] = circular_rows[i].period;
            n_periods++;
        }
    }
    for (int i = 0; i < 8 && n_periods < max_periods; i++) {
        double period = U[0] * exp(-((double)i / 7.0));
        if (period >= L[0] && period <= U[0] && astfit_period_is_distinct(period, periods, n_periods, 0.01)) {
            periods[n_periods] = period;
            n_periods++;
        }
    }

    AstfitCandidate* candidates = (AstfitCandidate*)malloc(max_candidates * sizeof(AstfitCandidate));
    int n_candidates = 0;
    for (int i = 0; i < n_rows && i < 128; i++) {
        insert_astfit_candidate(candidates, &n_candidates, max_candidates, circular_rows[i].chi2, circular_rows[i].period, circular_rows[i].phi_p, circular_rows[i].ecc);
    }
    for (int ip = 0; ip < n_periods; ip++) {
        double period = periods[ip];
        for (int ie = 0; ie < 6; ie++) {
            double ecc = ecc_grid[ie];
            if (ecc < L[2] || ecc > U[2]) {
                continue;
            }
            for (int iph = 0; iph < n_ecc_phase; iph++) {
                double phi_p = L[1] + ((double)iph / (double)n_ecc_phase) * (U[1] - L[1]);
                double chi2 = eval_astfit_candidate(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_obs, reject_outlier, period, phi_p, ecc);
                insert_astfit_candidate(candidates, &n_candidates, max_candidates, chi2, period, phi_p, ecc);
            }
        }
    }

    AstfitCandidate* starts = (AstfitCandidate*)malloc(max_starts * sizeof(AstfitCandidate));
    int n_starts = 0;
    for (int i = 0; i < n_candidates && n_starts < max_starts; i++) {
        if (astfit_start_is_distinct(candidates[i], starts, n_starts)) {
            starts[n_starts] = candidates[i];
            n_starts++;
        }
    }

    double best_chi2 = INFINITY;
    double best_x[3] = {(U[0] + L[0]) / 2.0, (U[1] + L[1]) / 2.0, (U[2] + L[2]) / 2.0};
    for (int i = 0; i < n_starts; i++) {
        double x[3] = {starts[i].period, starts[i].phi_p, starts[i].ecc};
        polish_astfit_start_nelder_mead(t_ast_yr, psi, plx_factor, ast_obs, ast_err, L, U, n_obs, reject_outlier, x);
        double chi2 = eval_astfit_candidate(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_obs, reject_outlier, x[0], x[1], x[2]);
        if (chi2 < best_chi2) {
            best_chi2 = chi2;
            best_x[0] = x[0];
            best_x[1] = x[1];
            best_x[2] = x[2];
        }
    }

    results[0] = best_x[0];
    results[1] = best_x[1];
    results[2] = best_x[2];
    results[3] = best_chi2;

    free(circular_rows);
    free(periods);
    free(candidates);
    free(starts);
}

/* Deterministic C-only nonlinear search seeded by a linear circular-orbit periodogram.
   The circular phase is absorbed by the linear Thiele-Innes coefficients, so this scores
   each trial period once, then keeps the existing eccentric phase scan and Nelder-Mead polish.
   results[0:4] = P, phi_p, e, chi2. */
/* Original implementation of the periodogram multistart search, with every chi2 evaluated through
   GSL (get_chi2_astrometry / get_chi2_astrometry_reject_outliers / linear_circular_periodogram_chi2).
   Kept for reference and testing; run_astfit_periodogram_multistart_config() below runs the same
   search with the faster objective. */
void run_astfit_periodogram_multistart_config_gsl(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, double* results, int n_obs, int reject_outlier, int n_period, int n_short_period, int n_ecc_phase, int max_periods_keep, int max_starts, int use_full_nm) {
    int max_periods = 32;
    int max_candidates = 256;
    double short_period_max = fmin(U[0], 300.0);
    double ecc_grid[6] = {0.0, 0.2, 0.4, 0.6, 0.8, 0.95};

    int max_rows = n_period + ((short_period_max > L[0]) ? n_short_period : 0) + 8;
    AstfitCandidate* periodogram_rows = (AstfitCandidate*)malloc(max_rows * sizeof(AstfitCandidate));
    int n_rows = 0;
    scan_circular_periodogram_grid(t_ast_yr, psi, plx_factor, ast_obs, ast_err, L[0], U[0], n_obs, reject_outlier, n_period, periodogram_rows, &n_rows);
    if (short_period_max > L[0]) {
        scan_circular_periodogram_grid(t_ast_yr, psi, plx_factor, ast_obs, ast_err, L[0], short_period_max, n_obs, reject_outlier, n_short_period, periodogram_rows, &n_rows);
    }
    qsort(periodogram_rows, n_rows, sizeof(AstfitCandidate), compare_astfit_candidate);

    double* periods = (double*)malloc(max_periods * sizeof(double));
    int n_periods = 0;
    for (int i = 0; i < n_rows && n_periods < max_periods_keep; i++) {
        if (astfit_period_is_distinct(periodogram_rows[i].period, periods, n_periods, 0.025)) {
            periods[n_periods] = periodogram_rows[i].period;
            n_periods++;
        }
    }
    for (int i = 0; i < 8 && n_periods < max_periods; i++) {
        double period = U[0] * exp(-((double)i / 7.0));
        if (period >= L[0] && period <= U[0] && astfit_period_is_distinct(period, periods, n_periods, 0.01)) {
            periods[n_periods] = period;
            n_periods++;
        }
    }

    AstfitCandidate* candidates = (AstfitCandidate*)malloc(max_candidates * sizeof(AstfitCandidate));
    int n_candidates = 0;
    for (int i = 0; i < n_rows && i < 128; i++) {
        insert_astfit_candidate(candidates, &n_candidates, max_candidates, periodogram_rows[i].chi2, periodogram_rows[i].period, periodogram_rows[i].phi_p, periodogram_rows[i].ecc);
    }
    for (int ip = 0; ip < n_periods; ip++) {
        double period = periods[ip];
        for (int ie = 0; ie < 6; ie++) {
            double ecc = ecc_grid[ie];
            if (ecc < L[2] || ecc > U[2]) {
                continue;
            }
            for (int iph = 0; iph < n_ecc_phase; iph++) {
                double phi_p = L[1] + ((double)iph / (double)n_ecc_phase) * (U[1] - L[1]);
                double chi2 = eval_astfit_candidate(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_obs, reject_outlier, period, phi_p, ecc);
                insert_astfit_candidate(candidates, &n_candidates, max_candidates, chi2, period, phi_p, ecc);
            }
        }
    }

    AstfitCandidate* starts = (AstfitCandidate*)malloc(max_starts * sizeof(AstfitCandidate));
    int n_starts = 0;
    for (int i = 0; i < n_candidates && n_starts < max_starts; i++) {
        if (astfit_start_is_distinct(candidates[i], starts, n_starts)) {
            starts[n_starts] = candidates[i];
            n_starts++;
        }
    }

    double best_chi2 = INFINITY;
    double best_x[3] = {(U[0] + L[0]) / 2.0, (U[1] + L[1]) / 2.0, (U[2] + L[2]) / 2.0};
    for (int i = 0; i < n_starts; i++) {
        double x[3] = {starts[i].period, starts[i].phi_p, starts[i].ecc};
        if (use_full_nm) {
            polish_astfit_start_nelder_mead_full(t_ast_yr, psi, plx_factor, ast_obs, ast_err, L, U, n_obs, reject_outlier, x);
        } else {
            polish_astfit_start_nelder_mead(t_ast_yr, psi, plx_factor, ast_obs, ast_err, L, U, n_obs, reject_outlier, x);
        }
        double chi2 = eval_astfit_candidate(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_obs, reject_outlier, x[0], x[1], x[2]);
        if (chi2 < best_chi2) {
            best_chi2 = chi2;
            best_x[0] = x[0];
            best_x[1] = x[1];
            best_x[2] = x[2];
        }
    }

    results[0] = best_x[0];
    results[1] = best_x[1];
    results[2] = best_x[2];
    results[3] = best_chi2;

    free(periodogram_rows);
    free(periods);
    free(candidates);
    free(starts);
}

void run_astfit_periodogram_multistart_gsl(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, double* results, int n_obs, int reject_outlier) {
    run_astfit_periodogram_multistart_config_gsl(t_ast_yr, psi, plx_factor, ast_obs, ast_err, L, U, results, n_obs, reject_outlier, ASTFIT_N_PERIOD, ASTFIT_N_SHORT_PERIOD, ASTFIT_N_ECC_PHASE, ASTFIT_MAX_PERIODS_KEEP, ASTFIT_MAX_STARTS, 0);
}

/* ---------------------------------------------------------------------------------------------
   Fast evaluation of chi2(P, phi_p, ecc) for the optimizers.

   The 9-parameter linear model has 5 columns (offsets, proper motions, parallax) that do not
   depend on (P, phi_p, ecc). AstfitWorkspace whitens them and orthonormalizes them once per source,
   and stores the whitened data with those columns projected out. Each chi2 evaluation then only
   builds the 4 Thiele-Innes columns, orthogonalizes them against the fixed ones (classical
   Gram-Schmidt with one re-orthogonalization pass), and returns the squared norm of the updated
   residual. There are no GSL calls or allocations per evaluation. Orbital columns that are
   numerically degenerate (e.g. P >> the time baseline) are dropped, which gives the
   minimum-norm projection rather than an ill-conditioned 9x9 solve.

   With reject_outlier, the epoch with the largest residual is removed using the exact
   delete-one identity chi2_(-i) = chi2 - r_i^2 / (1 - h_ii), where h_ii is the leverage of epoch
   i. This equals refitting without that epoch, as in get_chi2_astrometry_reject_outliers().

   The chi2 agrees with get_chi2_astrometry() / get_chi2_astrometry_reject_outliers() to ~1e-12
   (relative) and is several times faster.
   --------------------------------------------------------------------------------------------- */

typedef struct {
    int n, reject, nfix;
    double *a;      /* 2*PI*t*365.25 */
    double *sw, *cw;/* sin(psi)/err, cos(psi)/err */
    double *Q;      /* nfix*n orthonormal fixed columns */
    double *r0;     /* whitened data with the fixed columns projected out */
    double *h0;     /* leverage of the fixed columns */
    double *Qo;     /* 4*n scratch: orthonormal orbital columns */
    double *res;    /* n scratch: residual */
    double tspan;   /* time baseline in days */
    double *t, *psi, *pf, *y, *err;
} AstfitWorkspace;

static double ws_dot(const double* x, const double* y, int n) {
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int j = 0;
    for (; j + 3 < n; j += 4) {
        s0 += x[j] * y[j];
        s1 += x[j + 1] * y[j + 1];
        s2 += x[j + 2] * y[j + 2];
        s3 += x[j + 3] * y[j + 3];
    }
    for (; j < n; j++) s0 += x[j] * y[j];
    return (s0 + s1) + (s2 + s3);
}

/* Orthogonalize v against nq orthonormal columns stored in Q (classical Gram-Schmidt, repeated
   once if there was heavy cancellation). Returns the norm of the result relative to the input. */
static double ws_orth(double* v, const double* Q, int nq, int n) {
    double nrm0 = sqrt(ws_dot(v, v, n));
    if (nrm0 == 0.0 || nq == 0) return nrm0 == 0.0 ? 0.0 : 1.0;
    double nrm = nrm0;
    for (int pass = 0; pass < 2; pass++) {
        double d[16] = {0};
        for (int k0 = 0; k0 < nq; k0 += 4) {
            int m = nq - k0 < 4 ? nq - k0 : 4;
            const double *q0 = Q + (long)k0 * n, *q1 = q0 + n, *q2 = q1 + n, *q3 = q2 + n;
            double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            if (m == 4) for (int j = 0; j < n; j++) { double x = v[j]; s0 += q0[j] * x; s1 += q1[j] * x; s2 += q2[j] * x; s3 += q3[j] * x; }
            else if (m == 3) for (int j = 0; j < n; j++) { double x = v[j]; s0 += q0[j] * x; s1 += q1[j] * x; s2 += q2[j] * x; }
            else if (m == 2) for (int j = 0; j < n; j++) { double x = v[j]; s0 += q0[j] * x; s1 += q1[j] * x; }
            else for (int j = 0; j < n; j++) s0 += q0[j] * v[j];
            d[k0] = s0; d[k0 + 1] = s1; d[k0 + 2] = s2; d[k0 + 3] = s3;
        }
        double a0 = 0, a1 = 0;
        for (int j = 0; j < n; j++) {
            double x = v[j];
            for (int k = 0; k < nq; k++) x -= d[k] * Q[(long)k * n + j];
            v[j] = x;
            if (j & 1) a1 += x * x; else a0 += x * x;
        }
        double nn = sqrt(a0 + a1);
        if (nn > 0.5 * nrm) { nrm = nn; break; }
        nrm = nn;
    }
    return nrm / nrm0;
}

/* Fill w for one source. Epoch `skip` (if >= 0) gets zero weight. */
static void ws_build(AstfitWorkspace* w, double* t, double* psi, double* pf, double* y, double* err, int n, int reject, int skip) {
    w->n = n; w->reject = reject;
    w->t = t; w->psi = psi; w->pf = pf; w->y = y; w->err = err;
    double* buf = (double*)malloc(sizeof(double) * (long)n * 15);
    w->a = buf; w->sw = buf + n; w->cw = buf + 2 * n; w->Q = buf + 3 * n;
    w->r0 = buf + 8 * n; w->h0 = buf + 9 * n; w->Qo = buf + 10 * n; w->res = buf + 14 * n;
    double tmin = t[0], tmax = t[0];
    for (int j = 0; j < n; j++) {
        double iw = (j == skip) ? 0.0 : 1.0 / err[j];
        w->a[j] = 2 * PI * t[j] * 365.25;
        w->sw[j] = sin(psi[j]) * iw;
        w->cw[j] = cos(psi[j]) * iw;
        w->r0[j] = y[j] * iw;
        if (t[j] < tmin) tmin = t[j];
        if (t[j] > tmax) tmax = t[j];
    }
    w->tspan = (tmax - tmin) * 365.25;
    int nq = 0;
    for (int c = 0; c < 5; c++) {
        double* v = w->Q + (long)nq * n;
        for (int j = 0; j < n; j++) {
            switch (c) {
                case 0: v[j] = w->sw[j]; break;
                case 1: v[j] = t[j] * w->sw[j]; break;
                case 2: v[j] = w->cw[j]; break;
                case 3: v[j] = t[j] * w->cw[j]; break;
                default: v[j] = (j == skip) ? 0.0 : pf[j] / err[j];
            }
        }
        if (ws_orth(v, w->Q, nq, n) > 1e-10) {
            double s = 1.0 / sqrt(ws_dot(v, v, n));
            for (int j = 0; j < n; j++) v[j] *= s;
            nq++;
        }
    }
    w->nfix = nq;
    ws_orth(w->r0, w->Q, nq, n);
    for (int j = 0; j < n; j++) {
        double h = 0.0;
        for (int k = 0; k < nq; k++) h += w->Q[(long)k * n + j] * w->Q[(long)k * n + j];
        w->h0[j] = h;
    }
}

static AstfitWorkspace* astfit_workspace_create(double* t, double* psi, double* pf, double* y, double* err, int n, int reject) {
    AstfitWorkspace* w = (AstfitWorkspace*)malloc(sizeof(AstfitWorkspace));
    ws_build(w, t, psi, pf, y, err, n, reject, -1);
    return w;
}

static void astfit_workspace_free(AstfitWorkspace* w) {
    if (w) { free(w->a); free(w); }
}

/* Kepler solve returning cos(E), sin(E). Same starter and third-order iteration as
   solve_Kepler_equation(). Because the update is third order, once a correction is below 1e-4
   the updated E is converged to machine precision (max |dE| < 2e-14 for e <= 0.99), so we stop
   there. cos/sin of the final E are obtained by rotating the last evaluated pair by the small
   final correction (5th-order Taylor series, error < 1e-26) instead of calling cos/sin again. */
static inline void kepler_cos_sin(double Mi, double ecc, double* cE, double* sE) {
    double M = Mi - (2 * PI) * floor(Mi * (1.0 / (2 * PI)));
    if (M < 0.0) M += 2 * PI;
    if (M >= 2 * PI) M -= 2 * PI;
    if (ecc == 0.0) { *cE = cos(M); *sE = sin(M); return; }
    double E = KeplerStart3(ecc, M);
    for (int it = 0; it < 100; it++) {
        double c = cos(E), s = sin(E);
        double t2 = -1 + ecc * c, t4 = ecc * s, t5 = -E + t4 + M;
        double t6 = t5 / (0.5 * t5 * t4 / t2 + t2);
        double d = t5 / ((0.5 * s - (1.0 / 6.0) * c * t6) * ecc * t6 + t2);
        if (fabs(d) < 1e-4 || it == 99) {
            double d2 = d * d;
            double cd = 1.0 - d2 * (0.5 - d2 / 24.0), sd = d * (1.0 - d2 * (1.0 / 6.0 - d2 / 120.0));
            *cE = c * cd + s * sd;   /* cos(E - d) */
            *sE = s * cd - c * sd;   /* sin(E - d) */
            return;
        }
        E -= d;
    }
}

/* chi2 without outlier rejection. Leaves the residual in w->res and the nor orthonormal
   orbital columns in w->Qo. */
static double ws_chi2_all(AstfitWorkspace* w, double P, double phi_p, double ecc, int* nor_out) {
    int n = w->n;
    double sq = sqrt(1 - ecc * ecc);
    double* B = w->Qo;
    for (int j = 0; j < n; j++) {
        double cE, sE;
        kepler_cos_sin(w->a[j] / P - phi_p, ecc, &cE, &sE);
        double X = cE - ecc, Y = sq * sE;
        B[j] = X * w->sw[j];
        B[n + j] = Y * w->sw[j];
        B[2 * n + j] = X * w->cw[j];
        B[3 * n + j] = Y * w->cw[j];
    }
    memcpy(w->res, w->r0, sizeof(double) * n);
    int nor = 0;
    for (int c = 0; c < 4; c++) {
        double* v = B + (long)c * n;
        double rel = ws_orth(v, w->Q, w->nfix, n);
        if (rel > 1e-10 && nor > 0) rel *= ws_orth(v, B, nor, n);
        if (!(rel > 1e-10)) continue;
        double s = 1.0 / sqrt(ws_dot(v, v, n));
        for (int j = 0; j < n; j++) v[j] *= s;
        if (nor != c) memcpy(B + (long)nor * n, v, sizeof(double) * n);
        double proj = ws_dot(B + (long)nor * n, w->res, n);
        for (int j = 0; j < n; j++) w->res[j] -= proj * B[(long)nor * n + j];
        nor++;
    }
    *nor_out = nor;
    return ws_dot(w->res, w->res, n);
}

/* ------------- faster chi2 evaluation: trig-free Kepler solver and a 4x4 Schur-complement solve -------------

   astfit_workspace_chi2() first tries ws_chi2_schur(), which gives the same chi2 as ws_chi2_all()
   (to ~1e-13 relative without outlier rejection, ~1e-10 with it) but is several times faster:
   1. cos E and sin E come from orvara's trig-free Kepler solver (kepler_cos_sin_orvara below), with
      its starter table built once per chi2 call (one eccentricity).
   2. The 4 orbital columns are not orthogonalized one by one. A single pass over the epochs
      accumulates their overlaps with the orthonormal fixed columns, their Gram matrix and their
      overlap with the projected data. The 4x4 Schur complement is solved by Cholesky, and a
      second pass forms the residual explicitly.
   ws_chi2_schur() returns 0, and the caller uses ws_chi2_all(), if any orbital column keeps less
   than 1e-2 of its norm after projecting out the fixed and preceding orbital columns (where the
   normal equations lose accuracy; mostly at the longest periods), or if the rejected epoch
   has leverage ~1. */
/* Trig-free Kepler solver from orvara (Brandt et al. 2021, AJ 162, 186, Sec. 3.1, orvara/calcEA.h), which
   follows Raposo-Pulido & Pelaez (2017, MNRAS 467, 1702); ported via the JAX version in harv
   (harv/kepler/_kepler_solver.py). orvara is BSD-3 licensed; see THIRD_PARTY_NOTICES.
   1. reduce M to [0, pi];
   2. start from a piecewise quintic in M (12 pieces, matching E, E' and E'' at E = k pi/12; built once per
      eccentricity by kepler_orvara_table), or, where 2M + 1 - e <= 0.2, from RPP's series in sqrt(1 - e);
   3. sin E and cos E from a 15th-order Taylor series on an argument in [0, pi/4] plus a square root;
   4. one third-order (Householder) step, and (sin E, cos E) rotated by it to third order.
   No trig calls per epoch; |cos E|, |sin E| errors <~ 1e-14 for e <= 0.99 (5e-14 at e = 0.99). */
#define KEPLER_ORVARA_PI 3.14159265358979323846
typedef struct { double e, bounds[13], c[12][6]; } KeplerOrvaraTable;

static void kepler_orvara_table(KeplerOrvaraTable* T, double e) {
    double En[13], s[13], co[13], d1[13], d2[13];
    T->e = e;
    for (int k = 0; k < 13; k++) {
        En[k] = k * KEPLER_ORVARA_PI / 12; s[k] = sin(En[k]); co[k] = cos(En[k]);
        T->bounds[k] = En[k] - e * s[k];
        d1[k] = 1 / (1 - e * co[k]);                       /* dE/dM */
        d2[k] = -0.5 * e * s[k] * d1[k] * d1[k] * d1[k];    /* (d^2E/dM^2) / 2 */
    }
    for (int k = 0; k < 12; k++) {
        double idx = 1 / (T->bounds[k + 1] - T->bounds[k]);
        double a1 = d1[k], a2 = d2[k];
        double b0 = idx * (-a2 - idx * (a1 - idx * KEPLER_ORVARA_PI / 12));
        double b1 = idx * (-2 * a2 - idx * (a1 - d1[k + 1]));
        double b2 = idx * (d2[k + 1] - a2);
        T->c[k][0] = En[k]; T->c[k][1] = a1; T->c[k][2] = a2;
        T->c[k][3] = b2 - 4 * b1 + 10 * b0;
        T->c[k][4] = (-2 * b2 + 7 * b1 - 15 * b0) * idx;
        T->c[k][5] = (b2 - 3 * b1 + 6 * b0) * idx * idx;
    }
}

/* RPP (2017) Eq. 19: series start for E when e -> 1 and M -> 0 */
static inline double kepler_orvara_corner(double M, double e) {
    double ome = 1 - e, sq = sqrt(ome), chi = M / (sq * ome);
    double lam = sqrt(8 + 9 * chi * chi), s = cbrt(lam + 3 * chi), s2 = s * s;
    double sigma = 6 * chi / (2 + s2 + 4 / s2), sig2 = sigma * sigma, den = sig2 + 2;
    double poly = sig2 * (sig2 * (sig2 + 25) + 340) + 840;
    double series = (sig2 + 20) / (60 * den) + sig2 * ome * poly / (1400 * den * den * den);
    return sigma * (1 + sig2 * ome * series) * sq;
}

/* 15th-order Taylor series for sin x; float64-accurate for |x| <= pi/4 */
static inline double kepler_orvara_short_sin(double x) {
    double x2 = x * x;
    double p = 1.0 / 6227020800.0 - x2 * (1.0 / 1307674368000.0);
    p = 1.0 / 39916800.0 - x2 * p;
    p = 1.0 / 362880.0 - x2 * p;
    p = 1.0 / 5040.0 - x2 * p;
    p = 1.0 / 120.0 - x2 * p;
    p = 1.0 / 6.0 - x2 * p;
    return x * (1 - x2 * p);
}

/* cos E, sin E for mean anomaly Mi (any value), eccentricity T->e */
static inline void kepler_cos_sin_orvara(const KeplerOrvaraTable* T, double Mi, double* cE, double* sE) {
    const double pi = KEPLER_ORVARA_PI, e = T->e;
    double M = Mi - 2 * pi * floor(Mi * (1.0 / (2 * pi)));
    int lower = M > pi;
    if (lower) M = 2 * pi - M;
    double E;
    if (2 * M + (1 - e) <= 0.2) E = kepler_orvara_corner(M, e);
    else {
        int k = 0;
        while (k < 11 && T->bounds[k + 1] < M) k++;
        const double* a = T->c[k];
        double x = M - T->bounds[k];
        E = a[0] + x * (a[1] + x * (a[2] + x * (a[3] + x * (a[4] + x * a[5]))));
    }
    int near0 = E <= pi / 4, nearpi = E >= 3 * pi / 4;
    double s = kepler_orvara_short_sin(near0 ? E : (nearpi ? pi - E : pi / 2 - E));
    double c = sqrt(fmax(1 - s * s, 0.0));
    double sinE = (near0 || nearpi) ? s : c;
    double cosE = near0 ? c : (nearpi ? -c : s);
    /* one Householder step on f(E) = E - e sin E - M */
    double num = M - E + e * sinE, den = 1 - e * cosE, es = e * sinE, ec = e * cosE;
    double dE = num * (den * den + 0.5 * num * es) / (den * den * den + num * (den * es + num * ec / 6));
    /* rotate (sin E, cos E) by dE, to third order in dE */
    double q = dE * dE / 6, cd = 1 - 3 * q, sd = dE * (1 - q);
    double s2 = sinE * cd + cosE * sd, c2 = cosE * cd - sinE * sd;
    *sE = lower ? -s2 : s2;
    *cE = c2;
}

/* Everything one Schur-complement chi2 evaluation produces. B (4n) and r (n) are required; cE and sE (n each) may be
   NULL. The Levenberg-Marquardt normal equations (lm_state_normal) are computed from these. */
typedef struct { int ok; double P, phi, e, beta[4], G[5][4], Lc[4][4]; double *cE, *sE, *B, *r; } SchurState;

/* The Schur-complement solve at (P, phi_p, ecc): fills s (orbital columns B, G = Q^T B, the Cholesky factor Lc of
   S = B^T B - G^T G, the coefficients beta, the residual r and, if s->cE is not NULL, cos E and sin E), and returns
   chi2 = |r|^2 and the largest squared residual and its epoch. Returns 0 if S is too ill-conditioned (see above). */
static int ws_schur_solve(AstfitWorkspace* w, double P, double phi_p, double ecc, SchurState* s, double* chi2_out, int* worst_out, double* rw_out) {
    int n = w->n, nf = w->nfix;
    double sq = sqrt(1 - ecc * ecc);
    double* B = s->B;
    double *cEs = s->cE, *sEs = s->sE;
    double (*G)[4] = s->G, H[4][4] = {{0}}, b[4] = {0};
    memset(s->G, 0, sizeof s->G);
    s->P = P; s->phi = phi_p; s->e = ecc; s->ok = 0;
    KeplerOrvaraTable kt;
    if (ecc > 0.0) kepler_orvara_table(&kt, ecc);
    for (int j = 0; j < n; j++) {
        double cE, sE, M = w->a[j] / P - phi_p;
        if (ecc > 0.0) kepler_cos_sin_orvara(&kt, M, &cE, &sE);
        else { cE = cos(M); sE = sin(M); }
        if (cEs) { cEs[j] = cE; sEs[j] = sE; }
        double X = cE - ecc, Y = sq * sE;
        double v0 = X * w->sw[j], v1 = Y * w->sw[j], v2 = X * w->cw[j], v3 = Y * w->cw[j];
        B[j] = v0; B[n + j] = v1; B[2 * n + j] = v2; B[3 * n + j] = v3;
        for (int k = 0; k < nf; k++) {
            double q = w->Q[(long)k * n + j];
            G[k][0] += q * v0; G[k][1] += q * v1; G[k][2] += q * v2; G[k][3] += q * v3;
        }
        H[0][0] += v0 * v0; H[1][0] += v1 * v0; H[1][1] += v1 * v1;
        H[2][0] += v2 * v0; H[2][1] += v2 * v1; H[2][2] += v2 * v2;
        H[3][0] += v3 * v0; H[3][1] += v3 * v1; H[3][2] += v3 * v2; H[3][3] += v3 * v3;
        double r = w->r0[j];
        b[0] += v0 * r; b[1] += v1 * r; b[2] += v2 * r; b[3] += v3 * r;
    }
    /* Cholesky of S = H - G^T G */
    double (*Lc)[4] = s->Lc;
    for (int c = 0; c < 4; c++) {
        for (int d = 0; d <= c; d++) {
            double v = H[c][d];
            for (int k = 0; k < nf; k++) v -= G[k][c] * G[k][d];
            for (int m = 0; m < d; m++) v -= Lc[c][m] * Lc[d][m];
            if (d == c) {
                if (!(H[c][c] > 0) || !(v > 1e-4 * H[c][c])) return 0;
                Lc[c][c] = sqrt(v);
            } else Lc[c][d] = v / Lc[d][d];
        }
    }
    double y[4], *beta = s->beta;
    for (int c = 0; c < 4; c++) {
        double v = b[c];
        for (int m = 0; m < c; m++) v -= Lc[c][m] * y[m];
        y[c] = v / Lc[c][c];
    }
    for (int c = 3; c >= 0; c--) {
        double v = y[c];
        for (int m = c + 1; m < 4; m++) v -= Lc[m][c] * beta[m];
        beta[c] = v / Lc[c][c];
    }
    /* residual = r0 - (B - Q G) beta */
    double gam[5] = {0};
    for (int k = 0; k < nf; k++) gam[k] = G[k][0] * beta[0] + G[k][1] * beta[1] + G[k][2] * beta[2] + G[k][3] * beta[3];
    double chi2 = 0.0, rw = 0.0;
    int worst = 0;
    double* res = s->r;
    for (int j = 0; j < n; j++) {
        double r = w->r0[j] - beta[0] * B[j] - beta[1] * B[n + j] - beta[2] * B[2 * n + j] - beta[3] * B[3 * n + j];
        for (int k = 0; k < nf; k++) r += gam[k] * w->Q[(long)k * n + j];
        res[j] = r;
        chi2 += r * r;
        if (r * r > rw) { rw = r * r; worst = j; }
    }
    s->ok = 1;
    *chi2_out = chi2; *worst_out = worst; *rw_out = rw;
    return 1;
}

static int ws_chi2_schur(AstfitWorkspace* w, double P, double phi_p, double ecc, double* chi2_out) {
    int n = w->n, nf = w->nfix, worst;
    double chi2, rw;
    SchurState st; st.cE = NULL; st.sE = NULL; st.B = w->Qo; st.r = w->res;
    if (!ws_schur_solve(w, P, phi_p, ecc, &st, &chi2, &worst, &rw)) return 0;
    double* B = st.B;
    double (*G)[4] = st.G, (*Lc)[4] = st.Lc;
    if (!w->reject) { *chi2_out = chi2; return 1; }
    /* delete-one identity, with leverage h0 + v^T S^-1 v for the projected orbital row v */
    double h = w->h0[worst], z[4];
    for (int c = 0; c < 4; c++) {
        double s = B[(long)c * n + worst];
        for (int k = 0; k < nf; k++) s -= w->Q[(long)k * n + worst] * G[k][c];
        for (int m = 0; m < c; m++) s -= Lc[c][m] * z[m];
        z[c] = s / Lc[c][c];
        h += z[c] * z[c];
    }
    double denom = 1.0 - h;
    if (!(denom > 1e-10 && isfinite(denom))) return 0;
    *chi2_out = chi2 - rw / denom;
    return 1;
}

/* chi2 at (P, phi_p, ecc), with the worst epoch removed if w->reject. */
static double astfit_workspace_chi2(AstfitWorkspace* w, double P, double phi_p, double ecc) {
    double fast;
    if (ws_chi2_schur(w, P, phi_p, ecc, &fast)) return fast;
    int nor;
    double chi2 = ws_chi2_all(w, P, phi_p, ecc, &nor);
    if (!w->reject) return chi2;
    int n = w->n, worst = 0;
    double rw = 0.0;
    for (int j = 0; j < n; j++) {
        double r2 = w->res[j] * w->res[j];
        if (r2 > rw) { rw = r2; worst = j; }
    }
    double h = w->h0[worst];
    for (int k = 0; k < nor; k++) h += w->Qo[(long)k * n + worst] * w->Qo[(long)k * n + worst];
    double denom = 1.0 - h;
    if (denom > 1e-10 && isfinite(denom)) return chi2 - rw / denom;
    /* epoch with leverage ~1: refit explicitly without it */
    AstfitWorkspace m;
    ws_build(&m, w->t, w->psi, w->pf, w->y, w->err, n, 0, worst);
    double c2 = ws_chi2_all(&m, P, phi_p, ecc, &nor);
    free(m.a);
    return c2;
}

/* Circular periodogram value at P (equivalent to linear_circular_periodogram_chi2(); the
   phase is absorbed by the linear coefficients, and rejection is only applied for n > 13). */
static double astfit_workspace_periodogram(AstfitWorkspace* w, double P) {
    if (w->reject && w->n <= 13) { int nor; return ws_chi2_all(w, P, 0.0, 0.0, &nor); }
    return astfit_workspace_chi2(w, P, 0.0, 0.0);
}

/* Same as get_chi2_astrometry()[0] (or get_chi2_astrometry_reject_outliers()[0]), via the fast path. */
double get_chi2_astrometry_fast(int n_obs, double *t_ast_yr, double *psi, double *plx_factor, double *ast_obs, double *ast_err, double P, double phi_p, double ecc, int reject_outlier) {
    AstfitWorkspace* w = astfit_workspace_create(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_obs, reject_outlier);
    double chi2 = astfit_workspace_chi2(w, P, phi_p, ecc);
    astfit_workspace_free(w);
    return chi2;
}

/* chi2 at n_eval points (P[i], phi_p[i], ecc[i]) evaluated in order on one workspace, as the
   optimizers do. Exposed for testing. */
void get_chi2_astrometry_fast_sequence(int n_obs, double *t_ast_yr, double *psi, double *plx_factor, double *ast_obs, double *ast_err, int n_eval, double *P, double *phi_p, double *ecc, int reject_outlier, double *chi2_out) {
    AstfitWorkspace* w = astfit_workspace_create(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_obs, reject_outlier);
    for (int i = 0; i < n_eval; i++) chi2_out[i] = astfit_workspace_chi2(w, P[i], phi_p[i], ecc[i]);
    astfit_workspace_free(w);
}

/* ------------- Nelder-Mead on the fast objective (same logic as polish_astfit_start_nelder_mead_config) ------------- */

typedef double (*astfit_ws_objective)(AstfitWorkspace*, double*, double*, double*);
typedef void (*astfit_ws_clamp)(double*, double*, double*);

static double ws_objective_logp(AstfitWorkspace* w, double* L, double* U, double* y) {
    if (y[0] < log(L[0]) || y[0] > log(U[0]) || y[1] < L[1] || y[1] > U[1] || y[2] < L[2] || y[2] > U[2]) return INFINITY;
    return astfit_workspace_chi2(w, exp(y[0]), y[1], y[2]);
}

static void ws_nelder_mead(AstfitWorkspace* w, double* L, double* U, astfit_ws_objective obj, astfit_ws_clamp clamp,
                           double simplex[4][3], double f[4], int max_iter, double fspread_tol, double xspread_tol) {
    for (int i = 0; i < 4; i++) f[i] = obj(w, L, U, simplex[i]);
    for (int iter = 0; iter < max_iter; iter++) {
        sort_simplex4(simplex, f);
        double xspread = 0.0;
        for (int i = 1; i < 4; i++) {
            for (int k = 0; k < 3; k++) {
                double d = fabs(simplex[i][k] - simplex[0][k]);
                if (d > xspread) xspread = d;
            }
        }
        if (fabs(f[3] - f[0]) < fspread_tol && xspread < xspread_tol) break;
        double cen[3] = {0, 0, 0};
        for (int i = 0; i < 3; i++) for (int k = 0; k < 3; k++) cen[k] += simplex[i][k] / 3.0;
        double xr[3];
        for (int k = 0; k < 3; k++) xr[k] = cen[k] + (cen[k] - simplex[3][k]);
        clamp(xr, L, U);
        double fr = obj(w, L, U, xr);
        if (fr < f[0]) {
            double xe[3];
            for (int k = 0; k < 3; k++) xe[k] = cen[k] + 2.0 * (xr[k] - cen[k]);
            clamp(xe, L, U);
            double fe = obj(w, L, U, xe);
            if (fe < fr) { memcpy(simplex[3], xe, sizeof xe); f[3] = fe; }
            else { memcpy(simplex[3], xr, sizeof xr); f[3] = fr; }
        } else if (fr < f[2]) {
            memcpy(simplex[3], xr, sizeof xr); f[3] = fr;
        } else {
            double xc[3];
            const double* from = (fr < f[3]) ? xr : simplex[3];
            for (int k = 0; k < 3; k++) xc[k] = cen[k] + 0.5 * (from[k] - cen[k]);
            clamp(xc, L, U);
            double fc = obj(w, L, U, xc);
            if (fc < f[3]) { memcpy(simplex[3], xc, sizeof xc); f[3] = fc; }
            else {
                for (int i = 1; i < 4; i++) {
                    for (int k = 0; k < 3; k++) simplex[i][k] = simplex[0][k] + 0.5 * (simplex[i][k] - simplex[0][k]);
                    clamp(simplex[i], L, U);
                    f[i] = obj(w, L, U, simplex[i]);
                }
            }
        }
    }
    sort_simplex4(simplex, f);
}

/* Nelder-Mead in (ln P, phi_p, ecc), x = (P, phi_p, ecc) in/out. */
static void ws_polish_logp(AstfitWorkspace* w, double* L, double* U, double* x, int max_iter, double fspread_tol, double xspread_tol) {
    double steps[3] = {0.05, 0.25, 0.08};
    double s[4][3], f[4];
    s[0][0] = log(x[0]); s[0][1] = x[1]; s[0][2] = x[2];
    clamp_astfit_logp_vertex(s[0], L, U);
    for (int i = 1; i < 4; i++) {
        memcpy(s[i], s[0], sizeof s[0]);
        s[i][i - 1] += steps[i - 1];
        clamp_astfit_logp_vertex(s[i], L, U);
        if (s[i][i - 1] == s[0][i - 1]) { s[i][i - 1] -= steps[i - 1]; clamp_astfit_logp_vertex(s[i], L, U); }
    }
    ws_nelder_mead(w, L, U, ws_objective_logp, clamp_astfit_logp_vertex, s, f, max_iter, fspread_tol, xspread_tol);
    x[0] = exp(s[0][0]); x[1] = s[0][1]; x[2] = s[0][2];
}

/* Periodogram + eccentric grid + multistart Nelder-Mead search for (P, phi_p, e).
   This is the same search as run_astfit_periodogram_multistart_config_gsl(), evaluated with
   the fast objective above. results[0:4] = P, phi_p, e, chi2. */
void run_astfit_periodogram_multistart_config(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, double* results, int n_obs, int reject_outlier, int n_period, int n_short_period, int n_ecc_phase, int max_periods_keep, int max_starts, int use_full_nm) {
    AstfitWorkspace* w = astfit_workspace_create(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_obs, reject_outlier);
    int max_periods = 32;
    int max_candidates = 256;
    double short_period_max = fmin(U[0], 300.0);
    double ecc_grid[6] = {0.0, 0.2, 0.4, 0.6, 0.8, 0.95};

    int max_rows = n_period + ((short_period_max > L[0]) ? n_short_period : 0) + 8;
    AstfitCandidate* rows = (AstfitCandidate*)malloc(max_rows * sizeof(AstfitCandidate));
    int n_rows = 0;
    for (int pass = 0; pass < 2; pass++) {
        double p_min = L[0], p_max = pass ? short_period_max : U[0];
        int np = pass ? n_short_period : n_period;
        if (pass && !(short_period_max > L[0])) break;
        for (int ip = 0; ip < np; ip++) {
            double frac = (np == 1) ? 0.0 : ((double)ip / (double)(np - 1));
            double period = exp(log(p_min) + frac * (log(p_max) - log(p_min)));
            rows[n_rows].chi2 = astfit_workspace_periodogram(w, period);
            rows[n_rows].period = period;
            rows[n_rows].phi_p = 0.0;
            rows[n_rows].ecc = 0.0;
            n_rows++;
        }
    }
    qsort(rows, n_rows, sizeof(AstfitCandidate), compare_astfit_candidate);

    double periods[32];
    int n_periods = 0;
    for (int i = 0; i < n_rows && n_periods < max_periods_keep; i++) {
        if (astfit_period_is_distinct(rows[i].period, periods, n_periods, 0.025)) periods[n_periods++] = rows[i].period;
    }
    for (int i = 0; i < 8 && n_periods < max_periods; i++) {
        double period = U[0] * exp(-((double)i / 7.0));
        if (period >= L[0] && period <= U[0] && astfit_period_is_distinct(period, periods, n_periods, 0.01)) periods[n_periods++] = period;
    }

    AstfitCandidate* candidates = (AstfitCandidate*)malloc(max_candidates * sizeof(AstfitCandidate));
    int n_candidates = 0;
    for (int i = 0; i < n_rows && i < 128; i++) {
        insert_astfit_candidate(candidates, &n_candidates, max_candidates, rows[i].chi2, rows[i].period, rows[i].phi_p, rows[i].ecc);
    }
    for (int ip = 0; ip < n_periods; ip++) {
        for (int ie = 0; ie < 6; ie++) {
            double ecc = ecc_grid[ie];
            if (ecc < L[2] || ecc > U[2]) continue;
            for (int iph = 0; iph < n_ecc_phase; iph++) {
                double phi_p = L[1] + ((double)iph / (double)n_ecc_phase) * (U[1] - L[1]);
                double chi2 = astfit_workspace_chi2(w, periods[ip], phi_p, ecc);
                insert_astfit_candidate(candidates, &n_candidates, max_candidates, chi2, periods[ip], phi_p, ecc);
            }
        }
    }

    AstfitCandidate* starts = (AstfitCandidate*)malloc(max_starts * sizeof(AstfitCandidate));
    int n_starts = 0;
    for (int i = 0; i < n_candidates && n_starts < max_starts; i++) {
        if (astfit_start_is_distinct(candidates[i], starts, n_starts)) starts[n_starts++] = candidates[i];
    }

    double best_chi2 = INFINITY;
    double best_x[3] = {(U[0] + L[0]) / 2.0, (U[1] + L[1]) / 2.0, (U[2] + L[2]) / 2.0};
    for (int i = 0; i < n_starts; i++) {
        double x[3] = {starts[i].period, starts[i].phi_p, starts[i].ecc};
        if (use_full_nm) {
            ws_polish_logp(w, L, U, x, ASTFIT_FULL_NM_MAX_ITER, ASTFIT_FULL_NM_FSPREAD_TOL, ASTFIT_FULL_NM_XSPREAD_TOL);
        } else {
            ws_polish_logp(w, L, U, x, ASTFIT_NM_MAX_ITER, ASTFIT_NM_FSPREAD_TOL, ASTFIT_NM_XSPREAD_TOL);
        }
        double chi2 = astfit_workspace_chi2(w, x[0], x[1], x[2]);
        if (chi2 < best_chi2) {
            best_chi2 = chi2;
            memcpy(best_x, x, sizeof best_x);
        }
    }

    results[0] = best_x[0];
    results[1] = best_x[1];
    results[2] = best_x[2];
    results[3] = best_chi2;

    free(rows);
    free(candidates);
    free(starts);
    astfit_workspace_free(w);
}

/* ------------- approximate chi2 for scoring the v2 eccentric seed grid -------------

   The eccentric grid of run_astfit_periodogram_multistart_v2 only ranks (phi_p, e) cells to pick
   starting points, so it does not need the exact chi2. grid_chi2_approx() takes
   X = cos E - e and Y = sqrt(1 - e^2) sin E from a table in M (4096 intervals, linear
   interpolation, no Kepler iteration) for the five grid eccentricities, and accumulates every sum
   needed for the 4x4 Schur complement of the orbital columns in a single pass, from per-epoch
   products of the fixed columns that are computed once per source. It returns chi2 without
   outlier rejection, to ~4e-4 relative. The best GRID_N_RESCORE cells at each seed period are
   then rescored with the exact chi2 (including outlier rejection), and the best of those is used,
   so the approximation only affects which cells are rescored. */
#define GRID_NT 4096
#define GRID_NE 5
#define GRID_N_RESCORE 12
static const double grid_e[GRID_NE] = {0.2, 0.4, 0.6, 0.8, 0.95};   /* must match the v2 eccentric grid */
static double grid_X[GRID_NE][GRID_NT + 1], grid_Y[GRID_NE][GRID_NT + 1];
static int grid_claimed = 0, grid_ready = 0;

/* Fill the tables once. Callers that find them not yet ready use the exact chi2 instead. */
static void grid_tables_init(void) {
    if (__atomic_load_n(&grid_ready, __ATOMIC_ACQUIRE)) return;
    if (__atomic_exchange_n(&grid_claimed, 1, __ATOMIC_ACQ_REL)) return;
    for (int ie = 0; ie < GRID_NE; ie++) {
        double e = grid_e[ie], sq = sqrt(1 - e * e);
        for (int k = 0; k <= GRID_NT; k++) {
            double M = 2 * PI * k / GRID_NT, E = KeplerStart3(e, M);
            for (int it = 0; it < 100; it++) {
                double d = (E - e * sin(E) - M) / (1 - e * cos(E));
                E -= d;
                if (fabs(d) < 1e-15) break;
            }
            grid_X[ie][k] = cos(E) - e;
            grid_Y[ie][k] = sq * sin(E);
        }
    }
    __atomic_store_n(&grid_ready, 1, __ATOMIC_RELEASE);
}

typedef struct {
    int n;
    double *W;              /* 12 per epoch: Q_k sin(psi)/err (k < 5), Q_k cos(psi)/err, r0 sin(psi)/err, r0 cos(psi)/err */
    double *s2, *sc, *c2;   /* sin^2(psi)/err^2, sin(psi) cos(psi)/err^2, cos^2(psi)/err^2 */
    double rr0;             /* |r0|^2 */
} GridWorkspace;

static GridWorkspace* grid_workspace_alloc(const AstfitWorkspace* w) {
    int n = w->n, nf = w->nfix;
    GridWorkspace* g = (GridWorkspace*)malloc(sizeof(GridWorkspace));
    g->n = n;
    g->W = (double*)calloc((long)n * 15, sizeof(double));
    g->s2 = g->W + 12 * n; g->sc = g->s2 + n; g->c2 = g->sc + n;
    g->rr0 = 0.0;
    for (int j = 0; j < n; j++) {
        double* wj = g->W + 12 * j;
        for (int k = 0; k < nf; k++) {
            wj[k] = w->Q[(long)k * n + j] * w->sw[j];
            wj[5 + k] = w->Q[(long)k * n + j] * w->cw[j];
        }
        wj[10] = w->r0[j] * w->sw[j];
        wj[11] = w->r0[j] * w->cw[j];
        g->s2[j] = w->sw[j] * w->sw[j]; g->sc[j] = w->sw[j] * w->cw[j]; g->c2[j] = w->cw[j] * w->cw[j];
        g->rr0 += w->r0[j] * w->r0[j];
    }
    return g;
}

static GridWorkspace* grid_workspace_create(AstfitWorkspace* w) {
    grid_tables_init();
    if (!__atomic_load_n(&grid_ready, __ATOMIC_ACQUIRE)) return NULL;
    return grid_workspace_alloc(w);
}

static void grid_workspace_free(GridWorkspace* g) {
    if (g) { free(g->W); free(g); }
}

/* Approximate chi2 (no outlier rejection) at (P, phi_p, grid_e[ie]); NAN if the 4x4 system is degenerate. */
static double grid_chi2_approx(GridWorkspace* g, AstfitWorkspace* w, double P, double phi_p, int ie) {
    int n = g->n;
    double aX[12] = {0}, aY[12] = {0}, h[9] = {0};
    const double *xt = grid_X[ie], *yt = grid_Y[ie];
    double s = 1.0 / (2 * PI * P), o = -phi_p / (2 * PI);
    for (int j = 0; j < n; j++) {
        double u = w->a[j] * s + o;   /* M / (2 pi) */
        u -= floor(u);
        double x = u * GRID_NT;
        int k = (int)x;
        if (k >= GRID_NT) k = GRID_NT - 1;
        double f = x - k;
        double X = xt[k] + f * (xt[k + 1] - xt[k]), Y = yt[k] + f * (yt[k + 1] - yt[k]);
        const double* wj = g->W + 12 * j;
        for (int m = 0; m < 12; m++) { aX[m] += X * wj[m]; aY[m] += Y * wj[m]; }
        double XX = X * X, XY = X * Y, YY = Y * Y;
        h[0] += XX * g->s2[j]; h[1] += XY * g->s2[j]; h[2] += YY * g->s2[j];
        h[3] += XX * g->sc[j]; h[4] += XY * g->sc[j]; h[5] += YY * g->sc[j];
        h[6] += XX * g->c2[j]; h[7] += XY * g->c2[j]; h[8] += YY * g->c2[j];
    }
    /* orbital columns: 0 = X sin(psi), 1 = Y sin(psi), 2 = X cos(psi), 3 = Y cos(psi) (whitened) */
    double S[4][4] = {{h[0], h[1], h[3], h[4]}, {h[1], h[2], h[4], h[5]}, {h[3], h[4], h[6], h[7]}, {h[4], h[5], h[7], h[8]}};
    double G[5][4], b[4] = {aX[10], aY[10], aX[11], aY[11]};
    for (int k = 0; k < 5; k++) { G[k][0] = aX[k]; G[k][1] = aY[k]; G[k][2] = aX[5 + k]; G[k][3] = aY[5 + k]; }
    /* chi2 = |r0|^2 - b^T (S - G^T G)^-1 b, via Cholesky */
    double Lc[4][4], z[4], drop = 0.0;
    for (int c = 0; c < 4; c++) {
        for (int d = 0; d <= c; d++) {
            double v = S[c][d];
            for (int k = 0; k < 5; k++) v -= G[k][c] * G[k][d];
            for (int m = 0; m < d; m++) v -= Lc[c][m] * Lc[d][m];
            if (d == c) {
                if (!(v > 1e-9 * S[c][c])) return NAN;
                Lc[c][c] = sqrt(v);
            } else Lc[c][d] = v / Lc[d][d];
        }
        double v = b[c];
        for (int m = 0; m < c; m++) v -= Lc[c][m] * z[m];
        z[c] = v / Lc[c][c];
        drop += z[c] * z[c];
    }
    return g->rr0 - drop;
}

/* grid_chi2_approx() at n_eval points (P[i], phi_p[i], ecc = grid_e[ie[i]]) on one source. Exposed for testing;
   NAN where the approximate system is degenerate. */
void get_chi2_astrometry_grid_approx(int n_obs, double *t_ast_yr, double *psi, double *plx_factor, double *ast_obs, double *ast_err, int n_eval, double *P, double *phi_p, int *ie, double *chi2_out) {
    AstfitWorkspace* w = astfit_workspace_create(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_obs, 0);
    GridWorkspace* g = grid_workspace_create(w);
    for (int i = 0; i < n_eval; i++) chi2_out[i] = g ? grid_chi2_approx(g, w, P[i], phi_p[i], ie[i]) : NAN;
    grid_workspace_free(g);
    astfit_workspace_free(w);
}

/* ---------------------------------------------------------------------------------------------
   run_astfit_periodogram_multistart_v2: a broader search on the same fast objective.

   Differences from run_astfit_periodogram_multistart:
   1. The circular periodogram is evaluated on a grid uniform in frequency, df = 1/(5 T) (T = time
      baseline), with the step capped at 0.02 in ln P. The log-spaced grid undersamples short
      periods, where peaks have width ~P/T in ln P.
   2. Seeds are the 12 deepest local minima of the periodogram, rather than the lowest grid rows
      (which tend to cluster on one broad peak). Around the 4 deepest, it also seeds the
      neighbouring frequencies f +- {0.5, 1, 2}/T and the harmonics 2P and 3P (the circular
      periodogram of an eccentric orbit often peaks at P/2 or P/3). Long-period anchors run from
      T/4 to P_max every 0.08 in ln P.
   3. At every seed period, an eccentric grid (e = 0.2 ... 0.95; 36 phases at periodogram seeds,
      18 at anchors) picks the best (phi_p, e). The cells are ranked with grid_chi2_approx() and the
      best 12 are rescored with the exact chi2 (run_astfit_periodogram_multistart_v2_exactgrid scores
      every cell exactly; it is kept for testing).
   4. Local optimization runs in (ln P, e cos phi_p, e sin phi_p), which has no degeneracy at e = 0
      and wraps phi_p: Levenberg-Marquardt (hk_levenberg_marquardt, analytic Jacobian) from the 8 deepest
      periodogram minima (circular) and the 12 best eccentric seeds, and the winner gets a final tight
      Nelder-Mead polish. With outlier rejection, Nelder-Mead (initial ln P step min(0.05, 0.15 P/T)) is used from
      all 12 periodogram minima and the 24 best eccentric seeds instead
      (run_astfit_periodogram_multistart_v2_nm does this in all cases; it is kept for testing).
   If the phi_p bounds do not span a full 2*pi, the local optimization falls back to Nelder-Mead
   in (ln P, phi_p, e) with clamped bounds. results[0:4] = P, phi_p, e, chi2.
   --------------------------------------------------------------------------------------------- */
#define ASTFIT_V2_OVERSAMPLE 5.0
#define ASTFIT_V2_MAX_DLNP 0.02
#define ASTFIT_V2_N_PEAKS 12
#define ASTFIT_V2_N_DENSE_PEAKS 4
#define ASTFIT_V2_ANCHOR_MIN_OVER_T 0.25
#define ASTFIT_V2_ANCHOR_DLNP 0.08
#define ASTFIT_V2_N_PHASE_PEAK 36
#define ASTFIT_V2_N_PHASE_ANCHOR 18
#define ASTFIT_V2_N_ECC_STARTS 24
#define ASTFIT_V2_STEP_SCALE 0.15
#define ASTFIT_V2_NM_MAX_ITER 400
#define ASTFIT_V2_NM_FSPREAD_TOL 1e-4
#define ASTFIT_V2_NM_XSPREAD_TOL 3e-6
#define ASTFIT_V2_FINAL_MAX_ITER 600
#define ASTFIT_V2_FINAL_FSPREAD_TOL 1e-7
#define ASTFIT_V2_FINAL_XSPREAD_TOL 1e-8

/* objective in q = (ln P, e cos phi_p, e sin phi_p) */
static double hk_objective(AstfitWorkspace* w, double* L, double* U, double* q) {
    double e = hypot(q[1], q[2]);
    if (q[0] < log(L[0]) || q[0] > log(U[0]) || e < L[2] - 1e-12 || e > U[2] + 1e-12) return INFINITY;
    return astfit_workspace_chi2(w, exp(q[0]), atan2(q[2], q[1]), e);
}

static void hk_clamp(double* q, double* L, double* U) {
    q[0] = fmin(fmax(q[0], log(L[0])), log(U[0]));
    double e = hypot(q[1], q[2]);
    if (e > U[2]) { q[1] *= U[2] / e; q[2] *= U[2] / e; }
    else if (e < L[2]) {
        if (e > 0) { q[1] *= L[2] / e; q[2] *= L[2] / e; }
        else q[1] = L[2];
    }
}

/* Nelder-Mead in hk coordinates. x = (P, phi_p, e) in/out; returns chi2. */
static double hk_nelder_mead(AstfitWorkspace* w, double* L, double* U, double* x, double step_scale, int max_iter, double fspread_tol, double xspread_tol) {
    double steps[3] = {fmin(0.05, step_scale * x[0] / fmax(w->tspan, 1.0)), 0.08, 0.08};
    double s[4][3], f[4];
    s[0][0] = log(x[0]); s[0][1] = x[2] * cos(x[1]); s[0][2] = x[2] * sin(x[1]);
    hk_clamp(s[0], L, U);
    for (int i = 1; i < 4; i++) {
        memcpy(s[i], s[0], sizeof s[0]);
        s[i][i - 1] += steps[i - 1];
        hk_clamp(s[i], L, U);
        if (s[i][i - 1] == s[0][i - 1]) { s[i][i - 1] -= steps[i - 1]; hk_clamp(s[i], L, U); }
    }
    ws_nelder_mead(w, L, U, hk_objective, hk_clamp, s, f, max_iter, fspread_tol, xspread_tol);
    x[0] = exp(s[0][0]);
    x[1] = fmod(atan2(s[0][2], s[0][1]) - L[1], 2 * PI);
    if (x[1] < 0) x[1] += 2 * PI;
    x[1] += L[1];
    x[2] = hypot(s[0][1], s[0][2]);
    return f[0];
}

/* ------------- Levenberg-Marquardt in q = (ln P, e cos phi_p, e sin phi_p) -------------

   Used by run_astfit_periodogram_multistart_v2 for the local optimization from each start (not the final
   polish, which stays Nelder-Mead) when the phi_p bounds span 2 pi and there is no outlier rejection.
   Variable projection: the residual vector r(q) is the whitened residual after the 9 linear parameters are
   solved for, so chi2 = |r|^2. Steps are capped at |d ln P| <= 0.5 and |d(e cos, e sin)| <= 0.3 and pulled
   back inside the bounds with hk_clamp. Iterations stop when chi2 improves by less than atol twice in a
   row, or when no damped step improves it.

   The normal equations J^T J and J^T r (J = dr/dq) are computed analytically (lm_state_normal). With Q the
   orthonormal fixed columns, B(q) the four orbital columns (X sw, Y sw, X cw, Y cw; X = cos E - e,
   Y = sqrt(1 - e^2) sin E), G = Q^T B, Bt = B - Q G, S = Bt^T Bt and P_perp the projector off span(Q, B),
   the derivative of the projected residual is (Golub & Pereyra 1973)
       dr/dtheta = -P_perp d - Bt S^-1 (dB^T r),      d = sum_c beta_c dB_c/dtheta.
   Both terms are kept, so this is the exact Jacobian and not the Kaufman approximation. The two terms are
   orthogonal and r is orthogonal to span(Q, B), so J is never formed:
       J_p.J_q = d_p.d_q - (Q^T d_p).(Q^T d_q) - z_p.z_q + w_p.w_q,   J_p.r = -sum_c beta_c (dB_p^T r)_c,
   with z_p = Lc^-1 (B^T d_p - G^T Q^T d_p), w_p = Lc^-1 dB_p^T r and S = Lc Lc^T. One chi2 evaluation
   (ws_schur_solve, shared with ws_chi2_schur) stores cos E, sin E, B, G, Lc, beta and r, and the
   normal equations then take one more pass over the epochs with no Kepler solve. The derivatives are taken
   in (ln P, phi_p, e) and transformed to q. Where the Schur solve is not used (ws_chi2_all fallback) or
   e < 1e-8 (phi_p undefined), the Jacobian is taken by forward differences instead, as it was everywhere
   before (run_astfit_periodogram_multistart_v2_fdjac still does this at every step; it is kept for testing). */
#define ASTFIT_V2_LM_N_CIRC_STARTS 8
#define ASTFIT_V2_LM_N_ECC_STARTS 12
#define ASTFIT_V2_LM_MAX_ITER 60
#define ASTFIT_V2_LM_ATOL 1e-4

static double hk_resid(AstfitWorkspace* w, double* L, double* U, const double* q, double* r) {
    double e = hypot(q[1], q[2]);
    if (q[0] < log(L[0]) || q[0] > log(U[0]) || e > U[2] + 1e-12 || e < L[2] - 1e-12) return INFINITY;
    double c = astfit_workspace_chi2(w, exp(q[0]), atan2(q[2], q[1]), e);
    memcpy(r, w->res, sizeof(double) * w->n);
    return c;
}

/* chi2 at q with everything lm_state_normal() needs left in s; s->r is the residual. Returns INFINITY outside
   the bounds. s->ok = 0 if the Schur solve was not usable (then chi2 and s->r come from ws_chi2_all), exactly as in
   astfit_workspace_chi2(). */
static double lm_state_eval(AstfitWorkspace* w, double* L, double* U, const double* q, SchurState* s) {
    double ecc = hypot(q[1], q[2]);
    if (q[0] < log(L[0]) || q[0] > log(U[0]) || ecc > U[2] + 1e-12 || ecc < L[2] - 1e-12) return INFINITY;
    double P = exp(q[0]), phi_p = atan2(q[2], q[1]), chi2, rw;
    int worst, nor;
    if (ws_schur_solve(w, P, phi_p, ecc, s, &chi2, &worst, &rw)) return chi2;
    chi2 = ws_chi2_all(w, P, phi_p, ecc, &nor);
    memcpy(s->r, w->res, sizeof(double) * w->n);
    return chi2;
}

/* A = J^T J and g = J^T r for J = dr/d(ln P, e cos phi_p, e sin phi_p) at the state s (see the notes above).
   Returns 0 if they are not available there. */
static int lm_state_normal(AstfitWorkspace* w, const SchurState* s, double A[3][3], double g[3]) {
    if (!s->ok || !(s->e > 1e-8)) return 0;
    int n = w->n, nf = w->nfix;
    double e = s->e, sq = sqrt(1 - e * e), iP = 1.0 / s->P, eosq = e / sq;
    const double* be = s->beta; const double* B = s->B;
    double Qd[3][5] = {{0}}, Bd[3][4] = {{0}}, Dr[3][4] = {{0}}, dd[6] = {0};
    for (int j = 0; j < n; j++) {
        double cE = s->cE[j], sE = s->sE[j], D = 1.0 / (1 - e * cE);     /* D = dE/dM; dE/de = sin E * D */
        double sw = w->sw[j], cw = w->cw[j];
        double Uc = be[0] * sw + be[2] * cw, Vc = be[1] * sw + be[3] * cw;  /* model = X Uc + Y Vc */
        double dXM = -sE * D, dYM = sq * cE * D;
        double dXe = sE * dXM - 1.0, dYe = sE * dYM - eosq * sE;
        double mlp = -w->a[j] * iP;                                      /* dM/dlnP; dM/dphi_p = -1 */
        double dX0 = dXM * mlp, dY0 = dYM * mlp, dX1 = -dXM, dY1 = -dYM;
        double d1 = Uc * dX1 + Vc * dY1, d0 = -d1 * mlp, d2 = Uc * dXe + Vc * dYe;
        double r = s->r[j], rs = r * sw, rc = r * cw;
        double v0 = B[j], v1 = B[n + j], v2 = B[2 * n + j], v3 = B[3 * n + j];
        for (int k = 0; k < nf; k++) { double qk = w->Q[(long)k * n + j]; Qd[0][k] += qk * d0; Qd[1][k] += qk * d1; Qd[2][k] += qk * d2; }
        Bd[0][0] += v0 * d0; Bd[0][1] += v1 * d0; Bd[0][2] += v2 * d0; Bd[0][3] += v3 * d0;
        Bd[1][0] += v0 * d1; Bd[1][1] += v1 * d1; Bd[1][2] += v2 * d1; Bd[1][3] += v3 * d1;
        Bd[2][0] += v0 * d2; Bd[2][1] += v1 * d2; Bd[2][2] += v2 * d2; Bd[2][3] += v3 * d2;
        Dr[0][0] += dX0 * rs; Dr[0][1] += dY0 * rs; Dr[0][2] += dX0 * rc; Dr[0][3] += dY0 * rc;
        Dr[1][0] += dX1 * rs; Dr[1][1] += dY1 * rs; Dr[1][2] += dX1 * rc; Dr[1][3] += dY1 * rc;
        Dr[2][0] += dXe * rs; Dr[2][1] += dYe * rs; Dr[2][2] += dXe * rc; Dr[2][3] += dYe * rc;
        dd[0] += d0 * d0; dd[1] += d1 * d0; dd[2] += d1 * d1; dd[3] += d2 * d0; dd[4] += d2 * d1; dd[5] += d2 * d2;
    }
    double z[3][4], ww[3][4], A0[3][3], g0[3];
    for (int p = 0; p < 3; p++) {
        for (int c = 0; c < 4; c++) {
            double v = Bd[p][c], u = Dr[p][c];
            for (int k = 0; k < nf; k++) v -= s->G[k][c] * Qd[p][k];
            for (int m = 0; m < c; m++) { v -= s->Lc[c][m] * z[p][m]; u -= s->Lc[c][m] * ww[p][m]; }
            z[p][c] = v / s->Lc[c][c]; ww[p][c] = u / s->Lc[c][c];
        }
        g0[p] = -(be[0] * Dr[p][0] + be[1] * Dr[p][1] + be[2] * Dr[p][2] + be[3] * Dr[p][3]);
    }
    for (int p = 0; p < 3; p++) for (int q = 0; q <= p; q++) {
        double v = dd[p * (p + 1) / 2 + q];
        for (int k = 0; k < nf; k++) v -= Qd[p][k] * Qd[q][k];
        for (int c = 0; c < 4; c++) v += ww[p][c] * ww[q][c] - z[p][c] * z[q][c];
        A0[p][q] = A0[q][p] = v;
    }
    /* (ln P, phi_p, e) -> (ln P, h, k):  d/dh = cos(phi) d/de - sin(phi)/e d/dphi,  d/dk = sin(phi) d/de + cos(phi)/e d/dphi */
    double cp = cos(s->phi), sp = sin(s->phi), ie = 1.0 / e;
    double Mt[3][3] = {{1, 0, 0}, {0, -sp * ie, cp * ie}, {0, cp, sp}};
    for (int a = 0; a < 3; a++) {
        g[a] = Mt[0][a] * g0[0] + Mt[1][a] * g0[1] + Mt[2][a] * g0[2];
        for (int b = 0; b < 3; b++) {
            double v = 0;
            for (int p = 0; p < 3; p++) for (int q = 0; q < 3; q++) v += Mt[p][a] * A0[p][q] * Mt[q][b];
            A[a][b] = v;
        }
    }
    return 1;
}

/* x = (P, phi_p, e) in/out; returns chi2 (reject_outlier must be 0). analytic = 0 takes the Jacobian by forward
   differences at every step. */
static double hk_levenberg_marquardt(AstfitWorkspace* w, double* L, double* U, double* x, int max_iter, double atol, int analytic) {
    int n = w->n;
    double* buf = (double*)malloc(sizeof(double) * n * 17);
    SchurState sa = {0, 0, 0, 0, {0}, {{0}}, {{0}}, buf, buf + n, buf + 2 * n, buf + 6 * n};
    SchurState sb = {0, 0, 0, 0, {0}, {{0}}, {{0}}, buf + 7 * n, buf + 8 * n, buf + 9 * n, buf + 13 * n};
    SchurState *cur = &sa, *tri = &sb;
    double* J = buf + 14 * n;
    double q[3] = {log(x[0]), x[2] * cos(x[1]), x[2] * sin(x[1])};
    hk_clamp(q, L, U);
    double f = lm_state_eval(w, L, U, q, cur);
    double lam = 1e-3, T = fmax(w->tspan, 1.0);
    int small = 0;
    for (int it = 0; it < max_iter && isfinite(f); it++) {
        double A[3][3], g[3];
        if (!analytic || !lm_state_normal(w, cur, A, g)) {
            /* forward-difference Jacobian; ln P step scaled to the period resolution P/T */
            double hs[3] = {1e-6 * fmin(1.0, exp(q[0]) / T), 1e-6, 1e-6};
            int ok = 1;
            for (int c = 0; c < 3 && ok; c++) {
                double qq[3] = {q[0], q[1], q[2]}, h = hs[c];
                double* Jc = J + (long)c * n;
                qq[c] += h;
                double fc = hk_resid(w, L, U, qq, Jc);
                if (!isfinite(fc)) { qq[c] = q[c] - h; h = -h; fc = hk_resid(w, L, U, qq, Jc); }
                if (!isfinite(fc)) { ok = 0; break; }
                for (int j = 0; j < n; j++) Jc[j] = (Jc[j] - cur->r[j]) / h;
            }
            if (!ok) break;
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b <= a; b++) A[a][b] = A[b][a] = ws_dot(J + (long)a * n, J + (long)b * n, n);
                g[a] = ws_dot(J + (long)a * n, cur->r, n);
            }
        }
        int accepted = 0;
        double f_old = f;
        for (int tr = 0; tr < 10; tr++) {
            double Lc[3][3], d[3] = {-g[0], -g[1], -g[2]};
            int bad = 0;
            for (int a = 0; a < 3 && !bad; a++) {
                for (int b = 0; b <= a; b++) {
                    double v = A[a][b] + (a == b ? lam * (A[a][a] + 1e-12) : 0.0);
                    for (int k = 0; k < b; k++) v -= Lc[a][k] * Lc[b][k];
                    if (a == b) {
                        if (!(v > 0)) { bad = 1; break; }
                        Lc[a][a] = sqrt(v);
                    } else Lc[a][b] = v / Lc[b][b];
                }
            }
            if (bad) { lam *= 10; continue; }
            for (int a = 0; a < 3; a++) { double v = d[a]; for (int k = 0; k < a; k++) v -= Lc[a][k] * d[k]; d[a] = v / Lc[a][a]; }
            for (int a = 2; a >= 0; a--) { double v = d[a]; for (int k = a + 1; k < 3; k++) v -= Lc[k][a] * d[k]; d[a] = v / Lc[a][a]; }
            double s = 1.0, dh = hypot(d[1], d[2]);
            if (fabs(d[0]) > 0.5) s = fmin(s, 0.5 / fabs(d[0]));
            if (dh > 0.3) s = fmin(s, 0.3 / dh);
            double qn[3] = {q[0] + s * d[0], q[1] + s * d[1], q[2] + s * d[2]};
            hk_clamp(qn, L, U);
            double fn = lm_state_eval(w, L, U, qn, tri);
            if (fn < f) {
                memcpy(q, qn, sizeof q);
                SchurState* t = cur; cur = tri; tri = t;
                f = fn;
                lam = fmax(lam / 3, 1e-12);
                accepted = 1;
                break;
            }
            lam *= 4;
        }
        if (!accepted) break;
        if (f_old - f < atol) { if (++small >= 2) break; } else small = 0;
    }
    free(buf);
    x[0] = exp(q[0]);
    x[1] = fmod(atan2(q[2], q[1]) - L[1], 2 * PI);
    if (x[1] < 0) x[1] += 2 * PI;
    x[1] += L[1];
    x[2] = hypot(q[1], q[2]);
    return f;
}

/* J^T J (9 values, row-major) and J^T r (3 values) at (P, phi_p, ecc), for J = dr/d(ln P, e cos phi_p, e sin phi_p):
   out_analytic from lm_state_normal(), out_numeric from central differences of the projected residual.
   Returns 1 if the analytic values are available there. Exposed for testing. */
int get_astfit_lm_normal_equations(int n_obs, double *t_ast_yr, double *psi, double *plx_factor, double *ast_obs, double *ast_err, double P, double phi_p, double ecc, double *out_analytic, double *out_numeric) {
    AstfitWorkspace* w = astfit_workspace_create(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_obs, 0);
    int n = w->n;
    double L[3] = {1e-3, 0.0, 0.0}, U[3] = {1e9, 2 * PI, 0.9999};
    double* buf = (double*)malloc(sizeof(double) * n * 12);
    SchurState s = {0, 0, 0, 0, {0}, {{0}}, {{0}}, buf, buf + n, buf + 2 * n, buf + 6 * n};
    double *J = buf + 7 * n, *rp = buf + 10 * n, *rm = buf + 11 * n;
    double q[3] = {log(P), ecc * cos(phi_p), ecc * sin(phi_p)}, A[3][3], g[3];
    lm_state_eval(w, L, U, q, &s);
    int ok = lm_state_normal(w, &s, A, g);
    for (int a = 0; a < 3; a++) {
        out_analytic[9 + a] = ok ? g[a] : NAN;
        for (int b = 0; b < 3; b++) out_analytic[3 * a + b] = ok ? A[a][b] : NAN;
    }
    for (int c = 0; c < 3; c++) {
        double h = (c == 0) ? 1e-6 * fmin(1.0, P / fmax(w->tspan, 1.0)) : 1e-6, qq[3] = {q[0], q[1], q[2]};
        qq[c] = q[c] + h; hk_resid(w, L, U, qq, rp);
        qq[c] = q[c] - h; hk_resid(w, L, U, qq, rm);
        for (int j = 0; j < n; j++) J[(long)c * n + j] = (rp[j] - rm[j]) / (2 * h);
    }
    for (int a = 0; a < 3; a++) {
        out_numeric[9 + a] = ws_dot(J + (long)a * n, s.r, n);
        for (int b = 0; b < 3; b++) out_numeric[3 * a + b] = ws_dot(J + (long)a * n, J + (long)b * n, n);
    }
    free(buf);
    astfit_workspace_free(w);
    return ok;
}

typedef struct { double chi2, period, phi_p, ecc; } AstfitSeed;

static int compare_astfit_seed(const void* a, const void* b) {
    double x = ((const AstfitSeed*)a)->chi2, y = ((const AstfitSeed*)b)->chi2;
    return (x < y) ? -1 : (x > y);
}

static void astfit_v2_search(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, double* results, int n_obs, int reject_outlier, int approx_grid, int use_lm) {
    AstfitWorkspace* w = astfit_workspace_create(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_obs, reject_outlier);
    double T = fmax(w->tspan, 1.0);
    int full_phase = (U[1] - L[1] >= 2 * PI - 1e-10);

    /* 1. circular periodogram, uniform in frequency */
    int cap = 1024, ng = 0;
    double* grid_P = (double*)malloc(sizeof(double) * cap);
    double* grid_chi2 = (double*)malloc(sizeof(double) * cap);
    double f = 1.0 / U[0], f_max = 1.0 / L[0];
    while (1) {
        if (ng == cap) {
            cap *= 2;
            grid_P = (double*)realloc(grid_P, sizeof(double) * cap);
            grid_chi2 = (double*)realloc(grid_chi2, sizeof(double) * cap);
        }
        grid_P[ng] = 1.0 / f;
        grid_chi2[ng] = astfit_workspace_periodogram(w, grid_P[ng]);
        ng++;
        if (f >= f_max) break;
        f = fmin(f + fmin(1.0 / (ASTFIT_V2_OVERSAMPLE * T), ASTFIT_V2_MAX_DLNP * f), f_max);
    }

    /* 2. seed periods: deepest local minima, their neighbours and harmonics, long-period anchors */
    AstfitSeed* minima = (AstfitSeed*)malloc(sizeof(AstfitSeed) * ng);
    int n_min = 0;
    for (int i = 0; i < ng; i++) {
        if (!isfinite(grid_chi2[i])) continue;
        if ((i == 0 || grid_chi2[i] < grid_chi2[i - 1]) && (i == ng - 1 || grid_chi2[i] <= grid_chi2[i + 1])) {
            AstfitSeed s = {grid_chi2[i], grid_P[i], L[1], 0.0};
            minima[n_min++] = s;
        }
    }
    qsort(minima, n_min, sizeof(AstfitSeed), compare_astfit_seed);

    int cap_seeds = ASTFIT_V2_N_PEAKS + ASTFIT_V2_N_DENSE_PEAKS * 8 + 512;
    AstfitSeed* seeds = (AstfitSeed*)malloc(sizeof(AstfitSeed) * cap_seeds);
    int n_seeds = 0;
    for (int i = 0; i < n_min && n_seeds < ASTFIT_V2_N_PEAKS; i++) seeds[n_seeds++] = minima[i];
    int n_peak_seeds = n_seeds;
    for (int i = 0; i < ASTFIT_V2_N_DENSE_PEAKS && i < n_peak_seeds; i++) {
        double f0 = 1.0 / seeds[i].period;
        double offsets[6] = {-2, -1, -0.5, 0.5, 1, 2};
        double extra[8];
        int n_extra = 0;
        for (int k = 0; k < 6; k++) if (f0 + offsets[k] / T > 0) extra[n_extra++] = 1.0 / (f0 + offsets[k] / T);
        extra[n_extra++] = 2 * seeds[i].period;
        extra[n_extra++] = 3 * seeds[i].period;
        for (int k = 0; k < n_extra; k++) {
            if (extra[k] < L[0] || extra[k] > U[0]) continue;
            AstfitSeed s = {INFINITY, extra[k], L[1], 0.0};
            seeds[n_seeds++] = s;
        }
    }
    int n_dense_seeds = n_seeds;   /* these get the finer phase grid */
    double lo = fmax(L[0], ASTFIT_V2_ANCHOR_MIN_OVER_T * T), hi = U[0];
    if (hi > lo) {
        int na = (int)ceil(log(hi / lo) / ASTFIT_V2_ANCHOR_DLNP);
        if (na < 1) na = 1;
        for (int i = 0; i <= na && n_seeds < cap_seeds; i++) {
            AstfitSeed s = {INFINITY, exp(log(lo) + (log(hi) - log(lo)) * i / (double)na), L[1], 0.0};
            seeds[n_seeds++] = s;
        }
    }

    /* 3. eccentric grid at each seed period -> best (phi_p, e) per seed */
    double ecc_grid[5] = {0.2, 0.4, 0.6, 0.8, 0.95};   /* grid_e tabulates X, Y at these */
    AstfitSeed* ecc_seeds = (AstfitSeed*)malloc(sizeof(AstfitSeed) * (n_seeds > 0 ? n_seeds : 1));
    GridWorkspace* gw = approx_grid ? grid_workspace_create(w) : NULL;
    for (int s = 0; s < n_seeds; s++) {
        int n_phase = (s < n_dense_seeds) ? ASTFIT_V2_N_PHASE_PEAK : ASTFIT_V2_N_PHASE_ANCHOR;
        AstfitSeed b = {INFINITY, seeds[s].period, L[1], 0.0};
        /* cells kept for exact rescoring, sorted by approximate chi2 */
        double kc[GRID_N_RESCORE], kp[GRID_N_RESCORE], ke[GRID_N_RESCORE];
        int nk = 0;
        for (int ie = 0; ie < 5; ie++) {
            if (ecc_grid[ie] < L[2] || ecc_grid[ie] > U[2]) continue;
            for (int ip = 0; ip < n_phase; ip++) {
                double phi_p = L[1] + (U[1] - L[1]) * ip / (double)n_phase;
                double chi2 = gw ? grid_chi2_approx(gw, w, seeds[s].period, phi_p, ie) : NAN;
                if (!(chi2 == chi2)) {
                    /* exact score: used directly */
                    chi2 = astfit_workspace_chi2(w, seeds[s].period, phi_p, ecc_grid[ie]);
                    if (chi2 < b.chi2) { b.chi2 = chi2; b.phi_p = phi_p; b.ecc = ecc_grid[ie]; }
                    continue;
                }
                if (nk < GRID_N_RESCORE || chi2 < kc[nk - 1]) {
                    int q = (nk < GRID_N_RESCORE) ? nk++ : nk - 1;
                    while (q > 0 && kc[q - 1] > chi2) { kc[q] = kc[q - 1]; kp[q] = kp[q - 1]; ke[q] = ke[q - 1]; q--; }
                    kc[q] = chi2; kp[q] = phi_p; ke[q] = ecc_grid[ie];
                }
            }
        }
        for (int q = 0; q < nk; q++) {
            double chi2 = astfit_workspace_chi2(w, seeds[s].period, kp[q], ke[q]);
            if (chi2 < b.chi2) { b.chi2 = chi2; b.phi_p = kp[q]; b.ecc = ke[q]; }
        }
        ecc_seeds[s] = b;
    }
    grid_workspace_free(gw);
    qsort(ecc_seeds, n_seeds, sizeof(AstfitSeed), compare_astfit_seed);

    /* 4. local optimization from every periodogram minimum (circular) and the best eccentric seeds */
    double best_chi2 = INFINITY;
    double best_x[3] = {(U[0] + L[0]) / 2.0, (U[1] + L[1]) / 2.0, (U[2] + L[2]) / 2.0};
    for (int pass = 0; pass < 2; pass++) {
        int lm = use_lm && full_phase && !reject_outlier;
        int max_ecc = lm ? ASTFIT_V2_LM_N_ECC_STARTS : ASTFIT_V2_N_ECC_STARTS;
        int n_try = pass ? (n_seeds < max_ecc ? n_seeds : max_ecc) : n_peak_seeds;
        if (lm && !pass && n_try > ASTFIT_V2_LM_N_CIRC_STARTS) n_try = ASTFIT_V2_LM_N_CIRC_STARTS;
        for (int i = 0; i < n_try; i++) {
            double x[3];
            if (pass) {
                if (!isfinite(ecc_seeds[i].chi2)) continue;
                x[0] = ecc_seeds[i].period; x[1] = ecc_seeds[i].phi_p; x[2] = ecc_seeds[i].ecc;
            } else {
                x[0] = seeds[i].period; x[1] = L[1]; x[2] = fmax(L[2], 0.0);
            }
            double chi2;
            if (lm) {
                chi2 = hk_levenberg_marquardt(w, L, U, x, ASTFIT_V2_LM_MAX_ITER, ASTFIT_V2_LM_ATOL, use_lm == 1);
            } else if (full_phase) {
                chi2 = hk_nelder_mead(w, L, U, x, ASTFIT_V2_STEP_SCALE, ASTFIT_V2_NM_MAX_ITER, ASTFIT_V2_NM_FSPREAD_TOL, ASTFIT_V2_NM_XSPREAD_TOL);
            } else {
                ws_polish_logp(w, L, U, x, ASTFIT_V2_NM_MAX_ITER, ASTFIT_V2_NM_FSPREAD_TOL, ASTFIT_V2_NM_XSPREAD_TOL);
                chi2 = astfit_workspace_chi2(w, x[0], x[1], x[2]);
            }
            if (chi2 < best_chi2) { best_chi2 = chi2; memcpy(best_x, x, sizeof best_x); }
        }
    }

    /* 5. tight final polish of the best solution */
    if (isfinite(best_chi2)) {
        double x[3] = {best_x[0], best_x[1], best_x[2]};
        double chi2;
        if (full_phase) {
            chi2 = hk_nelder_mead(w, L, U, x, 0.1 * ASTFIT_V2_STEP_SCALE, ASTFIT_V2_FINAL_MAX_ITER, ASTFIT_V2_FINAL_FSPREAD_TOL, ASTFIT_V2_FINAL_XSPREAD_TOL);
        } else {
            ws_polish_logp(w, L, U, x, ASTFIT_V2_FINAL_MAX_ITER, ASTFIT_V2_FINAL_FSPREAD_TOL, ASTFIT_V2_FINAL_XSPREAD_TOL);
            chi2 = astfit_workspace_chi2(w, x[0], x[1], x[2]);
        }
        if (chi2 < best_chi2) { best_chi2 = chi2; memcpy(best_x, x, sizeof best_x); }
    }

    results[0] = best_x[0];
    results[1] = best_x[1];
    results[2] = best_x[2];
    results[3] = best_chi2;

    free(grid_P); free(grid_chi2); free(minima); free(seeds); free(ecc_seeds);
    astfit_workspace_free(w);
}

void run_astfit_periodogram_multistart_v2(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, double* results, int n_obs, int reject_outlier) {
    astfit_v2_search(t_ast_yr, psi, plx_factor, ast_obs, ast_err, L, U, results, n_obs, reject_outlier, 1, 1);
}

/* The same search with Nelder-Mead from every start (12 circular + 24 eccentric) instead of Levenberg-Marquardt
   from 8 + 12 (the behaviour before Levenberg-Marquardt was added); kept for testing. */
void run_astfit_periodogram_multistart_v2_nm(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, double* results, int n_obs, int reject_outlier) {
    astfit_v2_search(t_ast_yr, psi, plx_factor, ast_obs, ast_err, L, U, results, n_obs, reject_outlier, 1, 0);
}

/* The same search with the Levenberg-Marquardt Jacobian taken by forward differences at every step (the behaviour
   before the analytic Jacobian was added); kept for testing. */
void run_astfit_periodogram_multistart_v2_fdjac(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, double* results, int n_obs, int reject_outlier) {
    astfit_v2_search(t_ast_yr, psi, plx_factor, ast_obs, ast_err, L, U, results, n_obs, reject_outlier, 1, 2);
}

/* The same search with every eccentric-grid cell scored by the exact chi2 (the behaviour before the
   approximate grid scoring was added); kept for testing. */
void run_astfit_periodogram_multistart_v2_exactgrid(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, double* results, int n_obs, int reject_outlier) {
    astfit_v2_search(t_ast_yr, psi, plx_factor, ast_obs, ast_err, L, U, results, n_obs, reject_outlier, 0, 0);
}

/* ------------- outlier clipping with restoration around the v2 search -------------

   run_astfit_periodogram_multistart_v2_clip: fit with v2 on the epochs kept by astfit_prescreen (a data-only
   outlier screen); then up to ASTFIT_CLIP_ROUNDS times,
   compute the whitened residuals z of every epoch for that orbit (linear parameters fitted to the kept epochs),
   keep the epochs with |z| < clip_sigma * s, where s = max(1, 1.4826 * median |z| of the kept epochs), and, if
   the kept set changed, search again with v2 on the kept epochs. Clipped epochs return if they fit a later
   orbit (including epochs flagged by the prescreen). Without outliers nothing is clipped and the result is
   normally the plain v2 fit.
   If the final mask removes more than max_clipped epochs, the result is the plain v2 fit to all epochs instead
   (a source that needs many epochs removed to fit an orbit is not treated as an orbit with a few outliers).
   keep_out[i] = 1 for kept epochs, 0 for clipped ones. results = P, phi_p, e, chi2 (kept epochs), n_clipped. */
#define ASTFIT_CLIP_ROUNDS 4
#define ASTFIT_CLIP_MIN_KEEP 20

static int compare_double(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x < y) ? -1 : (x > y);
}

/* Least squares of the data column A[ncol] on the columns A[0..ncol-1] (each n long, already whitened), using only
   the epochs with keep[j] != 0 (modified Gram-Schmidt, applied twice; near-degenerate columns are dropped).
   z[j] = residual of every epoch, kept or not. */
static void ls_residuals(const double* A, int ncol, const int* keep, int n, double* z) {
    double* Q = (double*)malloc(sizeof(double) * (long)n * ncol);
    double* r = (double*)malloc(sizeof(double) * n);
    double R[16][16] = {{0}}, b[16] = {0}, coef[16] = {0};
    int ok[16] = {0};
    for (int j = 0; j < n; j++) r[j] = keep[j] ? A[(long)ncol * n + j] : 0.0;
    for (int k = 0; k < ncol; k++) {
        double* v = Q + (long)k * n;
        double n0 = 0;
        for (int j = 0; j < n; j++) { v[j] = keep[j] ? A[(long)k * n + j] : 0.0; n0 += v[j] * v[j]; }
        for (int pass = 0; pass < 2; pass++)
            for (int m = 0; m < k; m++) {
                if (!ok[m]) continue;
                double d = ws_dot(Q + (long)m * n, v, n);
                R[m][k] += d;
                for (int j = 0; j < n; j++) v[j] -= d * Q[(long)m * n + j];
            }
        double nn = ws_dot(v, v, n);
        if (n0 == 0 || !(nn > 1e-20 * n0)) continue;
        ok[k] = 1;
        nn = sqrt(nn); R[k][k] = nn;
        for (int j = 0; j < n; j++) v[j] /= nn;
        b[k] = ws_dot(v, r, n);
    }
    for (int k = ncol - 1; k >= 0; k--) {
        if (!ok[k]) continue;
        double s = b[k];
        for (int m = k + 1; m < ncol; m++) if (ok[m]) s -= R[k][m] * coef[m];
        coef[k] = s / R[k][k];
    }
    for (int j = 0; j < n; j++) {
        double model = 0;
        for (int k = 0; k < ncol; k++) model += coef[k] * A[(long)k * n + j];
        z[j] = A[(long)ncol * n + j] - model;
    }
    free(Q); free(r);
}

/* whitened residuals of all n epochs for the orbit (P, phi_p, ecc), with the 9 linear parameters fitted to the
   kept epochs */
static void astfit_residuals_all(double* t, double* psi, double* pf, double* y, double* err, int n, const int* keep, double P, double phi_p, double ecc, double* z) {
    double sq = sqrt(1 - ecc * ecc);
    double* A = (double*)malloc(sizeof(double) * (long)n * 10);
    for (int j = 0; j < n; j++) {
        double cE, sE, iw = 1.0 / err[j], s = sin(psi[j]), c = cos(psi[j]);
        kepler_cos_sin(2 * PI * t[j] * 365.25 / P - phi_p, ecc, &cE, &sE);
        double X = cE - ecc, Y = sq * sE;
        double col[10] = {s, t[j] * s, c, t[j] * c, pf[j], X * s, Y * s, X * c, Y * c, y[j]};
        for (int k = 0; k < 10; k++) A[(long)k * n + j] = col[k] * iw;
    }
    ls_residuals(A, 9, keep, n, z);
    free(A);
}

static double median_abs_kept(const double* z, const int* keep, int n, double* work) {
    int m = 0;
    for (int j = 0; j < n; j++) if (keep[j]) work[m++] = fabs(z[j]);
    if (m == 0) return 0.0;
    qsort(work, m, sizeof(double), compare_double);
    return (m % 2) ? work[m / 2] : 0.5 * (work[m / 2 - 1] + work[m / 2]);
}

/* Starting mask for the clipping, from the data alone (no orbit): an epoch is flagged if
   (a) its five-parameter residual disagrees with the other transits of its clump (transits less than 0.35 d
       apart, e.g. the two fields of view): by more than max(4.5, 0.15 rms) from the median for a pair (both
       members flagged), max(5, 0.15 rms) for three or more, in units of the epoch's error; or
   (b) its residual from a nine-parameter fit (five + cubic in time) exceeds 7 max(1, 1.4826 MAD), removing the
       worst epoch at a time, up to 6 times.
   Flags are dropped if fewer than ASTFIT_CLIP_MIN_KEEP epochs would remain. The clipping rounds restore any
   flagged epoch that fits the orbit. */
static void astfit_prescreen(double* t, double* psi, double* pf, double* y, double* err, int n, int* keep) {
    double* A = (double*)malloc(sizeof(double) * (long)n * 10);
    double* z = (double*)malloc(sizeof(double) * n);
    double* work = (double*)malloc(sizeof(double) * n);
    int* all = (int*)malloc(sizeof(int) * n);
    for (int j = 0; j < n; j++) all[j] = keep[j] = 1;
    /* (a) five-parameter residuals and clumps */
    for (int j = 0; j < n; j++) {
        double iw = 1.0 / err[j], s = sin(psi[j]), c = cos(psi[j]);
        double col[6] = {s, t[j] * s, c, t[j] * c, pf[j], y[j]};
        for (int k = 0; k < 6; k++) A[(long)k * n + j] = col[k] * iw;
    }
    ls_residuals(A, 5, all, n, z);
    double rms = sqrt(ws_dot(z, z, n) / n);
    int a = 0;
    while (a < n) {
        int b = a + 1;
        while (b < n && (t[b] - t[b - 1]) * 365.25 < 0.35) b++;
        int m = b - a;
        if (m >= 2) {
            for (int j = a; j < b; j++) work[j - a] = z[j];
            qsort(work, m, sizeof(double), compare_double);
            double med = (m % 2) ? work[m / 2] : 0.5 * (work[m / 2 - 1] + work[m / 2]);
            double thr = fmax(m == 2 ? 4.5 : 5.0, 0.15 * rms);
            for (int j = a; j < b; j++) if (fabs(z[j] - med) > thr) keep[j] = 0;
        }
        a = b;
    }
    /* (b) nine-parameter polynomial */
    for (int j = 0; j < n; j++) {
        double iw = 1.0 / err[j], s = sin(psi[j]), c = cos(psi[j]), t2 = t[j] * t[j], t3 = t2 * t[j];
        double col[10] = {s, t[j] * s, c, t[j] * c, pf[j], t2 * s, t2 * c, t3 * s, t3 * c, y[j]};
        for (int k = 0; k < 10; k++) A[(long)k * n + j] = col[k] * iw;
    }
    for (int pass = 0; pass < 6; pass++) {
        ls_residuals(A, 9, keep, n, z);
        double sg = fmax(1.0, 1.4826 * median_abs_kept(z, keep, n, work));
        int worst = -1; double zw = 0;
        for (int j = 0; j < n; j++) if (keep[j] && fabs(z[j]) > zw) { zw = fabs(z[j]); worst = j; }
        if (worst < 0 || zw <= 7 * sg) break;
        keep[worst] = 0;
    }
    int nk = 0;
    for (int j = 0; j < n; j++) nk += keep[j];
    if (nk < ASTFIT_CLIP_MIN_KEEP) for (int j = 0; j < n; j++) keep[j] = 1;
    free(A); free(z); free(work); free(all);
}

/* search with v2 on the epochs with keep[j] != 0; sub is 5n scratch */
static void v2_on_kept(double* t, double* psi, double* pf, double* y, double* err, int n, const int* keep, double* L, double* U, double* sol, double* sub) {
    int k = 0;
    for (int j = 0; j < n; j++) {
        if (!keep[j]) continue;
        sub[k] = t[j]; sub[n + k] = psi[j]; sub[2L * n + k] = pf[j]; sub[3L * n + k] = y[j]; sub[4L * n + k] = err[j];
        k++;
    }
    run_astfit_periodogram_multistart_v2(sub, sub + n, sub + 2L * n, sub + 3L * n, sub + 4L * n, L, U, sol, k, 0);
}

void run_astfit_periodogram_multistart_v2_clip(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, double* results, int n_obs, double* keep_out, double clip_sigma, int max_clipped) {
    int n = n_obs;
    int* keep = (int*)malloc(sizeof(int) * n);
    int* trial = (int*)malloc(sizeof(int) * n);
    double* z = (double*)malloc(sizeof(double) * n);
    double* work = (double*)malloc(sizeof(double) * n);
    double* sub = (double*)malloc(sizeof(double) * 5L * n);
    double sol[4];
    astfit_prescreen(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n, keep);
    v2_on_kept(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n, keep, L, U, sol, sub);
    for (int round = 0; round < ASTFIT_CLIP_ROUNDS; round++) {
        astfit_residuals_all(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n, keep, sol[0], sol[1], sol[2], z);
        double thr = clip_sigma * fmax(1.0, 1.4826 * median_abs_kept(z, keep, n, work));
        int nk = 0, changed = 0;
        for (int j = 0; j < n; j++) { trial[j] = fabs(z[j]) < thr; nk += trial[j]; }
        if (nk < ASTFIT_CLIP_MIN_KEEP) for (int j = 0; j < n; j++) trial[j] = 1;
        for (int j = 0; j < n; j++) if (trial[j] != keep[j]) changed = 1;
        if (!changed) break;
        memcpy(keep, trial, sizeof(int) * n);
        v2_on_kept(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n, keep, L, U, sol, sub);
    }
    int nclip = 0;
    for (int j = 0; j < n; j++) nclip += !keep[j];
    if (nclip > max_clipped) {
        /* more epochs would be clipped than allowed: return the plain fit to all epochs instead */
        for (int j = 0; j < n; j++) keep[j] = 1;
        run_astfit_periodogram_multistart_v2(t_ast_yr, psi, plx_factor, ast_obs, ast_err, L, U, sol, n, 0);
        nclip = 0;
    }
    for (int j = 0; j < n; j++) keep_out[j] = keep[j];
    results[0] = sol[0]; results[1] = sol[1]; results[2] = sol[2]; results[3] = sol[3]; results[4] = nclip;
    free(keep); free(trial); free(z); free(work); free(sub);
}

void run_astfit_periodogram_multistart(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, double* results, int n_obs, int reject_outlier) {
    run_astfit_periodogram_multistart_config(t_ast_yr, psi, plx_factor, ast_obs, ast_err, L, U, results, n_obs, reject_outlier, ASTFIT_N_PERIOD, ASTFIT_N_SHORT_PERIOD, ASTFIT_N_ECC_PHASE, ASTFIT_MAX_PERIODS_KEEP, ASTFIT_MAX_STARTS, 0);
}

void run_astfit_periodogram_multistart_full(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, double* results, int n_obs, int reject_outlier) {
    run_astfit_periodogram_multistart_config(t_ast_yr, psi, plx_factor, ast_obs, ast_err, L, U, results, n_obs, reject_outlier, ASTFIT_FULL_N_PERIOD, ASTFIT_FULL_N_SHORT_PERIOD, ASTFIT_FULL_N_ECC_PHASE, ASTFIT_FULL_MAX_PERIODS_KEEP, ASTFIT_FULL_MAX_STARTS, 1);
}

/* This is a helper function for the adaptive simulated annealing. x is the current guess of the array of parameters. L and U are arrays of lower and upper limits. xnew will be the proposed new gues. Tgen is the temperature and len is the number of free parameters. */
void generate_xnew_from_generation_temperatures(double* x, double* L, double* U, double* Tgen, int len, double* xnew) {
    // Copy x to xnew
    for (int i = 0; i < len; i++) {
        xnew[i] = x[i];
    }

    for (int g = 0; g < len; g++) {
        while (1) {
            double rand = random_double(); // rand between 0 and 1
            int sgn = (rand - 0.5) >= 0 ? 1 : -1;  // sign of rand-1/2
            double q = sgn * Tgen[g] * (pow((1.0 + 1.0 / Tgen[g]), fabs(2 * rand - 1)) - 1);
            xnew[g] = x[g] + q * (U[g] - L[g]);
            if (L[g] <= xnew[g] && xnew[g] <= U[g]) {
                break;
            }
        }
    }
}

/* This runs adaptive simulated annealing to find the best-fit nonlinear parameters (P, phi_p, e). t_ast_year are the timestamps of the epoch astrometry, psi are the scan angles, plx_factor the parallax factors, ast_obs and ast_err the epoch astrometry and uncertainties, L and U the lower and upper limits of fitting ranges, and n_obs the number of measurements. results is an array that will be filled with the best-fit parameters.
based on https://arxiv.org/abs/1505.04767
*/ 

void run_astfit(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, double* results, int n_obs) {
    double eps = 1e-5; // all the tunable parameters are taken from RVFIT ( https://arxiv.org/abs/1505.04767)
    int Neps = 5;
    int Nterm = 20;
    int npar = 3;

    double* fbestlist = (double*)malloc(Neps * sizeof(double));
    for (int i = 0; i < Neps; i++) {
        fbestlist[i] = 1.0;
    }
    int nrean = 0;

    double mean_err = 0.0;
    for (int i = 0; i < n_obs; i++) {
        mean_err += ast_err[i];
    }
    mean_err /= n_obs;

    double chislimit = 0.0;
    for (int i = 0; i < n_obs; i++) {
        chislimit += pow(mean_err / ast_err[i], 2);
    }

    double* x = (double*)malloc(npar * sizeof(double));
    for (int i = 0; i < npar; i++) {
        x[i] = (U[i] + L[i]) / 2.0;
    }

    double* chi2_array = (double*)malloc(10 * sizeof(double));
    get_chi2_astrometry(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err, x[0], x[1], x[2], chi2_array);
    double f = chi2_array[0];
    double* xbest = (double*)malloc(npar * sizeof(double));
    double fbest = f;
    for (int i = 0; i < npar; i++) {
        xbest[i] = x[i];
    }

    int ka = 0;
    double Ta0 = f;
    double Ta = Ta0;
    int nacep = 0;

    double* kgen = (double*)calloc(npar, sizeof(double));
    double* Tgen0 = (double*)malloc(npar * sizeof(double));
    for (int i = 0; i < npar; i++) {
        Tgen0[i] = 1.0;
    }
    double* Tgen = (double*)malloc(npar * sizeof(double));
    for (int i = 0; i < npar; i++) {
        Tgen[i] = Tgen0[i];
    }

    double c = 20.0;
    int Na = 1000;
    int Ngen = 10000;
    double* delta = (double*)malloc(npar * sizeof(double));
    for (int i = 0; i < npar; i++) {
        delta[i] = fabs(U[i] - L[i]) * 1e-8;
    }

    double acep = 0.25;
    int ntest = 100;
    double* ftest = (double*)calloc(ntest, sizeof(double));

    for (int j = 0; j < ntest; j++) {
        double xnew[npar];
        generate_xnew_from_generation_temperatures(x, L, U, Tgen, npar, xnew);
        get_chi2_astrometry(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err, xnew[0], xnew[1], xnew[2], chi2_array);
        ftest[j] = chi2_array[0];
        if (ftest[j] < fbest) {
            fbest = ftest[j];
            for (int i = 0; i < npar; i++) {
                xbest[i] = xnew[i];
            }
        }
    }

    double* dftest = (double*)malloc((ntest - 1) * sizeof(double));
    for (int i = 0; i < ntest - 1; i++) {
        dftest[i] = ftest[i + 1] - ftest[i];
    }

    double avdftest = 0.0;
    for (int i = 0; i < ntest - 1; i++) {
        avdftest += fabs(dftest[i]);
    }
    avdftest /= (ntest - 1);

    Ta0 = avdftest / log(1.0 / acep - 1.0);

    int repeat = 1;
    while (repeat) {
        for (int j = 0; j < Ngen; j++) {
            int flag_acceptance = 0;
            double xnew[npar];
            generate_xnew_from_generation_temperatures(x, L, U, Tgen, npar, xnew);
            get_chi2_astrometry(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err, xnew[0], xnew[1], xnew[2], chi2_array);
            double fnew = chi2_array[0];

            if (fnew <= f) {
                flag_acceptance = 1;
            } else {
                double test = (fnew - f) / Ta;
                double Pa = (test > 20) ? 0 : 1.0 / (1.0 + exp(test));
                if (random_double() <= Pa) {
                    flag_acceptance = 1;
                }
            }

            if (flag_acceptance) {
                if (fnew < fbest) {
                    fbest = fnew;
                    for (int i = 0; i < npar; i++) {
                        xbest[i] = xnew[i];
                    }
                    nrean = 0;
                }
                nacep++;
                ka++;
                for (int i = 0; i < npar; i++) {
                    x[i] = xnew[i];
                }
                f = fnew;
            }

            if (nacep >= Na) {
                double* s = (double*)calloc(npar, sizeof(double));

                for (int g = 0; g < npar; g++) {
                    double xdelta[npar];
                    for (int i = 0; i < npar; i++) {
                        xdelta[i] = xbest[i];
                    }
                    xdelta[g] += delta[g];

                    get_chi2_astrometry(
                        n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err,xdelta[0], xdelta[1], xdelta[2], chi2_array);

                    double fbestdelta = chi2_array[0];
                    s[g] = fabs((fbestdelta - fbest) / delta[g]);
                }

                for (int i = 0; i < npar; i++) {
                    if (s[i] == 0.0) {
                        double min_nonzero = 1e10;
                        for (int j = 0; j < npar; j++) {
                            if (s[j] != 0.0 && s[j] < min_nonzero) {
                                min_nonzero = s[j];
                            }
                        }
                        s[i] = min_nonzero;
                    }
                }

                double smax = 0.0;
                for (int i = 0; i < npar; i++) {
                    if (s[i] > smax) {
                        smax = s[i];
                    }
                }

                for (int i = 0; i < npar; i++) {
                    Tgen[i] = Tgen[i] * (smax / s[i]);
                    kgen[i] = pow(log(Tgen0[i] / Tgen[i]) / c, npar);
                    kgen[i] = fabs(kgen[i]);
                }

                Ta0 = f;
                Ta = fbest;
                ka = pow(log(Ta0 / Ta) / c, npar);

                int stop = 0;
                for (int i = 0; i < npar; i++) {
                    if (Tgen[i] == 0) {
                        stop = 1;
                        break;
                    }
                }
                if (stop) {
                    repeat = 0;
                    break;
                }

                nacep = 0;
                nrean++;
                free(s);
            }
        }

        for (int i = 0; i < npar; i++) {
            kgen[i] += 1;
            Tgen[i] = Tgen0[i] * exp(-c * pow(kgen[i], 1.0 / npar));
        }

        ka += 1;
        Ta = Ta0 * exp(-c * pow(ka, 1.0 / npar));

        for (int i = 1; i < Neps; i++) {
            fbestlist[i - 1] = fbestlist[i];
        }
        fbestlist[Neps - 1] = fbest;

        int count = 0;
        for (int i = 0; i < Neps - 1; i++) {
            if (fabs(fbestlist[i + 1] - fbestlist[i]) < eps) {
                count++;
            }
        }
        if (count == Neps - 1) {
            // printf("%f %f\n", fbest, chislimit);
            if (fbest < chislimit) {
                repeat = 0;
            } else {
                Ta = Ta0;
            }
        }
        if (nrean >= Nterm) {
            repeat = 0;
        }
    }

    free(fbestlist);
    free(kgen);
    free(Tgen0);
    free(Tgen);
    free(delta);
    free(ftest);
    free(dftest);
    free(x);
    free(chi2_array);

    for (int i = 0; i < npar; i++) {
        results[i] = xbest[i];
    }

    free(xbest);
}


/* This is identical to run_astfit(), except that it includes an eccentricity prior of ecc ~ beta(ecc, ecc_a, ecc_b) */

void run_astfit_eccentricity_prior(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, double* results, int n_obs, double ecc_a, double ecc_b) {
    double eps = 1e-5; // all the tunable parameters are taken from RVFIT ( https://arxiv.org/abs/1505.04767)
    int Neps = 5;
    int Nterm = 20;
    int npar = 3;

    double* fbestlist = (double*)malloc(Neps * sizeof(double));
    for (int i = 0; i < Neps; i++) {
        fbestlist[i] = 1.0;
    }
    int nrean = 0;

    double mean_err = 0.0;
    for (int i = 0; i < n_obs; i++) {
        mean_err += ast_err[i];
    }
    mean_err /= n_obs;

    double chislimit = 0.0;
    for (int i = 0; i < n_obs; i++) {
        chislimit += pow(mean_err / ast_err[i], 2);
    }

    double* x = (double*)malloc(npar * sizeof(double));
    for (int i = 0; i < npar; i++) {
        x[i] = (U[i] + L[i]) / 2.0;
    }

    double* chi2_array = (double*)malloc(10 * sizeof(double));
    get_chi2_astrometry(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err, x[0], x[1], x[2], chi2_array);
    double f = chi2_array[0] - 2.0 * log_beta_prior(x[2], ecc_a, ecc_b);
    double* xbest = (double*)malloc(npar * sizeof(double));
    double fbest = f;
    for (int i = 0; i < npar; i++) {
        xbest[i] = x[i];
    }

    int ka = 0;
    double Ta0 = f;
    double Ta = Ta0;
    int nacep = 0;

    double* kgen = (double*)calloc(npar, sizeof(double));
    double* Tgen0 = (double*)malloc(npar * sizeof(double));
    for (int i = 0; i < npar; i++) {
        Tgen0[i] = 1.0;
    }
    double* Tgen = (double*)malloc(npar * sizeof(double));
    for (int i = 0; i < npar; i++) {
        Tgen[i] = Tgen0[i];
    }

    double c = 20.0;
    int Na = 1000;
    int Ngen = 10000;
    double* delta = (double*)malloc(npar * sizeof(double));
    for (int i = 0; i < npar; i++) {
        delta[i] = fabs(U[i] - L[i]) * 1e-8;
    }

    double acep = 0.25;
    int ntest = 100;
    double* ftest = (double*)calloc(ntest, sizeof(double));

    for (int j = 0; j < ntest; j++) {
        double xnew[npar];
        generate_xnew_from_generation_temperatures(x, L, U, Tgen, npar, xnew);
        get_chi2_astrometry(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err, xnew[0], xnew[1], xnew[2], chi2_array);
        ftest[j] = chi2_array[0] - 2.0 * log_beta_prior(xnew[2], ecc_a, ecc_b);
        if (ftest[j] < fbest) {
            fbest = ftest[j];
            for (int i = 0; i < npar; i++) {
                xbest[i] = xnew[i];
            }
        }
    }

    double* dftest = (double*)malloc((ntest - 1) * sizeof(double));
    for (int i = 0; i < ntest - 1; i++) {
        dftest[i] = ftest[i + 1] - ftest[i];
    }

    double avdftest = 0.0;
    for (int i = 0; i < ntest - 1; i++) {
        avdftest += fabs(dftest[i]);
    }
    avdftest /= (ntest - 1);

    Ta0 = avdftest / log(1.0 / acep - 1.0);

    int repeat = 1;
    while (repeat) {
        for (int j = 0; j < Ngen; j++) {
            int flag_acceptance = 0;
            double xnew[npar];
            generate_xnew_from_generation_temperatures(x, L, U, Tgen, npar, xnew);
            get_chi2_astrometry(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err, xnew[0], xnew[1], xnew[2], chi2_array);
            double fnew = chi2_array[0] - 2.0 * log_beta_prior(xnew[2], ecc_a, ecc_b);

            if (fnew <= f) {
                flag_acceptance = 1;
            } else {
                double test = (fnew - f) / Ta;
                double Pa = (test > 20) ? 0 : 1.0 / (1.0 + exp(test));
                if (random_double() <= Pa) {
                    flag_acceptance = 1;
                }
            }

            if (flag_acceptance) {
                if (fnew < fbest) {
                    fbest = fnew;
                    for (int i = 0; i < npar; i++) {
                        xbest[i] = xnew[i];
                    }
                    nrean = 0;
                }
                nacep++;
                ka++;
                for (int i = 0; i < npar; i++) {
                    x[i] = xnew[i];
                }
                f = fnew;
            }

            if (nacep >= Na) {
                double* s = (double*)calloc(npar, sizeof(double));
                for (int g = 0; g < npar; g++) {
                    double xdelta[npar];

                    for (int i = 0; i < npar; i++) {
                        xdelta[i] = xbest[i];
                    }

                    xdelta[g] += delta[g];

                    for (int i = 0; i < npar; i++) {
                        if (xdelta[i] <= L[i]) {
                            xdelta[i] = L[i] + delta[i];
                        }
                        if (xdelta[i] >= U[i]) {
                            xdelta[i] = U[i] - delta[i];
                        }
                    }

                    get_chi2_astrometry(
                        n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err,
                        xdelta[0], xdelta[1], xdelta[2], chi2_array
                    );

                    double fbestdelta = chi2_array[0] - 2.0 * log_beta_prior(xdelta[2], ecc_a, ecc_b);
                    s[g] = fabs((fbestdelta - fbest) / delta[g]);
                }
                
                
                for (int i = 0; i < npar; i++) {
                    if (s[i] == 0.0) {
                        double min_nonzero = 1e10;
                        for (int j = 0; j < npar; j++) {
                            if (s[j] != 0.0 && s[j] < min_nonzero) {
                                min_nonzero = s[j];
                            }
                        }
                        s[i] = min_nonzero;
                    }
                }

                double smax = 0.0;
                for (int i = 0; i < npar; i++) {
                    if (s[i] > smax) {
                        smax = s[i];
                    }
                }

                for (int i = 0; i < npar; i++) {
                    Tgen[i] = Tgen[i] * (smax / s[i]);
                    kgen[i] = pow(log(Tgen0[i] / Tgen[i]) / c, npar);
                    kgen[i] = fabs(kgen[i]);
                }

                Ta0 = f;
                Ta = fbest;
                ka = pow(log(Ta0 / Ta) / c, npar);

                int stop = 0;
                for (int i = 0; i < npar; i++) {
                    if (Tgen[i] == 0) {
                        stop = 1;
                        break;
                    }
                }
                if (stop) {
                    repeat = 0;
                    break;
                }

                nacep = 0;
                nrean++;
                free(s);
            }
        }

        for (int i = 0; i < npar; i++) {
            kgen[i] += 1;
            Tgen[i] = Tgen0[i] * exp(-c * pow(kgen[i], 1.0 / npar));
        }

        ka += 1;
        Ta = Ta0 * exp(-c * pow(ka, 1.0 / npar));

        for (int i = 1; i < Neps; i++) {
            fbestlist[i - 1] = fbestlist[i];
        }
        fbestlist[Neps - 1] = fbest;

        int count = 0;
        for (int i = 0; i < Neps - 1; i++) {
            if (fabs(fbestlist[i + 1] - fbestlist[i]) < eps) {
                count++;
            }
        }
        if (count == Neps - 1) {
            // printf("%f %f\n", fbest, chislimit);
            if (fbest < chislimit) {
                repeat = 0;
            } else {
                Ta = Ta0;
            }
        }
        if (nrean >= Nterm) {
            repeat = 0;
        }
    }

    free(fbestlist);
    free(kgen);
    free(Tgen0);
    free(Tgen);
    free(delta);
    free(ftest);
    free(dftest);
    free(x);
    free(chi2_array);

    for (int i = 0; i < npar; i++) {
        results[i] = xbest[i];
    }

    free(xbest);
}



/* This is identical to run_astfit(), except that it includes a parallax prior of parallax = pi0 +/- sig_pi
*/ 

void run_astfit_parallax_prior(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, double* results, int n_obs, double pi0, double sig_pi) {
    double eps = 1e-5; // all the tunable parameters are taken from RVFIT ( https://arxiv.org/abs/1505.04767)
    int Neps = 5;
    int Nterm = 20;
    int npar = 3;

    double* fbestlist = (double*)malloc(Neps * sizeof(double));
    for (int i = 0; i < Neps; i++) {
        fbestlist[i] = 1.0;
    }
    int nrean = 0;

    double mean_err = 0.0;
    for (int i = 0; i < n_obs; i++) {
        mean_err += ast_err[i];
    }
    mean_err /= n_obs;

    double chislimit = 0.0;
    for (int i = 0; i < n_obs; i++) {
        chislimit += pow(mean_err / ast_err[i], 2);
    }

    double* x = (double*)malloc(npar * sizeof(double));
    for (int i = 0; i < npar; i++) {
        x[i] = (U[i] + L[i]) / 2.0;
    }

    double* chi2_array = (double*)malloc(10 * sizeof(double));
    get_chi2_astrometry_parallax_prior(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err, x[0], x[1], x[2], pi0, sig_pi, chi2_array);
    double f = chi2_array[0];
    double* xbest = (double*)malloc(npar * sizeof(double));
    double fbest = f;
    for (int i = 0; i < npar; i++) {
        xbest[i] = x[i];
    }

    int ka = 0;
    double Ta0 = f;
    double Ta = Ta0;
    int nacep = 0;

    double* kgen = (double*)calloc(npar, sizeof(double));
    double* Tgen0 = (double*)malloc(npar * sizeof(double));
    for (int i = 0; i < npar; i++) {
        Tgen0[i] = 1.0;
    }
    double* Tgen = (double*)malloc(npar * sizeof(double));
    for (int i = 0; i < npar; i++) {
        Tgen[i] = Tgen0[i];
    }

    double c = 20.0;
    int Na = 1000;
    int Ngen = 10000;
    double* delta = (double*)malloc(npar * sizeof(double));
    for (int i = 0; i < npar; i++) {
        delta[i] = fabs(U[i] - L[i]) * 1e-8;
    }

    double acep = 0.25;
    int ntest = 100;
    double* ftest = (double*)calloc(ntest, sizeof(double));

    for (int j = 0; j < ntest; j++) {
        double xnew[npar];
        generate_xnew_from_generation_temperatures(x, L, U, Tgen, npar, xnew);
        get_chi2_astrometry_parallax_prior(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err, xnew[0], xnew[1], xnew[2], pi0, sig_pi, chi2_array);
        ftest[j] = chi2_array[0];
        if (ftest[j] < fbest) {
            fbest = ftest[j];
            for (int i = 0; i < npar; i++) {
                xbest[i] = xnew[i];
            }
        }
    }

    double* dftest = (double*)malloc((ntest - 1) * sizeof(double));
    for (int i = 0; i < ntest - 1; i++) {
        dftest[i] = ftest[i + 1] - ftest[i];
    }

    double avdftest = 0.0;
    for (int i = 0; i < ntest - 1; i++) {
        avdftest += fabs(dftest[i]);
    }
    avdftest /= (ntest - 1);

    Ta0 = avdftest / log(1.0 / acep - 1.0);

    int repeat = 1;
    while (repeat) {
        for (int j = 0; j < Ngen; j++) {
            int flag_acceptance = 0;
            double xnew[npar];
            generate_xnew_from_generation_temperatures(x, L, U, Tgen, npar, xnew);
            get_chi2_astrometry_parallax_prior(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err, xnew[0], xnew[1], xnew[2], pi0, sig_pi, chi2_array);
            double fnew = chi2_array[0];

            if (fnew <= f) {
                flag_acceptance = 1;
            } else {
                double test = (fnew - f) / Ta;
                double Pa = (test > 20) ? 0 : 1.0 / (1.0 + exp(test));
                if (random_double() <= Pa) {
                    flag_acceptance = 1;
                }
            }

            if (flag_acceptance) {
                if (fnew < fbest) {
                    fbest = fnew;
                    for (int i = 0; i < npar; i++) {
                        xbest[i] = xnew[i];
                    }
                    nrean = 0;
                }
                nacep++;
                ka++;
                for (int i = 0; i < npar; i++) {
                    x[i] = xnew[i];
                }
                f = fnew;
            }

            if (nacep >= Na) {
                double* s = (double*)calloc(npar, sizeof(double));
                for (int g = 0; g < npar; g++) {
                    double xdelta[npar];

                    for (int i = 0; i < npar; i++) {
                        xdelta[i] = xbest[i];
                    }

                    xdelta[g] += delta[g];

                    for (int i = 0; i < npar; i++) {
                        if (xdelta[i] <= L[i]) {
                            xdelta[i] = L[i] + delta[i];
                        }
                        if (xdelta[i] >= U[i]) {
                            xdelta[i] = U[i] - delta[i];
                        }
                    }

                    get_chi2_astrometry_parallax_prior(
                        n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err,
                        xdelta[0], xdelta[1], xdelta[2],
                        pi0, sig_pi, chi2_array
                    );

                    double fbestdelta = chi2_array[0];
                    s[g] = fabs((fbestdelta - fbest) / delta[g]);
                }
                


                for (int i = 0; i < npar; i++) {
                    if (s[i] == 0.0) {
                        double min_nonzero = 1e10;
                        for (int j = 0; j < npar; j++) {
                            if (s[j] != 0.0 && s[j] < min_nonzero) {
                                min_nonzero = s[j];
                            }
                        }
                        s[i] = min_nonzero;
                    }
                }

                double smax = 0.0;
                for (int i = 0; i < npar; i++) {
                    if (s[i] > smax) {
                        smax = s[i];
                    }
                }

                for (int i = 0; i < npar; i++) {
                    Tgen[i] = Tgen[i] * (smax / s[i]);
                    kgen[i] = pow(log(Tgen0[i] / Tgen[i]) / c, npar);
                    kgen[i] = fabs(kgen[i]);
                }

                Ta0 = f;
                Ta = fbest;
                ka = pow(log(Ta0 / Ta) / c, npar);

                int stop = 0;
                for (int i = 0; i < npar; i++) {
                    if (Tgen[i] == 0) {
                        stop = 1;
                        break;
                    }
                }
                if (stop) {
                    repeat = 0;
                    break;
                }

                nacep = 0;
                nrean++;
                free(s);
            }
        }

        for (int i = 0; i < npar; i++) {
            kgen[i] += 1;
            Tgen[i] = Tgen0[i] * exp(-c * pow(kgen[i], 1.0 / npar));
        }

        ka += 1;
        Ta = Ta0 * exp(-c * pow(ka, 1.0 / npar));

        for (int i = 1; i < Neps; i++) {
            fbestlist[i - 1] = fbestlist[i];
        }
        fbestlist[Neps - 1] = fbest;

        int count = 0;
        for (int i = 0; i < Neps - 1; i++) {
            if (fabs(fbestlist[i + 1] - fbestlist[i]) < eps) {
                count++;
            }
        }
        if (count == Neps - 1) {
            // printf("%f %f\n", fbest, chislimit);
            if (fbest < chislimit) {
                repeat = 0;
            } else {
                Ta = Ta0;
            }
        }
        if (nrean >= Nterm) {
            repeat = 0;
        }
    }

    free(fbestlist);
    free(kgen);
    free(Tgen0);
    free(Tgen);
    free(delta);
    free(ftest);
    free(dftest);
    free(x);
    free(chi2_array);

    for (int i = 0; i < npar; i++) {
        results[i] = xbest[i];
    }

    free(xbest);
}


/* Single-pass exact Schur-complement chi2 without outlier rejection or residual materialization.
   Uses precomputed per-epoch products in GridWorkspace (W, s2, sc, c2, rr0) and exact orvara
   Kepler solve; returns 1 on success, 0 if S is ill-conditioned or chi2 << rr0 (cancellation risk). */
static int ws_chi2_schur_nodrop(const GridWorkspace* g, const AstfitWorkspace* w, double P, double phi_p, double ecc, double* chi2_out) {
    int n = g->n;
    double sq = sqrt(1.0 - ecc * ecc);
    double aX[12] = {0}, aY[12] = {0}, h[9] = {0};
    KeplerOrvaraTable kt;
    if (ecc > 0.0) kepler_orvara_table(&kt, ecc);
    double invP = 1.0 / P;
    const double* a_arr = w->a;
    const double* W = g->W;
    const double* s2 = g->s2;
    const double* sc = g->sc;
    const double* c2 = g->c2;
    for (int j = 0; j < n; j++) {
        double cE, sE, M = a_arr[j] * invP - phi_p;
        if (ecc > 0.0) kepler_cos_sin_orvara(&kt, M, &cE, &sE);
        else { cE = cos(M); sE = sin(M); }
        double X = cE - ecc, Y = sq * sE;
        const double* wj = W + 12 * j;
        for (int m = 0; m < 12; m++) {
            aX[m] += X * wj[m];
            aY[m] += Y * wj[m];
        }
        double XX = X * X, XY = X * Y, YY = Y * Y;
        h[0] += XX * s2[j]; h[1] += XY * s2[j]; h[2] += YY * s2[j];
        h[3] += XX * sc[j]; h[4] += XY * sc[j]; h[5] += YY * sc[j];
        h[6] += XX * c2[j]; h[7] += XY * c2[j]; h[8] += YY * c2[j];
    }
    double H[4][4] = {
        {h[0], h[1], h[3], h[4]},
        {h[1], h[2], h[4], h[5]},
        {h[3], h[4], h[6], h[7]},
        {h[4], h[5], h[7], h[8]}
    };
    double G[5][4], b[4] = {aX[10], aY[10], aX[11], aY[11]};
    for (int k = 0; k < 5; k++) {
        G[k][0] = aX[k]; G[k][1] = aY[k]; G[k][2] = aX[5 + k]; G[k][3] = aY[5 + k];
    }
    double Lc[4][4], z[4], drop = 0.0;
    for (int c = 0; c < 4; c++) {
        for (int d = 0; d <= c; d++) {
            double v = H[c][d];
            for (int k = 0; k < 5; k++) v -= G[k][c] * G[k][d];
            for (int m = 0; m < d; m++) v -= Lc[c][m] * Lc[d][m];
            if (d == c) {
                if (!(H[c][c] > 0.0) || !(v > 1e-4 * H[c][c])) return 0;
                Lc[c][c] = sqrt(v);
            } else {
                Lc[c][d] = v / Lc[d][d];
            }
        }
        double v = b[c];
        for (int m = 0; m < c; m++) v -= Lc[c][m] * z[m];
        z[c] = v / Lc[c][c];
        drop += z[c] * z[c];
    }
    double chi2 = g->rr0 - drop;
    if (!(chi2 >= 1e-6 * g->rr0) || !isfinite(chi2)) return 0;
    *chi2_out = chi2;
    return 1;
}

/* This is a Joker-like approach, which is useful at low SNR.  n_samp is length of the samples array; n_obs is length of the observations array. This populates the lnL_array with likelihoods corresponding to the arrays P, phi_p, ecc.
lnL = -chi2/2 at the best-fit linear parameters, evaluated with the same fast chi2 as the optimizers.
*/
void get_likelihoods_astrometry(int n_obs, int n_samp, double *t_ast_yr, double *psi, double *plx_factor, double *ast_obs, double *ast_err, double *P, double *phi_p, double *ecc, double *lnL_array){
    if (n_samp <= 0) return;
    AstfitWorkspace* w = astfit_workspace_create(t_ast_yr, psi, plx_factor, ast_obs, ast_err, n_obs, 0);
    GridWorkspace* g = grid_workspace_alloc(w);
    for (int i = 0; i < n_samp; i++) {
        double chi2;
        if (!ws_chi2_schur_nodrop(g, w, P[i], phi_p[i], ecc[i], &chi2)) {
            int nor;
            chi2 = ws_chi2_all(w, P[i], phi_p[i], ecc[i], &nor);
        }
        lnL_array[i] = -0.5 * chi2;
    }
    grid_workspace_free(g);
    astfit_workspace_free(w);
}

/* The previous implementation of get_likelihoods_astrometry() (GSL normal equations); kept for testing. */
void get_likelihoods_astrometry_gsl(int n_obs, int n_samp, double *t_ast_yr, double *psi, double *plx_factor, double *ast_obs, double *ast_err, double *P, double *phi_p, double *ecc, double *lnL_array){
    double xtol = 1e-10;
    double Ei;
    double Mi;
    double X[n_obs];
    double Y[n_obs]; 
    double ivar[n_obs];
    
    gsl_matrix *M = gsl_matrix_calloc(n_obs, 9);
    gsl_matrix *Mt = gsl_matrix_alloc(9, n_obs);
    
    gsl_matrix *Cinv = gsl_matrix_calloc(n_obs, n_obs); // Allocate and initialize to zero

    gsl_matrix *MtCinvM = gsl_matrix_alloc(9, 9); // Result of Mt * Cinv * M
    gsl_vector *MtCinvY = gsl_vector_alloc(9); // Result of Mt * Cinv * ast_obs
    gsl_vector *mu = gsl_vector_alloc(9); // Solution vector
    gsl_vector *Lambda_pred = gsl_vector_alloc(n_obs); // Predicted lambda
    gsl_matrix *temp = gsl_matrix_alloc(9, n_obs); // Temporary matrix for intermediate calculations
    gsl_vector *Cinv_ast_obs = gsl_vector_alloc(n_obs); 
    gsl_vector_view ast_obs_view = gsl_vector_view_array(ast_obs, n_obs); 
    

    // Compute ivar and fill Cinv
    for (int j = 0; j < n_obs; j++) {
        ivar[j] = 1.0 / (ast_err[j] * ast_err[j]); // Compute inverse variance
        gsl_matrix_set(Cinv, j, j, ivar[j]); // Set diagonal elements of Cinv
    }    
    
    for(int i = 0; i < n_samp; i++) {
        for(int j = 0; j < n_obs; j++){
            Mi = 2*PI*t_ast_yr[j]*365.25/P[i] - phi_p[i];
            Ei = solve_Kepler_equation(Mi, ecc[i], xtol);
            
            X[j] = cos(Ei) - ecc[i];
            Y[j] = sqrt(1-ecc[i]*ecc[i])*sin(Ei);
        }
        
        // Fill the matrix M 
        for (int j = 0; j < n_obs; j++) {
            gsl_matrix_set(M, j, 0, sin(psi[j]) );
            gsl_matrix_set(M, j, 1, t_ast_yr[j] * sin(psi[j]) );
            gsl_matrix_set(M, j, 2, cos(psi[j]) );
            gsl_matrix_set(M, j, 3, t_ast_yr[j] * cos(psi[j]));
            gsl_matrix_set(M, j, 4, plx_factor[j]);
            gsl_matrix_set(M, j, 5, X[j] * sin(psi[j]) );
            gsl_matrix_set(M, j, 6, Y[j] * sin(psi[j]) );
            gsl_matrix_set(M, j, 7, X[j] * cos(psi[j]));
            gsl_matrix_set(M, j, 8, Y[j] * cos(psi[j]));
        }
        
        gsl_matrix_transpose_memcpy(Mt, M);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Mt, Cinv, 0.0, temp);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, temp, M, 0.0, MtCinvM);
        
        // Compute Mt * Cinv * ast_obs
        // Perform the multiplication Cinv * ast_obs; multiply Mt and the result of the first multiplication
        gsl_blas_dgemv(CblasNoTrans, 1.0, Cinv, &ast_obs_view.vector, 0.0, Cinv_ast_obs);
        gsl_blas_dgemv(CblasNoTrans, 1.0, Mt, Cinv_ast_obs, 0.0, MtCinvY);

        // Solve for mu
        gsl_linalg_HH_solve(MtCinvM, MtCinvY, mu);

        // Compute predicted values Lambda_pred = M * mu
        gsl_blas_dgemv(CblasNoTrans, 1.0, M, mu, 0.0, Lambda_pred);
        

        // Compute log-likelihood
        double lnL = 0.0; // Log likelihood
        for (int j = 0; j < n_obs; j++) {
            double pred = gsl_vector_get(Lambda_pred, j);
            double obs = ast_obs[j];
            double err = ast_err[j];
            lnL -= 0.5 * ((pred - obs) * (pred - obs)) / (err * err);
        }

        lnL_array[i] = lnL;
    }
    // Cleanup
    gsl_matrix_free(M);
    gsl_matrix_free(Mt);
    gsl_matrix_free(Cinv);
    gsl_matrix_free(MtCinvM);
    gsl_vector_free(MtCinvY);
    gsl_vector_free(mu);
    gsl_vector_free(Lambda_pred);
    gsl_matrix_free(temp);
    gsl_vector_free(Cinv_ast_obs);
}


/*  This function is like get_chi2_astrometry(), except that it ignores the worst data point */
void get_chi2_astrometry_reject_outliers(int n_obs, double *t_ast_yr, double *psi, double *plx_factor, double *ast_obs, double *ast_err, double P, double phi_p, double ecc, double *chi2_array){
    double xtol = 1e-10;
    double Ei;
    double Mi;
    double X[n_obs];
    double Y[n_obs]; 
    double ivar[n_obs];
    
    gsl_matrix *M = gsl_matrix_calloc(n_obs, 9);
    gsl_matrix *Mw = gsl_matrix_calloc(n_obs, 9);
    gsl_matrix *MtCinvM = gsl_matrix_alloc(9, 9); // Result of Mt * Cinv * M
    gsl_vector *MtCinvY = gsl_vector_alloc(9); // Result of Mt * Cinv * ast_obs
    gsl_vector *mu = gsl_vector_alloc(9); // Solution vector
    gsl_vector *Lambda_pred = gsl_vector_alloc(n_obs); // Predicted lambda
    gsl_vector *weighted_ast_obs = gsl_vector_alloc(n_obs);
    gsl_matrix *normal_matrix = gsl_matrix_alloc(9, 9);
    gsl_matrix *normal_inverse = gsl_matrix_alloc(9, 9);
    gsl_permutation *perm = gsl_permutation_alloc(9);

    for (int j = 0; j < n_obs; j++) {
        ivar[j] = 1.0 / (ast_err[j] * ast_err[j]);
    }    
    
    for(int j = 0; j < n_obs; j++){
        Mi = 2*PI*t_ast_yr[j]*365.25/P - phi_p;
        Ei = solve_Kepler_equation(Mi, ecc, xtol);
        
        X[j] = cos(Ei) - ecc;
        Y[j] = sqrt(1-ecc*ecc)*sin(Ei);
    }
      
    for (int j = 0; j < n_obs; j++) {
        gsl_matrix_set(M, j, 0, sin(psi[j]) );
        gsl_matrix_set(M, j, 1, t_ast_yr[j] * sin(psi[j]) );
        gsl_matrix_set(M, j, 2, cos(psi[j]) );
        gsl_matrix_set(M, j, 3, t_ast_yr[j] * cos(psi[j]));
        gsl_matrix_set(M, j, 4, plx_factor[j]);
        gsl_matrix_set(M, j, 5, X[j] * sin(psi[j]) );
        gsl_matrix_set(M, j, 6, Y[j] * sin(psi[j]) );
        gsl_matrix_set(M, j, 7, X[j] * cos(psi[j]));
        gsl_matrix_set(M, j, 8, Y[j] * cos(psi[j]));
        double inv_err = 1.0 / ast_err[j];
        for (int k = 0; k < 9; k++) {
            gsl_matrix_set(Mw, j, k, gsl_matrix_get(M, j, k) * inv_err);
        }
        gsl_vector_set(weighted_ast_obs, j, ast_obs[j] * inv_err);
    }

    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, Mw, Mw, 0.0, MtCinvM);
    gsl_blas_dgemv(CblasTrans, 1.0, Mw, weighted_ast_obs, 0.0, MtCinvY);

    gsl_matrix_memcpy(normal_matrix, MtCinvM);
    int signum;
    int lu_status = gsl_linalg_LU_decomp(normal_matrix, perm, &signum);
    if (lu_status == 0) {
        lu_status = gsl_linalg_LU_solve(normal_matrix, perm, MtCinvY, mu);
    }
    if (lu_status == 0) {
        lu_status = gsl_linalg_LU_invert(normal_matrix, perm, normal_inverse);
    }

    if (lu_status != 0) {
        double full_chi2_array[10];
        get_chi2_astrometry(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err, P, phi_p, ecc, full_chi2_array);

        double running_worst = 0.0;
        int worst_index = 0;
        for (int j = 0; j < n_obs; j++) {
            double pred = 0.0;
            for (int k = 0; k < 9; k++) {
                pred += gsl_matrix_get(M, j, k) * full_chi2_array[k + 1];
            }
            double resid_j = pred - ast_obs[j];
            double chi2_j = resid_j * resid_j * ivar[j];
            if (chi2_j > running_worst) {
                running_worst = chi2_j;
                worst_index = j;
            }
        }

        double* ast_err_new = (double*)malloc(n_obs * sizeof(double));
        for (int i = 0; i < n_obs; i++) {
            ast_err_new[i] = ast_err[i];
        }
        ast_err_new[worst_index] = 1e9; // turn off the worst data point
        get_chi2_astrometry(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err_new, P, phi_p, ecc, chi2_array);
        free(ast_err_new);

        gsl_matrix_free(M);
        gsl_matrix_free(Mw);
        gsl_matrix_free(MtCinvM);
        gsl_vector_free(MtCinvY);
        gsl_vector_free(mu);
        gsl_vector_free(Lambda_pred);
        gsl_vector_free(weighted_ast_obs);
        gsl_matrix_free(normal_matrix);
        gsl_matrix_free(normal_inverse);
        gsl_permutation_free(perm);
        return;
    }

    gsl_blas_dgemv(CblasNoTrans, 1.0, M, mu, 0.0, Lambda_pred);
    
    double chi2_j = 0.0; 
    double running_worst = 0.0; 
    int worst_index = 0;
    for (int j = 0; j < n_obs; j++) {
        chi2_j = ((gsl_vector_get(Lambda_pred, j) - ast_obs[j]) * (gsl_vector_get(Lambda_pred, j) - ast_obs[j])) / (ast_err[j] * ast_err[j]);
        if (chi2_j > running_worst) {
            running_worst = chi2_j;
            worst_index = j;
        }
    }

    double xworst[9];
    for (int k = 0; k < 9; k++) {
        xworst[k] = gsl_matrix_get(M, worst_index, k);
    }

    double Cx[9];
    for (int k = 0; k < 9; k++) {
        Cx[k] = 0.0;
        for (int l = 0; l < 9; l++) {
            Cx[k] += gsl_matrix_get(normal_inverse, k, l) * xworst[l];
        }
    }

    double leverage = 0.0;
    for (int k = 0; k < 9; k++) {
        leverage += xworst[k] * Cx[k];
    }
    leverage *= ivar[worst_index];

    double denom = 1.0 - leverage;
    if (denom <= 1e-10 || !isfinite(denom)) {
        double* ast_err_new = (double*)malloc(n_obs * sizeof(double));
        for (int i = 0; i < n_obs; i++) {
            ast_err_new[i] = ast_err[i];
        }
        ast_err_new[worst_index] = 1e9; // turn off the worst data point
        get_chi2_astrometry(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err_new, P, phi_p, ecc, chi2_array);
        free(ast_err_new);
    } else {
        double mu_reject[9];
        double residual = ast_obs[worst_index] - gsl_vector_get(Lambda_pred, worst_index);
        double scale = ivar[worst_index] * residual / denom;
        for (int k = 0; k < 9; k++) {
            mu_reject[k] = gsl_vector_get(mu, k) - Cx[k] * scale;
        }

        double chi2_reject = 0.0;
        for (int j = 0; j < n_obs; j++) {
            if (j == worst_index) {
                continue;
            }
            double pred = 0.0;
            for (int k = 0; k < 9; k++) {
                pred += gsl_matrix_get(M, j, k) * mu_reject[k];
            }
            double resid_j = pred - ast_obs[j];
            chi2_reject += resid_j * resid_j * ivar[j];
        }

        chi2_array[0] = chi2_reject;
        for (int k = 0; k < 9; k++) {
            chi2_array[k + 1] = mu_reject[k];
        }
    }
    
 
    // Cleanup
    gsl_matrix_free(M);
    gsl_matrix_free(Mw);
    gsl_matrix_free(MtCinvM);
    gsl_vector_free(MtCinvY);
    gsl_vector_free(mu);
    gsl_vector_free(Lambda_pred);
    gsl_vector_free(weighted_ast_obs);
    gsl_matrix_free(normal_matrix);
    gsl_matrix_free(normal_inverse);
    gsl_permutation_free(perm);
}

/* This is like run_astfit(), except it ignores the worst datapoint for each likelihood call. 
*/ 
void run_astfit_reject_outlier(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, double* results, int n_obs) {
    double eps = 1e-5; // all the tunable parameters are taken from RVFIT ( https://arxiv.org/abs/1505.04767)
    int Neps = 5;
    int Nterm = 20;
    int npar = 3;

    double* fbestlist = (double*)malloc(Neps * sizeof(double));
    for (int i = 0; i < Neps; i++) {
        fbestlist[i] = 1.0;
    }
    int nrean = 0;

    double mean_err = 0.0;
    for (int i = 0; i < n_obs; i++) {
        mean_err += ast_err[i];
    }
    mean_err /= n_obs;

    double chislimit = 0.0;
    for (int i = 0; i < n_obs; i++) {
        chislimit += pow(mean_err / ast_err[i], 2);
    }

    double* x = (double*)malloc(npar * sizeof(double));
    for (int i = 0; i < npar; i++) {
        x[i] = (U[i] + L[i]) / 2.0;
    }

    double* chi2_array = (double*)malloc(10 * sizeof(double));
    get_chi2_astrometry_reject_outliers(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err, x[0], x[1], x[2], chi2_array);
    double f = chi2_array[0];
    double* xbest = (double*)malloc(npar * sizeof(double));
    double fbest = f;
    for (int i = 0; i < npar; i++) {
        xbest[i] = x[i];
    }

    int ka = 0;
    double Ta0 = f;
    double Ta = Ta0;
    int nacep = 0;

    double* kgen = (double*)calloc(npar, sizeof(double));
    double* Tgen0 = (double*)malloc(npar * sizeof(double));
    for (int i = 0; i < npar; i++) {
        Tgen0[i] = 1.0;
    }
    double* Tgen = (double*)malloc(npar * sizeof(double));
    for (int i = 0; i < npar; i++) {
        Tgen[i] = Tgen0[i];
    }

    double c = 20.0;
    int Na = 1000;
    int Ngen = 10000;
    double* delta = (double*)malloc(npar * sizeof(double));
    for (int i = 0; i < npar; i++) {
        delta[i] = fabs(U[i] - L[i]) * 1e-8;
    }

    double acep = 0.25;
    int ntest = 100;
    double* ftest = (double*)calloc(ntest, sizeof(double));

    for (int j = 0; j < ntest; j++) {
        double xnew[npar];
        generate_xnew_from_generation_temperatures(x, L, U, Tgen, npar, xnew);
        get_chi2_astrometry_reject_outliers(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err, xnew[0], xnew[1], xnew[2], chi2_array);
        ftest[j] = chi2_array[0];
        if (ftest[j] < fbest) {
            fbest = ftest[j];
            for (int i = 0; i < npar; i++) {
                xbest[i] = xnew[i];
            }
        }
    }

    double* dftest = (double*)malloc((ntest - 1) * sizeof(double));
    for (int i = 0; i < ntest - 1; i++) {
        dftest[i] = ftest[i + 1] - ftest[i];
    }

    double avdftest = 0.0;
    for (int i = 0; i < ntest - 1; i++) {
        avdftest += fabs(dftest[i]);
    }
    avdftest /= (ntest - 1);

    Ta0 = avdftest / log(1.0 / acep - 1.0);

    int repeat = 1;
    while (repeat) {
        for (int j = 0; j < Ngen; j++) {
            int flag_acceptance = 0;
            double xnew[npar];
            generate_xnew_from_generation_temperatures(x, L, U, Tgen, npar, xnew);
            get_chi2_astrometry_reject_outliers(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err, xnew[0], xnew[1], xnew[2], chi2_array);
            double fnew = chi2_array[0];

            if (fnew <= f) {
                flag_acceptance = 1;
            } else {
                double test = (fnew - f) / Ta;
                double Pa = (test > 20) ? 0 : 1.0 / (1.0 + exp(test));
                if (random_double() <= Pa) {
                    flag_acceptance = 1;
                }
            }

            if (flag_acceptance) {
                if (fnew < fbest) {
                    fbest = fnew;
                    for (int i = 0; i < npar; i++) {
                        xbest[i] = xnew[i];
                    }
                    nrean = 0;
                }
                nacep++;
                ka++;
                for (int i = 0; i < npar; i++) {
                    x[i] = xnew[i];
                }
                f = fnew;
            }

            if (nacep >= Na) {
                double* s = (double*)calloc(npar, sizeof(double));
                
                for (int g = 0; g < npar; g++) {
                    double xdelta[npar];

                    for (int i = 0; i < npar; i++) {
                        xdelta[i] = xbest[i];
                    }
                    xdelta[g] += delta[g];
                    get_chi2_astrometry_reject_outliers(
                        n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err,
                        xdelta[0], xdelta[1], xdelta[2], chi2_array
                    );
                    double fbestdelta = chi2_array[0];
                    s[g] = fabs((fbestdelta - fbest) / delta[g]);
                }
                

                for (int i = 0; i < npar; i++) {
                    if (s[i] == 0.0) {
                        double min_nonzero = 1e10;
                        for (int j = 0; j < npar; j++) {
                            if (s[j] != 0.0 && s[j] < min_nonzero) {
                                min_nonzero = s[j];
                            }
                        }
                        s[i] = min_nonzero;
                    }
                }

                double smax = 0.0;
                for (int i = 0; i < npar; i++) {
                    if (s[i] > smax) {
                        smax = s[i];
                    }
                }

                for (int i = 0; i < npar; i++) {
                    Tgen[i] = Tgen[i] * (smax / s[i]);
                    kgen[i] = pow(log(Tgen0[i] / Tgen[i]) / c, npar);
                    kgen[i] = fabs(kgen[i]);
                }

                Ta0 = f;
                Ta = fbest;
                ka = pow(log(Ta0 / Ta) / c, npar);

                int stop = 0;
                for (int i = 0; i < npar; i++) {
                    if (Tgen[i] == 0) {
                        stop = 1;
                        break;
                    }
                }
                if (stop) {
                    repeat = 0;
                    break;
                }

                nacep = 0;
                nrean++;
                free(s);
            }
        }

        for (int i = 0; i < npar; i++) {
            kgen[i] += 1;
            Tgen[i] = Tgen0[i] * exp(-c * pow(kgen[i], 1.0 / npar));
        }

        ka += 1;
        Ta = Ta0 * exp(-c * pow(ka, 1.0 / npar));

        for (int i = 1; i < Neps; i++) {
            fbestlist[i - 1] = fbestlist[i];
        }
        fbestlist[Neps - 1] = fbest;

        int count = 0;
        for (int i = 0; i < Neps - 1; i++) {
            if (fabs(fbestlist[i + 1] - fbestlist[i]) < eps) {
                count++;
            }
        }
        if (count == Neps - 1) {
            // printf("%f %f\n", fbest, chislimit);
            if (fbest < chislimit) {
                repeat = 0;
            } else {
                Ta = Ta0;
            }
        }
        if (nrean >= Nterm) {
            repeat = 0;
        }
    }

    free(fbestlist);
    free(kgen);
    free(Tgen0);
    free(Tgen);
    free(delta);
    free(ftest);
    free(dftest);
    free(x);
    free(chi2_array);

    for (int i = 0; i < npar; i++) {
        results[i] = xbest[i];
    }

    free(xbest);
}
