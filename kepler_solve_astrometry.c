/* Helper functions for quickly solving Kepler's equation and some other utilities for fitting astrometric binaries. 
This needs to be compiled before the package can be used, and gsl needs to be linked for it to be compiled. On my Macbook, I installed gsl using homebrew and then compiled with: 
 gcc -shared -o kepler_solve_astrometry.so kepler_solve_astrometry.c -I/opt/homebrew/Cellar/gsl/2.8/include  -L/opt/homebrew/Cellar/gsl/2.8/lib -lgsl -lgslcblas -lm 

On my local cluster, I compiled with:
 gcc -shared -o kepler_solve_astrometry.so kepler_solve_astrometry.c -lgsl -lgslcblas -lm -fPIC */

#include <math.h>
#include <stdlib.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_errno.h>
#define PI 3.14159265358979


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

void polish_astfit_start_nelder_mead(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, int n_obs, int reject_outlier, double* x) {
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

    for (int iter = 0; iter < 400; iter++) {
        sort_simplex4(simplex, f);
        double fspread = fabs(f[3] - f[0]);
        double xspread = 0.0;
        for (int i = 1; i < m; i++) {
            for (int k = 0; k < n; k++) {
                double dx = fabs(simplex[i][k] - simplex[0][k]);
                if (dx > xspread) xspread = dx;
            }
        }
        if (fspread < 1e-6 && xspread < 1e-7) {
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
    int n_period = 260;
    int n_phase = 8;
    int n_short_period = 800;
    int n_ecc_phase = 48;
    int max_periods = 32;
    int max_candidates = 256;
    int max_starts = 64;
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
    for (int i = 0; i < n_rows && n_periods < 24; i++) {
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
void run_astfit_periodogram_multistart(double* t_ast_yr, double* psi, double *plx_factor, double* ast_obs, double* ast_err, double* L, double* U, double* results, int n_obs, int reject_outlier) {
    int n_period = 260;
    int n_short_period = 800;
    int n_ecc_phase = 48;
    int max_periods = 32;
    int max_candidates = 256;
    int max_starts = 64;
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
    for (int i = 0; i < n_rows && n_periods < 24; i++) {
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

    free(periodogram_rows);
    free(periods);
    free(candidates);
    free(starts);
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


/* This is a Joker-like approach, which is useful at low SNR.  n_samp is length of the samples array; n_obs is length of the observations array. This populates the lnL_array with likelihoods corresponding to the arrays P, phi_p, ecc. 
*/
void get_likelihoods_astrometry(int n_obs, int n_samp, double *t_ast_yr, double *psi, double *plx_factor, double *ast_obs, double *ast_err, double *P, double *phi_p, double *ecc, double *lnL_array){
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
