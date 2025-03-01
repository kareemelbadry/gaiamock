/* Helper functions for quickly solving Kepler's equation and some other utilities for fitting astrometric binaries. 
This needs to be compiled before the package can be used, and gsl needs to be linked for it to be compiled. On my Macbook, I installed gsl using homebrew and then compiled with: 
 gcc -shared -o kepler_solve_astrometry.so kepler_solve_astrometry.c -I/opt/homebrew/Cellar/gsl/2.8/include  -L/opt/homebrew/Cellar/gsl/2.8/lib -lgsl -lgslcblas -lm 

On my local cluster, I compiled with:
 gcc -shared -o kepler_solve_astrometry.so kepler_solve_astrometry.c -lgsl -lgslcblas -lm -fPIC*/

#include <math.h>
#include <stdlib.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
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
    double EE = KeplerStart3(ecc, Mnorm);
    while(eps > xtol){
        EE0 = EE;
        EE = EE0 - eps3(ecc, Mnorm, EE0);
        eps = fabs(EE0 - EE);
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
    double ivar[n_obs];
    
    gsl_matrix *M = gsl_matrix_calloc(n_obs, 9);
    gsl_matrix *Mt = gsl_matrix_alloc(9, n_obs);
    
    gsl_matrix *Cinv = gsl_matrix_calloc(n_obs, n_obs);
    gsl_matrix *C = gsl_matrix_calloc(n_obs, n_obs); 
    gsl_matrix *MtCinvM = gsl_matrix_alloc(9, 9); // Result of Mt * Cinv * M
    gsl_vector *MtCinvY = gsl_vector_alloc(9); // Result of Mt * Cinv * ast_obs
    gsl_vector *mu = gsl_vector_alloc(9); // Solution vector
    gsl_vector *Lambda_pred = gsl_vector_alloc(n_obs); // Predicted lambda
    gsl_matrix *temp = gsl_matrix_alloc(9, n_obs); // Temporary matrix for intermediate calculations

    // Compute ivar and fill Cinv and C
    for (int j = 0; j < n_obs; j++) {
        ivar[j] = 1.0 / (ast_err[j] * ast_err[j]); // Compute inverse variance
        gsl_matrix_set(Cinv, j, j, ivar[j]); // Set diagonal elements of Cinv
        gsl_matrix_set(C, j, j, 1.0 / ivar[j]); // Set diagonal elements of C
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
    gsl_matrix_free(Mt);
    gsl_matrix_free(Cinv);
    gsl_matrix_free(C);
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
    gsl_matrix *C = gsl_matrix_calloc(n_obs, n_obs); 
    gsl_matrix *MtCinvM = gsl_matrix_alloc(9, 9); // Result of Mt * Cinv * M
    gsl_vector *MtCinvY = gsl_vector_alloc(9); // Result of Mt * Cinv * ast_obs
    gsl_vector *mu = gsl_vector_alloc(9); // Solution vector
    gsl_vector *Lambda_pred = gsl_vector_alloc(n_obs); // Predicted lambda
    gsl_matrix *temp = gsl_matrix_alloc(9, n_obs); // Temporary matrix for intermediate calculations
    

    // Compute ivar and fill Cinv and C
    for (int j = 0; j < n_obs; j++) {
        ivar[j] = 1.0 / (ast_err[j] * ast_err[j]); // Compute inverse variance
        gsl_matrix_set(Cinv, j, j, ivar[j]); // Set diagonal elements of Cinv
        gsl_matrix_set(C, j, j, 1.0 / ivar[j]); // Set diagonal elements of C
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
    gsl_matrix_free(C);
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
                    double* ee = (double*)calloc(npar, sizeof(double));
                    ee[g] = delta[g];
                    get_chi2_astrometry(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err, xnew[0], xnew[1], xnew[2], chi2_array);
                    double fbestdelta = chi2_array[0];
                    s[g] = fabs((fbestdelta - fbest) / delta[g]);
                    free(ee);
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
    gsl_matrix *C = gsl_matrix_calloc(n_obs, n_obs); // Allocate and initialize to zero

    gsl_matrix *MtCinvM = gsl_matrix_alloc(9, 9); // Result of Mt * Cinv * M
    gsl_vector *MtCinvY = gsl_vector_alloc(9); // Result of Mt * Cinv * ast_obs
    gsl_vector *mu = gsl_vector_alloc(9); // Solution vector
    gsl_vector *Lambda_pred = gsl_vector_alloc(n_obs); // Predicted lambda
    gsl_matrix *temp = gsl_matrix_alloc(9, n_obs); // Temporary matrix for intermediate calculations
    

    // Compute ivar and fill Cinv and C
    for (int j = 0; j < n_obs; j++) {
        ivar[j] = 1.0 / (ast_err[j] * ast_err[j]); // Compute inverse variance
        gsl_matrix_set(Cinv, j, j, ivar[j]); // Set diagonal elements of Cinv
        gsl_matrix_set(C, j, j, 1.0 / ivar[j]); // Set diagonal elements of C
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
        gsl_vector *Cinv_ast_obs = gsl_vector_alloc(n_obs); 
        gsl_vector_view ast_obs_view = gsl_vector_view_array(ast_obs, n_obs); 
        
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
    gsl_matrix_free(C);
    gsl_matrix_free(MtCinvM);
    gsl_vector_free(MtCinvY);
    gsl_vector_free(mu);
    gsl_vector_free(Lambda_pred);
    gsl_matrix_free(temp);
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
    gsl_matrix *Mt = gsl_matrix_alloc(9, n_obs);
    
    gsl_matrix *Cinv = gsl_matrix_calloc(n_obs, n_obs);
    gsl_matrix *C = gsl_matrix_calloc(n_obs, n_obs); 
    gsl_matrix *MtCinvM = gsl_matrix_alloc(9, 9); // Result of Mt * Cinv * M
    gsl_vector *MtCinvY = gsl_vector_alloc(9); // Result of Mt * Cinv * ast_obs
    gsl_vector *mu = gsl_vector_alloc(9); // Solution vector
    gsl_vector *Lambda_pred = gsl_vector_alloc(n_obs); // Predicted lambda
    gsl_matrix *temp = gsl_matrix_alloc(9, n_obs); // Temporary matrix for intermediate calculations

    // Compute ivar and fill Cinv and C
    for (int j = 0; j < n_obs; j++) {
        ivar[j] = 1.0 / (ast_err[j] * ast_err[j]); // Compute inverse variance
        gsl_matrix_set(Cinv, j, j, ivar[j]); // Set diagonal elements of Cinv
        gsl_matrix_set(C, j, j, 1.0 / ivar[j]); // Set diagonal elements of C
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
    
    double chi2 = 0.0; 
    double running_worst = 0.0; 
    int worst_index = 0;
    for (int j = 0; j < n_obs; j++) {
        chi2 = ((gsl_vector_get(Lambda_pred, j) - ast_obs[j]) * (gsl_vector_get(Lambda_pred, j) - ast_obs[j])) / (ast_err[j] * ast_err[j]);
        if (chi2 > running_worst) {
            running_worst = chi2;
            worst_index = j;
        }
    }
    
    // make a new error vector 
    double* ast_err_new = (double*)malloc(n_obs * sizeof(double));
    for (int i = 0; i < n_obs; i++) {
        ast_err_new[i] = ast_err[i];
    }
    ast_err_new[worst_index] = 1e9; // turn off the worst data point 

    get_chi2_astrometry(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err_new, P, phi_p, ecc, chi2_array);
    
 
    // Cleanup
    gsl_matrix_free(M);
    gsl_matrix_free(Mt);
    gsl_matrix_free(Cinv);
    gsl_matrix_free(C);
    gsl_matrix_free(MtCinvM);
    gsl_vector_free(MtCinvY);
    gsl_vector_free(mu);
    gsl_vector_free(Lambda_pred);
    gsl_matrix_free(temp);
    gsl_vector_free(Cinv_ast_obs);
    free(ast_err_new);
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
                    double* ee = (double*)calloc(npar, sizeof(double));
                    ee[g] = delta[g];
                    get_chi2_astrometry_reject_outliers(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err, xnew[0], xnew[1], xnew[2], chi2_array);
                    double fbestdelta = chi2_array[0];
                    s[g] = fabs((fbestdelta - fbest) / delta[g]);
                    free(ee);
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

