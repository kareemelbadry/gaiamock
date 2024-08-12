#!/usr/bin/env python

#SBATCH --time=3:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --exclusive
#SBATCH --constraint=icelake
#SBATCH -J "example-job"   # job name
#SBATCH --output=output_detprob.txt

# this is an example bash submission script to generate the data plotted in Fig 13 of the paper. 
import gaiamock
import numpy as np
import os

N_grid = 50
m1 = 0.93
q = 10
m2 = m1*q
Mg1 = 4.5
Mg2 = 100
f = 10**(-0.4*(Mg2-Mg1))
dist_range_bh1 = np.linspace(0.1, 1.0, N_grid)
ecc = 0.43
pmra, pmdec, period = -7.70205, -25.8504, 185.3
Mg_tot = -2.5*np.log10(10**(-Mg1/2.5) + 10**(-Mg2/2.5))
N = 500
dr = 'dr3'

all_prob_BH1 = []
for dist in dist_range_bh1:
    ra, dec, phot_g_mean_mag, Tp, omega, w, inc_deg, accept = gaiamock.simulate_many_realizations_of_a_single_binary(dist_pc = dist*1000, period = period, Mg_tot = Mg_tot, 
                    f = f, m1 = m1, m2 = m2, ecc = ecc, N_realizations = N, data_release=dr)
    prob = np.sum(accept)/len(accept)
    print('found probability!', dist, prob)
    all_prob_BH1.append(prob)

m1 = 1.0
q = 9
m2 = m1*q
Mg1 = 1.38
Mg2 = 100
f = 10**(-0.4*(Mg2-Mg1))
dist_range_bh2 = np.linspace(0.1, 4.0, N_grid)
ecc = 0.52
pmra, pmdec, period = -10.48, -4.61, 1277
Mg_tot = -2.5*np.log10(10**(-Mg1/2.5) + 10**(-Mg2/2.5))
N = 500

all_prob_BH2 = []
for dist in dist_range_bh2:
    ra, dec, phot_g_mean_mag, Tp, omega, w, inc_deg, accept = gaiamock.simulate_many_realizations_of_a_single_binary(dist_pc = dist*1000, period = period, Mg_tot = Mg_tot, 
                    f = f, m1 = m1, m2 = m2, ecc = ecc, N_realizations = N, data_release=dr)
    prob = np.sum(accept)/len(accept)
    print('found probability!', dist, prob)
    all_prob_BH2.append(prob)


m1 = 0.78
q = 2.43
m2 = m1*q
Mg1 = 3.75
Mg2 = 100
f = 10**(-0.4*(Mg2-Mg1))
dist_range_ns1 = np.linspace(0.1, 2.0, N_grid)
ecc = 0.12
pmra, pmdec, period = -0.01, -92.36, 731
Mg_tot = -2.5*np.log10(10**(-Mg1/2.5) + 10**(-Mg2/2.5))
N = 500

all_prob_NS1 = []
for dist in dist_range_ns1:
    ra, dec, phot_g_mean_mag, Tp, omega, w, inc_deg, accept = gaiamock.simulate_many_realizations_of_a_single_binary(dist_pc = dist*1000, period = period, Mg_tot = Mg_tot, 
                    f = f, m1 = m1, m2 = m2, ecc = ecc, N_realizations = N, data_release=dr)
    prob = np.sum(accept)/len(accept)
    print('found probability!', dist, prob)
    all_prob_NS1.append(prob)


m1 = 0.78
q = 42.3
m2 = m1*q
Mg1 = 1.72
Mg2 = 100
f = 10**(-0.4*(Mg2-Mg1))
dist_range_bh3 = np.linspace(0.1, 5.0, N_grid)
ecc = 0.73
pmra, pmdec, period = -28.37, -155.14, 4250
Mg_tot = -2.5*np.log10(10**(-Mg1/2.5) + 10**(-Mg2/2.5))
N = 500

all_prob_BH3 = []
for dist in dist_range_bh3:
    ra, dec, phot_g_mean_mag, Tp, omega, w, inc_deg, accept = gaiamock.simulate_many_realizations_of_a_single_binary(dist_pc = dist*1000, period = period, Mg_tot = Mg_tot, 
                    f = f, m1 = m1, m2 = m2, ecc = ecc, N_realizations = N, data_release=dr)
    prob = np.sum(accept)/len(accept)
    print('found probability!', dist, prob)
    all_prob_BH3.append(prob)
    
 
all_ranges = [dist_range_bh1, dist_range_bh2, dist_range_ns1, dist_range_bh3]
all_probs = [all_prob_BH1, all_prob_BH2, all_prob_NS1, all_prob_BH3]
np.savez('all_probabilities_detect_sim_bhs.npz', all_ranges=all_ranges, all_probs=all_probs)