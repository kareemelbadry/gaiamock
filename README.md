# gaiamock

This is a package for simulating Gaia astrometry at the epoch level. Install it from PyPI with:

```
pip install gaiamock
```

Binary wheels include GSL and are available for Linux (Intel/AMD and ARM64) and macOS 15 or later (Intel and Apple Silicon). After installation, download the scanning-law data as described in step (3) below. If pip needs to build from source, or you want to install the development version, follow steps (1) and (2).

(1) If you don't already have it installed, install [GSL](https://www.gnu.org/software/gsl/), which is needed to compile gaiamock's C code. For example:

| | |
|---|---|
| macOS (Homebrew) | `brew install gsl` |
| Debian/Ubuntu | `sudo apt install libgsl-dev` |
| conda | `conda install -c conda-forge gsl` |
| HPC clusters | `module load gsl` (before running pip below) |

(2) Install gaiamock with pip:
```
pip install git+https://github.com/kareemelbadry/gaiamock.git
```
or, if you want to edit the code, clone this repository and install it in editable mode:
```
git clone https://github.com/kareemelbadry/gaiamock.git
cd gaiamock
pip install -e .
```
This installs the required packages (numpy, matplotlib, astropy, healpy, and joblib) and compiles kepler_solve_astrometry.c. GSL is found automatically if `gsl-config` is on your PATH or GSL is in a standard location (Homebrew, conda, /usr/local, /usr). If yours is somewhere else, point pip at it with `GSL_DIR=/path/to/gsl pip install ...`, where /path/to/gsl is the directory containing include/gsl/.

Some optional features need extra packages, which pip can install for you with e.g. `pip install -e ".[all]"`:
`[mcmc]` (emcee and scipy) for the MCMC fitting functions, and `[dust]` ([mwdust](https://github.com/jobovy/mwdust)) to simulate sources distributed throughout the Galaxy with dust.

(3) Download the precomputed Gaia scanning law (healpix_scans.zip: 49152 fits files, 984 MB after unzipping) by running
```
gaiamock-download
```
in a terminal. By default this puts the data in ~/.gaiamock. To keep it somewhere else, set the environment variable `GAIAMOCK_DATA_DIR` to that directory, both when downloading and when using gaiamock. `gaiamock-download --where` prints where gaiamock looks for the data.

If the download doesn't work, download healpix_scans.zip in a browser from [this](https://caltech.box.com/s/4f7q6qdh0bku881bzvzxc4cm5u0902cf) link and then run `gaiamock-download path/to/healpix_scans.zip` to unpack it into the right place.

If you already downloaded the data for an older version of gaiamock, you don't need to download it again:
- if you installed from your existing clone with `pip install -e .`, the healpix_scans/ directory (and healpix_16_med_ruwe.npz) at the top of the clone, where older versions of these instructions put them, is still found automatically.
- otherwise, set `GAIAMOCK_DATA_DIR=/path/to/old/gaiamock` (the directory containing healpix_scans/), or move healpix_scans/ into ~/.gaiamock.

(4) You are ready to go! Some basic functionality is demonstrated in the docs/demo.ipynb notebook. You can check that everything works with `pip install -e ".[test]"` and then `pytest`. 

A simple bash submission script to run on a cluster is provided in docs/example_bash_submission.py. That reproduces Fig 13 of the paper. 

This code is regularly updated and includes functionality beyond what is shown in the demo. If you have questions, or if you spot a bug, please contact me at kelbadry@caltech.edu. 

The release distribution is licensed under GPL-3.0-or-later because it links to GNU GSL. The original MIT notice for gaiamock-authored source is retained. Each official wheel includes the complete corresponding source for gaiamock and its bundled GSL in `gaiamock/_sources/`. See [THIRD_PARTY_NOTICES](https://github.com/kareemelbadry/gaiamock/blob/main/THIRD_PARTY_NOTICES) for license details and rebuild instructions.

# A modified version to predict RUWE more reliably for small orbits and single stars

The default version of gaiamock does a pretty good job of predicting RUWE for binaries with "large" photocenter orbits -- see e.g. Figure 1 of [this](https://arxiv.org/abs/2504.11528) paper. However, it doesn't accurately predict the RUWE distribution of single stars or binaries with barely-detectable orbital motion: it predicts a RUWE distribution that is narrower than observed. There are at least two reason for this: 

(a) The default gaiamock bins (i.e. averages) the 8-9 measurements from individual CCDs during a single FOV transit for computational speed. This reduces the variance in the predicted RUWE due to shot noise.

(b) The observed Gaia data (at least in DR3) displays systematic trends in the median RUWE with sky position, likely in part due to crowding (e.g. the median RUWE for bright stars is lowest near the Galactic center). The origin of these trends is not well understood.

An unbinned mode of gaiamock is available to improve the reliability of RUWE predictions. It does away with binning (and therefore is slower by a factor of a few, but still fast enough for most applications), and implements an empirical position-dependent rescaling of the epoch-level astrometric uncertainties. This is probably not the optimal way to model RUWE, but it is significantly more reliable than the default version of the code. 

To use it, do the following: 

(1) From [this](https://caltech.box.com/s/lnszhrytqjt4f28l6eoyghsxrs7vg25n) link, download the individual_ccds.zip and healpix_16_med_ruwe.npz files, and install them with
```
gaiamock-download individual_ccds.zip healpix_16_med_ruwe.npz
```
This unzips individual_ccds.zip into the same healpix_scans/ directory as the default scans (you can leave the other files that are already there).

(2) Pass `binned=False` to the functions that predict epoch astrometry (`predict_astrometry_luminous_binary`, `predict_astrometry_binary_in_terms_of_a0`, `predict_astrometry_single_source`) and to the functions you fit it with (e.g. `fit_full_astrometric_cascade`, `check_ruwe`). The functions that do both (`run_full_astrometric_cascade`, `run_only_5par_solution`, and `simulate_many_realizations_of_a_single_binary`) take `binned=False` too, e.g.
```
res = gaiamock.run_full_astrometric_cascade(..., binned=False)
```
This replaces the separate gaiamock_mod.py file used by older versions of gaiamock.
