# gaiamock

This is a package for simulating Gaia astrometry at the epoch level. To install it, do the following: 

(0) Install the required packages: numpy, matplotlib, os, ctypes, healpy, and joblib. In my experience these are usually relatively painless to install via pip. If you want to simulate sources distributed throughout the Galaxy with dust, you also need to install [mwdust](https://github.com/jobovy/mwdust), which is also available via pip. 

(1) clone this repository (click the green "code" button in the upper right of this page).

(2) download the file healpix_scans.zip from [this](https://caltech.box.com/s/4f7q6qdh0bku881bzvzxc4cm5u0902cf) link.
Unzip it inside gaiamock/ so that you have a directory gaiamock/healpix_scans/. That directory should have 49152 fits files inside it. The total size after unzipping will be 984 MB. 

(3) If you don't already have it installed, install [GSL](https://www.gnu.org/software/gsl/). On a Mac, this is likely most easily accomplished via Homebrew. 

(4) Inside the gaiamock/ directory, compile the file kepler_solve_astrometry.c. This will require linking GSL. The exact command will depend on where you installed GSL and on your compiler. On my Macbook, the command to compile was: 

gcc -shared -o kepler_solve_astrometry.so kepler_solve_astrometry.c -I/opt/homebrew/Cellar/gsl/2.7.1/include  -L/opt/homebrew/Cellar/gsl/2.7.1/lib -lgsl -lgslcblas -lm 

On my local cluster (where GSL is already linked by default), the command to compile was:

gcc -shared -o kepler_solve_astrometry.so kepler_solve_astrometry.c -lgsl -lgslcblas -lm -fPIC 

If everything works, this will create a compiled file gaiamock/kepler_solve_astrometry.so. If it didn't work, there is probably a problem with the linking of your GSL installation.

(5) You are ready to go! Some basic functionality is demonstrated in the demo.ipynb notebook. 

A simple bash submission script to run on a cluster is provided in example_bash_submission.py. That reproduces Fig 13 of the paper. 

# A modified version to predict RUWE more reliably for small orbits

The default version of gaiamock does a pretty good job of predicting RUWE for binaries with "large" photocenter orbits -- see e.g. Figure 1 of [this](https://arxiv.org/abs/2504.11528) paper. However, it doesn't accurately predict the RUWE distribution of single stars or binaries with barely-detectable orbital motion: it predicts a RUWE distribution that is narrower than observed. There are at least two reason for this: 

(a) The default gaiamock "bins" the 8-9 measurements from individual CCDs during a single FOV transit for computational speed. This reduces the variance in the predicted RUWE due to shot noise.

(b) The observed Gaia data (at least in DR3) displays systematic trends in the median RUWE with sky position, likely in part due to crowding (e.g. the median RUWE for bright stars is lowest near the Galactic center). The origin of these trends is not well understood.

A modified version of gaiamock is available to improve the reliability of RUWE predictions. It does away with binning (and therefore is slower by a factor of a few, but still fast enough for most applications), and implements an empirical position-dependent rescaling of the epoch-level astrometric uncertainties. This is probably not the optimal way to model RUWE, but it is significantly more reliable than the default version of the code. 

To use the modified version of gaiamock, do the following: 

(1) From [this](https://caltech.box.com/s/lnszhrytqjt4f28l6eoyghsxrs7vg25n) link, download the individual_ccds.zip file and unzip the contents into the healpix_scans/ directory of gaiamock. You can leave the other files that are already there.

(2) Download the healpix_16_med_ruwe.npz file and put it in your gaiamock/ directory.

(3) Download the gaiamock_mod.py file and put it in your gaiamock/ directory.

Now you can import gaiamock_mod as gaiamock and use the same functions you would use in gaiamock, e.g. for predicting epoch astrometry and computing ruwe. 
