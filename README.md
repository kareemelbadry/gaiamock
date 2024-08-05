# gaiamock

This is a package for simulating Gaia astrometry at the epoch level. To install it, do the following: 

(0) Install the required packages: numpy, matplotlib, os, ctypes, healpy, and joblib. In my experience these are usually relatively painless to install via pip. If you want to simulate sources distributed throughout the Galaxy with dust, you also need to install [mwdust](https://github.com/jobovy/mwdust), which is also available via pip. 

(1) clone this repository (click the green "code" button in the upper right of this page).

(2) download the file healpix_scans.zip from [this](https://caltech.box.com/s/hi8ftcz9aeis0a32p8edlrn1ejbhfpf3) link.
Unzip it inside gaiamock/ so that you have a directory gaiamock/healpix_scans/. That directory should have 49152 fits files inside it. The total size after unzipping will be 984 MB. 

(3) If you don't already have it installed, install [GSL](https://www.gnu.org/software/gsl/). On a Mac, this is likely most easily accomplished via Homebrew. 

(4) Compile the file kepler_solve_astrometry.c. This will require linking GSL. The exact command will depend on where you installed GSL and on your compiler. On my Macbook, the command to compile was: 

gcc -shared -o kepler_solve_astrometry.so kepler_solve_astrometry.c -I/opt/homebrew/Cellar/gsl/2.7.1/include  -L/opt/homebrew/Cellar/gsl/2.7.1/lib -lgsl -lgslcblas -lm 

On my local cluster (where GSL is already linked by default), the command to compile was:

gcc -shared -o kepler_solve_astrometry.so kepler_solve_astrometry.c -lgsl -lgslcblas -lm -fPIC 

If everything works, this will create a compiled file gaiamock/kepler_solve_astrometry.so. If it didn't work, there is probably a problem with the linking of your GSL installation.

(5) You are ready to go! Some basic functionality is demonstrated in the demo.ipynb notebook. 