'''
Compiles src/gaiamock/kepler_solve_astrometry.c into the shared library that gaiamock loads with ctypes.

The package metadata lives in pyproject.toml; this file only exists to build the C code, which needs GSL.
GSL is located using (in order) the GSL_DIR or GSL_ROOT environment variables, gsl-config, pkg-config, and
then a few common install locations (the active conda environment, Homebrew, /usr/local, /usr).
'''
import os
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext
from setuptools.errors import CompileError

GSL_NOT_FOUND = '''
Could not find GSL, which is needed to compile gaiamock's C code. Install it and try again, e.g. with

    macOS (Homebrew):  brew install gsl
    Debian/Ubuntu:     sudo apt install libgsl-dev
    Fedora/RHEL:       sudo dnf install gsl-devel
    conda:             conda install -c conda-forge gsl
    HPC clusters:      module load gsl

If GSL is installed somewhere else, set GSL_DIR to its install prefix (the directory containing include/gsl/),
e.g.  GSL_DIR=/path/to/gsl pip install gaiamock
'''


def _command_output(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def find_gsl():
    '''
    return (include_dirs, library_dirs) for GSL; empty lists mean it is in the compiler's default search path
    '''
    candidates = [os.environ.get('GSL_DIR'),
                  os.environ.get('GSL_ROOT'),  # set by `module load gsl` on the Flatiron clusters
                  _command_output('gsl-config', '--prefix'),
                  _command_output('pkg-config', '--variable=prefix', 'gsl'),
                  os.environ.get('CONDA_PREFIX'),
                  '/opt/homebrew', '/usr/local', '/usr']
    searched = []
    for prefix in filter(None, candidates):
        prefix = Path(prefix)
        searched.append(str(prefix))
        if not (prefix / 'include' / 'gsl' / 'gsl_matrix.h').exists():
            continue
        if prefix == Path('/usr'):
            return [], []
        library_dirs = [str(d) for d in (prefix / 'lib', prefix / 'lib64') if any(d.glob('libgsl.*'))]
        return [str(prefix / 'include')], library_dirs
    raise CompileError(f"{GSL_NOT_FOUND}Looked in: {', '.join(searched)}\n")


class build_ctypes_library(build_ext):
    '''
    builds a plain shared library to be loaded with ctypes, rather than a Python extension module
    '''
    def build_extensions(self):
        include_dirs, library_dirs = find_gsl()
        # put GSL's -L first: some Pythons (e.g. the Nix builds on the Flatiron clusters) have -L/usr/lib64 in their
        # own link flags, which would otherwise link against a different system GSL than the headers we found
        self.compiler.linker_so[1:1] = [f'-L{d}' for d in library_dirs]
        for ext in self.extensions:
            ext.include_dirs += include_dirs
            # record where GSL is so the library loads without needing LD_LIBRARY_PATH or `module load gsl`
            ext.extra_link_args += [f'-Wl,-rpath,{d}' for d in library_dirs]
        super().build_extensions()

    def get_ext_filename(self, fullname):
        # kepler_solve_astrometry.so rather than e.g. kepler_solve_astrometry.cpython-312-x86_64-linux-gnu.so,
        # since that is the name read_in_C_functions() looks for
        return os.path.join(*fullname.split('.')) + '.so'

    def get_export_symbols(self, ext):
        # there is no PyInit_ function to export
        return ext.export_symbols


class bdist_wheel_any_python(bdist_wheel):
    '''
    the C library doesn't use the Python C API, so one wheel per platform works for every Python 3 version
    '''
    def get_tag(self):
        return 'py3', 'none', super().get_tag()[2]


setup(
    ext_modules=[Extension('gaiamock.kepler_solve_astrometry',
                           sources=['src/gaiamock/kepler_solve_astrometry.c'],
                           libraries=['gsl', 'gslcblas', 'm'])],
    cmdclass={'build_ext': build_ctypes_library, 'bdist_wheel': bdist_wheel_any_python},
)
