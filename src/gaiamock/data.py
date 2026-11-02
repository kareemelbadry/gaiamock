'''
This script downloads all of the data necessary for gaiamock to run

The precomputed Gaia scanning law (healpix_scans/, 49152 fits files, 984MB) is rather large,
so it is downloaded once by running `gaiamock-download` in a terminal (or
gaiamock.download_data() in python). Data files are looked for in:

  1. $GAIAMOCK_DATA_DIR if that environment variable is set, otherwise it is stored in ~/.gaiamock/
  2. the gaiamock package directory, i.e. next to gaiamock.py (where the original instructions
     said to unzip healpix_scans.zip)

By default, downloads will be stored in (1)
'''
import argparse
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent

# direct-download version of https://caltech.box.com/s/4f7q6qdh0bku881bzvzxc4cm5u0902cf
HEALPIX_SCANS_URL = 'https://caltech.box.com/shared/static/4f7q6qdh0bku881bzvzxc4cm5u0902cf.zip'
HEALPIX_SCANS_BROWSER_URL = 'https://caltech.box.com/s/4f7q6qdh0bku881bzvzxc4cm5u0902cf'
N_HEALPIX_SCANS = 49152


def get_data_dir():
    if os.environ.get('GAIAMOCK_DATA_DIR'):
        return Path(os.environ['GAIAMOCK_DATA_DIR']).expanduser()
    else:
        return Path.home() / '.gaiamock'

    
def data_path(*parts):
    """Return the path to a data file.

    This looks in the following locations, in order:
    1. $GAIAMOCK_DATA_DIR if that environment variable is set, otherwise ~/.gaiamock/
    2. the gaiamock package directory, i.e. next to gaiamock.py (where the original instructions said to unzip healpix_scans.zip)

    Parameters
    ----------
    *parts : str
        Path components to append to the data directory.

    Returns
    -------
    path : pathlib.Path
        The path to the requested data file.
    
    Raises
    ------
    FileNotFoundError
        If the file does not exist in any of the search directories.
    """
    search_dirs = [get_data_dir(), PACKAGE_DIR]
    for directory in search_dirs:
        path = directory.joinpath(*parts)
        if path.exists():
            return path
    raise FileNotFoundError(
        f"I couldn't find gaiamock data file '{Path(*parts)}' in any of the following locations: {', '.join(str(d) for d in search_dirs)}. "
        "Run `gaiamock-download` in a terminal to download the Gaia scanning law, or see the README for the "
        "other data files. Set GAIAMOCK_DATA_DIR to keep the data somewhere else."
    )


def download_data(*files, data_dir=None):
    """Install gaiamock's data files into data_dir

    Parameters
    ----------
    files : str
        Paths of files you have already downloaded by hand: .zip files (healpix_scans.zip, individual_ccds.zip) are
        unpacked into data_dir/healpix_scans, and anything else (e.g. healpix_16_med_ruwe.npz) is copied into data_dir.
        If no files are given, healpix_scans.zip is downloaded from Box.

    data_dir : str or pathlib.Path, optional
        Where to put the data (default: $GAIAMOCK_DATA_DIR if set, otherwise ~/.gaiamock/).
        If the directory does not exist, it will be created.

    Returns
    -------
    data_dir : pathlib.Path
        The directory where the data files were installed.

    Raises
    ------
    RuntimeError
        If the download from Box fails or the downloaded file is not a zip file.
    """
    data_dir = Path(data_dir).expanduser() if data_dir else get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    if not files:
        # download next to the destination so the zip and its contents are on the same filesystem
        with tempfile.TemporaryDirectory(dir=data_dir) as tmp:
            zip_path = Path(tmp) / 'healpix_scans.zip'
            _download(HEALPIX_SCANS_URL, zip_path)
            if not zipfile.is_zipfile(zip_path):
                raise RuntimeError(
                    f'The file downloaded from {HEALPIX_SCANS_URL} is not a zip file, so Box may have changed the link. '
                    f'Download healpix_scans.zip in a browser from {HEALPIX_SCANS_BROWSER_URL} and then run '
                    '`gaiamock-download path/to/healpix_scans.zip`.')
            _unpack_scans(zip_path, data_dir / 'healpix_scans')

    for f in map(Path, files):
        # check the extension rather than zipfile.is_zipfile(), since .npz files are zip files too
        if f.suffix.lower() == '.zip':
            _unpack_scans(f, data_dir / 'healpix_scans')
        else:
            print(f'Copying {f} -> {data_dir}')
            shutil.copy(f, data_dir / f.name)

    n_scans = len(list((data_dir / 'healpix_scans').glob('healpix_64_*.fits')))
    if n_scans and n_scans < N_HEALPIX_SCANS:
        print(f'Warning: {data_dir / "healpix_scans"} contains {n_scans} of the expected {N_HEALPIX_SCANS} '
              'healpix_64_*.fits files; the download may be incomplete.')
    return data_dir


def _download(url, path):
    print(f'Downloading {url}\n         -> {path}')
    with urllib.request.urlopen(url) as response, open(path, 'wb') as out:
        total = int(response.headers.get('Content-Length') or 0)
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f'\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB', end='', flush=True)
    print()


def _unpack_scans(zip_path, dest):
    """Unpack the files in a zip of scans directly into dest, ignoring any folders inside the zip
    Parameters
    ----------
    zip_path : pathlib.Path
        Path to the zip file to unpack.
    dest : pathlib.Path
        Directory to unpack the files into. If it does not exist, it will be created.
    """
    print(f'Unpacking {zip_path} -> {dest}')
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            parts = Path(member.filename).parts
            # skip folders, macOS metadata, and anything that looks like it's trying to escape the zip
            if member.is_dir() or '__MACOSX' in parts or parts[-1].startswith('.'):
                continue
            if Path(member.filename).is_absolute() or '..' in parts:
                continue
            with zf.open(member) as src, open(dest / parts[-1], 'wb') as out:
                shutil.copyfileobj(src, out)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='gaiamock-download',
        description='Download the Gaia scanning law (healpix_scans.zip, ~1 GB) that gaiamock needs, or install '
                    'data files that you have already downloaded by hand.')
    parser.add_argument('files', nargs='*',
                        help='already-downloaded files to install instead of downloading healpix_scans.zip: zip files '
                             'are unpacked into healpix_scans/, anything else (e.g. healpix_16_med_ruwe.npz) is copied')
    parser.add_argument('--data-dir',
                        help=f'where to put the data (default: $GAIAMOCK_DATA_DIR if set, otherwise {get_data_dir()})')
    parser.add_argument('--where', action='store_true', help='print the data directory and exit')
    args = parser.parse_args(argv)

    if args.where:
        print(get_data_dir())
        return
    data_dir = download_data(*args.files, data_dir=args.data_dir)
    print(f'Done: gaiamock data is in {data_dir}')
    if data_dir.resolve() not in (get_data_dir().resolve(), PACKAGE_DIR):
        print(f'gaiamock only looks in {get_data_dir()} by default, so set GAIAMOCK_DATA_DIR={data_dir}')


if __name__ == '__main__':
    main()
