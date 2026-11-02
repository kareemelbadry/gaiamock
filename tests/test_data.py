import zipfile

import numpy as np
import pytest

import gaiamock
from gaiamock import data


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('GAIAMOCK_DATA_DIR', str(tmp_path / 'gaiamock_data'))
    return tmp_path / 'gaiamock_data'


def make_zip(path, names):
    with zipfile.ZipFile(path, 'w') as zf:
        for name in names:
            zf.writestr(name, 'contents of ' + name)
    return path


def test_data_dir_from_environment(data_dir):
    assert gaiamock.get_data_dir() == data_dir


def test_missing_file_message(data_dir):
    with pytest.raises(FileNotFoundError, match='gaiamock-download'):
        gaiamock.data_path('healpix_scans', 'healpix_64_0.fits')


@pytest.mark.parametrize('prefix', ['healpix_scans/', ''])
def test_install_scans_from_zip(data_dir, tmp_path, prefix):
    zip_path = make_zip(tmp_path / 'healpix_scans.zip',
                        [prefix + 'healpix_64_0.fits', prefix + 'healpix_64_1.fits',
                         '__MACOSX/healpix_scans/._healpix_64_0.fits', '../escape.txt'])
    gaiamock.download_data(zip_path)

    assert sorted(p.name for p in (data_dir / 'healpix_scans').iterdir()) == ['healpix_64_0.fits', 'healpix_64_1.fits']
    assert gaiamock.data_path('healpix_scans', 'healpix_64_1.fits') == data_dir / 'healpix_scans' / 'healpix_64_1.fits'
    assert not (tmp_path / 'escape.txt').exists()


def test_install_other_files(data_dir, tmp_path):
    # a real npz, since those are zip files too and must be copied rather than unpacked
    npz = tmp_path / 'healpix_16_med_ruwe.npz'
    np.savez(npz, med_ruwe=np.arange(3.0))
    ccds = make_zip(tmp_path / 'individual_ccds.zip', ['individual_ccds/healpix_16_0.fits'])
    gaiamock.download_data(npz, ccds)

    with np.load(gaiamock.data_path('healpix_16_med_ruwe.npz')) as installed:
        np.testing.assert_array_equal(installed['med_ruwe'], np.arange(3.0))
    assert sorted(p.name for p in (data_dir / 'healpix_scans').iterdir()) == ['healpix_16_0.fits']


def test_falls_back_to_package_directory(data_dir, tmp_path, monkeypatch):
    # data unzipped next to gaiamock.py, as the original README said to do
    monkeypatch.setattr(data, 'PACKAGE_DIR', tmp_path)
    (tmp_path / 'healpix_scans').mkdir()
    (tmp_path / 'healpix_scans' / 'healpix_64_7.fits').touch()
    assert gaiamock.data_path('healpix_scans', 'healpix_64_7.fits') == tmp_path / 'healpix_scans' / 'healpix_64_7.fits'


def test_cli_where(data_dir, capsys):
    data.main(['--where'])
    assert capsys.readouterr().out.strip() == str(data_dir)
