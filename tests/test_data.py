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


def make_old_data(root):
    # healpix_scans/ and the ruwe map at the top of the repository, as the original README said to do
    (root / 'healpix_scans').mkdir(parents=True)
    (root / 'healpix_scans' / 'healpix_64_7.fits').touch()
    (root / 'healpix_16_med_ruwe.npz').touch()


def test_finds_old_data_at_checkout_root(data_dir, tmp_path, monkeypatch):
    # editable install from a checkout that still has its data from before the move to src/
    root = tmp_path / 'gaiamock'
    (root / 'src' / 'gaiamock').mkdir(parents=True)
    (root / 'pyproject.toml').write_text('[project]\nname = "gaiamock"\n')
    make_old_data(root)
    monkeypatch.setattr(data, 'PACKAGE_DIR', root / 'src' / 'gaiamock')

    assert data.get_source_checkout_root() == root
    assert gaiamock.data_path('healpix_scans', 'healpix_64_7.fits') == root / 'healpix_scans' / 'healpix_64_7.fits'
    assert gaiamock.data_path('healpix_16_med_ruwe.npz') == root / 'healpix_16_med_ruwe.npz'


@pytest.mark.parametrize('layout', ['site-packages', 'other src project'])
def test_does_not_search_outside_checkout(data_dir, tmp_path, monkeypatch, layout):
    # a regular install, or a src/ directory that belongs to a different project, never looks two levels up
    if layout == 'site-packages':
        package_dir = tmp_path / 'lib' / 'python3.12' / 'site-packages' / 'gaiamock'
        make_old_data(tmp_path / 'lib' / 'python3.12')
    else:
        package_dir = tmp_path / 'other' / 'src' / 'gaiamock'
        (tmp_path / 'other').mkdir()
        (tmp_path / 'other' / 'pyproject.toml').write_text('[project]\nname = "something-else"\n')
        make_old_data(tmp_path / 'other')
    package_dir.mkdir(parents=True)
    monkeypatch.setattr(data, 'PACKAGE_DIR', package_dir)

    assert data.get_source_checkout_root() is None
    with pytest.raises(FileNotFoundError):
        gaiamock.data_path('healpix_scans', 'healpix_64_7.fits')


def test_cli_where(data_dir, capsys):
    data.main(['--where'])
    assert capsys.readouterr().out.strip() == str(data_dir)
