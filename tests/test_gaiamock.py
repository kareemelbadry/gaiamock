import healpy as hp
import numpy as np
import pytest
from astropy.table import Table

import gaiamock


def have_data(*parts):
    try:
        gaiamock.data_path(*parts)
        return True
    except FileNotFoundError:
        return False


@pytest.fixture
def fake_unbinned_data(tmp_path, monkeypatch):
    '''
    per-CCD scans for one sky position (ra=150, dec=20) and a median ruwe map, in a temporary GAIAMOCK_DATA_DIR
    '''
    monkeypatch.setenv('GAIAMOCK_DATA_DIR', str(tmp_path))
    (tmp_path / 'healpix_scans').mkdir()
    num = hp.ang2pix(16, np.radians(90.0 - 20), np.radians(150), nest=False)
    n_transits, n_ccds = 30, 9
    Table({
        'ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]': np.repeat(np.linspace(2456900, 2457890, n_transits), n_ccds),
        'scanAngle[rad]': np.repeat(np.linspace(0, 6, n_transits), n_ccds),
        'parallaxFactorAlongScan': np.repeat(np.linspace(-1, 1, n_transits), n_ccds),
    }).write(tmp_path / 'healpix_scans' / f'healpix_16_{num}.fits')
    np.savez(tmp_path / 'healpix_16_med_ruwe.npz', med_ruwe=np.full(hp.nside2npix(16), 1.2))
    return n_transits * n_ccds


def test_c_library_loads():
    c_funcs = gaiamock.read_in_C_functions()
    assert c_funcs.run_astfit_periodogram_multistart is not None


@pytest.mark.parametrize('ecc', [0.0, 0.3, 0.9])
def test_kepler_solver(ecc):
    c_funcs = gaiamock.read_in_C_functions()
    M = np.linspace(0.01, 2 * np.pi - 0.01, 200)
    E = gaiamock.solve_kepler_eqn_on_array(M, ecc, c_funcs)
    np.testing.assert_allclose(E - ecc * np.sin(E), M, atol=1e-8)


def test_orbit_fit_recovers_period():
    '''
    end-to-end check of the compiled optimizer on synthetic scans, so this doesn't need the healpix_scans data:
    same model as predict_astrometry_binary_in_terms_of_a0(), with made-up scan times, angles and parallax factors
    '''
    rng = np.random.default_rng(42)
    c_funcs = gaiamock.read_in_C_functions()

    # 25 visibility periods spread over ~3 years, each with 3 transits a few hours apart
    t_ast_yr = np.repeat(np.linspace(-1.4, 1.4, 25), 3) + np.tile([0, 0.25, 0.5], 25) / 365.25
    psi = np.repeat(rng.uniform(0, 2 * np.pi, 25), 3)
    plx_factor = np.repeat(rng.uniform(-1, 1, 25), 3)

    period, ecc, Tp, a0_mas = 300.0, 0.3, 50.0, 3.0
    omega, inc, w = 1.0, 0.8, 2.0
    parallax, pmra, pmdec, ast_err = 5.0, 10.0, -20.0, 0.05

    EE = gaiamock.solve_kepler_eqn_on_array(M=2 * np.pi / period * (t_ast_yr * 365.25 - Tp), ecc=ecc, c_funcs=c_funcs)
    A = a0_mas * (np.cos(w) * np.cos(omega) - np.sin(w) * np.sin(omega) * np.cos(inc))
    B = a0_mas * (np.cos(w) * np.sin(omega) + np.sin(w) * np.cos(omega) * np.cos(inc))
    F = -a0_mas * (np.sin(w) * np.cos(omega) + np.cos(w) * np.sin(omega) * np.cos(inc))
    G = -a0_mas * (np.sin(w) * np.sin(omega) - np.cos(w) * np.cos(omega) * np.cos(inc))
    X, Y = np.cos(EE) - ecc, np.sqrt(1 - ecc**2) * np.sin(EE)
    x, y = B * X + G * Y, A * X + F * Y
    ast_obs = (pmra * t_ast_yr * np.sin(psi) + pmdec * t_ast_yr * np.cos(psi) + parallax * plx_factor
               + y * np.cos(psi) + x * np.sin(psi) + ast_err * rng.standard_normal(len(psi)))

    res = gaiamock.fit_full_astrometric_cascade(t_ast_yr, psi, plx_factor, ast_obs, ast_err * np.ones_like(ast_obs),
                                                c_funcs, skip_acceleration=True)
    fit_plx, fit_period, fit_ecc, fit_a0 = res[0], res[10], res[14], res[17]
    assert fit_period == pytest.approx(period, rel=0.02)
    assert fit_ecc == pytest.approx(ecc, abs=0.05)
    assert fit_a0 == pytest.approx(a0_mas, rel=0.05)
    assert fit_plx == pytest.approx(parallax, abs=0.1)


def test_unbinned_single_source(fake_unbinned_data):
    '''
    binned=False reads the per-CCD scans and inflates the noise by the median ruwe map (1.2 here), but returns the
    expected per-CCD uncertainties, so a single star comes out with ruwe a bit above 1
    '''
    np.random.seed(0)
    t_ast_yr, psi, plx_factor, ast_obs, ast_err = gaiamock.predict_astrometry_single_source(
        ra=150, dec=20, parallax=2, pmra=5, pmdec=-5, phot_g_mean_mag=15, data_release='dr3', c_funcs=None, binned=False)

    assert 0.8 * fake_unbinned_data < len(t_ast_yr) < fake_unbinned_data  # one row per CCD, minus a random 10%
    np.testing.assert_allclose(ast_err, gaiamock.al_uncertainty_per_ccd_interp(15))
    ruwe, mu, sigma_mu = gaiamock.check_ruwe(t_ast_yr, psi, plx_factor, ast_obs, ast_err, binned=False)
    assert 1.0 < ruwe < 1.3


@pytest.mark.skipif(not have_data('healpix_scans', 'healpix_64_0.fits'), reason='healpix_scans not downloaded (run gaiamock-download)')
def test_predict_astrometry_with_scanning_law():
    c_funcs = gaiamock.read_in_C_functions()
    t_ast_yr, psi, plx_factor, ast_obs, ast_err = gaiamock.predict_astrometry_luminous_binary(
        ra=150, dec=20, parallax=2, pmra=5, pmdec=-5, m1=1, m2=0.8, period=500, Tp=0, ecc=0.2, omega=1, inc=0.5,
        w=2, phot_g_mean_mag=13, f=0.01, data_release='dr3', c_funcs=c_funcs)
    assert len(t_ast_yr) > 13
    assert np.all(np.isfinite(ast_obs))


@pytest.mark.skipif(not (have_data('healpix_scans', 'healpix_16_0.fits') and have_data('healpix_16_med_ruwe.npz')),
                    reason='individual_ccds.zip and healpix_16_med_ruwe.npz not installed (see the README)')
def test_unbinned_cascade_with_scanning_law():
    res = gaiamock.run_full_astrometric_cascade(
        ra=150, dec=20, parallax=2, pmra=5, pmdec=-5, m1=1, m2=0.8, period=500, Tp=0, ecc=0.2, omega=1, inc_deg=30,
        w=2, phot_g_mean_mag=13, f=0.01, data_release='dr3', c_funcs=None, skip_acceleration=True, binned=False)
    assert len(res) == 23 and np.all(np.isfinite(res))
