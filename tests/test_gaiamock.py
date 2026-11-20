import ctypes

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
    assert c_funcs.run_astfit_periodogram_multistart_v2 is not None


@pytest.mark.parametrize('ecc', [0.0, 0.3, 0.9])
def test_kepler_solver(ecc):
    c_funcs = gaiamock.read_in_C_functions()
    M = np.linspace(0.01, 2 * np.pi - 0.01, 200)
    E = gaiamock.solve_kepler_eqn_on_array(M, ecc, c_funcs)
    np.testing.assert_allclose(E - ecc * np.sin(E), M, atol=1e-8)


def synthetic_epoch_astrometry(rng, c_funcs, period=300.0, ecc=0.3, Tp=50.0, a0_mas=3.0, ast_err=0.05):
    '''
    same model as predict_astrometry_binary_in_terms_of_a0(), with made-up scan times, angles and parallax factors,
    so the tests don't need the healpix_scans data. 25 visibility periods spread over ~3 years, each with 3 transits
    a few hours apart.
    '''
    t_ast_yr = np.repeat(np.linspace(-1.4, 1.4, 25), 3) + np.tile([0, 0.25, 0.5], 25) / 365.25
    psi = np.repeat(rng.uniform(0, 2 * np.pi, 25), 3)
    plx_factor = np.repeat(rng.uniform(-1, 1, 25), 3)

    omega, inc, w = 1.0, 0.8, 2.0
    parallax, pmra, pmdec = 5.0, 10.0, -20.0

    EE = gaiamock.solve_kepler_eqn_on_array(M=2 * np.pi / period * (t_ast_yr * 365.25 - Tp), ecc=ecc, c_funcs=c_funcs)
    A = a0_mas * (np.cos(w) * np.cos(omega) - np.sin(w) * np.sin(omega) * np.cos(inc))
    B = a0_mas * (np.cos(w) * np.sin(omega) + np.sin(w) * np.cos(omega) * np.cos(inc))
    F = -a0_mas * (np.sin(w) * np.cos(omega) + np.cos(w) * np.sin(omega) * np.cos(inc))
    G = -a0_mas * (np.sin(w) * np.sin(omega) - np.cos(w) * np.cos(omega) * np.cos(inc))
    X, Y = np.cos(EE) - ecc, np.sqrt(1 - ecc**2) * np.sin(EE)
    x, y = B * X + G * Y, A * X + F * Y
    ast_obs = (pmra * t_ast_yr * np.sin(psi) + pmdec * t_ast_yr * np.cos(psi) + parallax * plx_factor
               + y * np.cos(psi) + x * np.sin(psi) + ast_err * rng.standard_normal(len(psi)))
    return t_ast_yr, psi, plx_factor, ast_obs, ast_err * np.ones_like(ast_obs)


@pytest.mark.parametrize('optimizer', ['periodogram_multistart', 'periodogram_multistart_v2'])
def test_orbit_fit_recovers_period(optimizer):
    '''
    end-to-end check of the compiled optimizers on synthetic scans
    '''
    rng = np.random.default_rng(42)
    c_funcs = gaiamock.read_in_C_functions()
    period, ecc, a0_mas, parallax = 300.0, 0.3, 3.0, 5.0
    t_ast_yr, psi, plx_factor, ast_obs, ast_err = synthetic_epoch_astrometry(rng, c_funcs, period=period, ecc=ecc, a0_mas=a0_mas)

    res = gaiamock.fit_full_astrometric_cascade(t_ast_yr, psi, plx_factor, ast_obs, ast_err, c_funcs,
                                                skip_acceleration=True, optimizer=optimizer)
    fit_plx, fit_period, fit_ecc, fit_a0 = res[0], res[10], res[14], res[17]
    assert fit_period == pytest.approx(period, rel=0.02)
    assert fit_ecc == pytest.approx(ecc, abs=0.05)
    assert fit_a0 == pytest.approx(a0_mas, rel=0.05)
    assert fit_plx == pytest.approx(parallax, abs=0.1)


@pytest.mark.parametrize('reject_outlier', [False, True])
def test_fast_chi2_matches_gsl(reject_outlier):
    '''
    the optimizers' chi2 (orthogonalized fixed columns, delete-one identity for reject_outlier) should agree with the
    GSL normal-equation chi2 used everywhere else
    '''
    rng = np.random.default_rng(1)
    c_funcs = gaiamock.read_in_C_functions()
    t_ast_yr, psi, plx_factor, ast_obs, ast_err = synthetic_epoch_astrometry(rng, c_funcs, period=700.0, ecc=0.6)
    ast_obs[10] += 2.0  # an outlier, so reject_outlier has something to do
    for P, phi_p, ecc in [(700.0, 1.0, 0.6), (31.7, 4.0, 0.0), (12.3, 0.2, 0.95), (9000.0, 5.5, 0.4), (365.25, 3.0, 0.99)]:
        chi2_gsl, _ = gaiamock.get_astrometric_chi2(t_ast_yr, psi, plx_factor, ast_obs, ast_err, P, phi_p, ecc, c_funcs,
                                                    reject_outlier=reject_outlier)
        chi2_fast = c_funcs.get_chi2_astrometry_fast(
            ctypes.c_int(len(t_ast_yr)), *[ctypes.c_void_p(np.ascontiguousarray(a, dtype=np.double).ctypes.data) for a in
                                           (t_ast_yr, psi, plx_factor, ast_obs, ast_err)],
            ctypes.c_double(P), ctypes.c_double(phi_p), ctypes.c_double(ecc), ctypes.c_int(int(reject_outlier)))
        assert chi2_fast == pytest.approx(chi2_gsl, rel=1e-9)


def fast_chi2_sequence(c_funcs, data, P, phi_p, ecc, reject_outlier):
    '''chi2 at each (P, phi_p, ecc), evaluated in order on one C workspace, as inside the optimizers'''
    arrs = [np.ascontiguousarray(a, dtype=np.double) for a in data]
    pts = [np.ascontiguousarray(a, dtype=np.double) for a in (P, phi_p, ecc)]
    out = np.zeros(len(pts[0]))
    c_funcs.get_chi2_astrometry_fast_sequence(
        ctypes.c_int(len(arrs[0])), *[ctypes.c_void_p(a.ctypes.data) for a in arrs], ctypes.c_int(len(out)),
        *[ctypes.c_void_p(a.ctypes.data) for a in pts], ctypes.c_int(int(reject_outlier)),
        ctypes.c_void_p(out.ctypes.data))
    return out


@pytest.mark.parametrize('reject_outlier', [False, True])
def test_fast_chi2_sequence_matches_gsl(reject_outlier):
    '''
    Inside the optimizers, consecutive chi2 calls on one workspace warm-start the Kepler solve from the previous call,
    and the eccentricities of the v2 grid use a tabulated starter. Neither should change chi2.
    '''
    rng = np.random.default_rng(0)
    c_funcs = gaiamock.read_in_C_functions()
    data = synthetic_epoch_astrometry(rng, c_funcs, period=200.0, ecc=0.8, a0_mas=2.0)
    data[3][5] += 1.5  # an outlier

    # 1. a long random walk at high eccentricity, with steps from tiny to large: compare with fresh evaluations
    n = 20000
    step = 10 ** rng.uniform(-4, -0.5, n)
    P = 200.0 * np.exp(np.cumsum(0.01 * step * rng.normal(size=n)))
    phi_p = (1.0 + np.cumsum(step * rng.normal(size=n))) % (2 * np.pi)
    ecc = np.clip(0.9 + np.cumsum(0.1 * step * rng.normal(size=n)), 0.0, 0.99)
    walk = fast_chi2_sequence(c_funcs, data, P, phi_p, ecc, reject_outlier)
    fresh = [fast_chi2_sequence(c_funcs, data, P[i:i + 1], phi_p[i:i + 1], ecc[i:i + 1], reject_outlier)[0]
             for i in range(n)]
    np.testing.assert_allclose(walk, fresh, rtol=1e-11)

    # 2. phase scans at the v2 grid eccentricities and e = 0, short to long periods: compare with GSL
    P, phi_p, ecc = np.array([(p, f, e) for p in [12.0, 200.0, 3000.0, 9500.0] for e in [0.2, 0.4, 0.6, 0.8, 0.95, 0.0]
                              for f in np.linspace(0, 2 * np.pi, 7)[:-1]]).T
    scan = fast_chi2_sequence(c_funcs, data, P, phi_p, ecc, reject_outlier)
    expected = [gaiamock.get_astrometric_chi2(*data, P[i], phi_p[i], ecc[i], c_funcs, reject_outlier=reject_outlier)[0]
                for i in range(len(P))]
    np.testing.assert_allclose(scan, expected, rtol=1e-9)


def test_rejection_sampling_likelihoods_match_gsl():
    '''get_likelihoods_astrometry() (rejection sampling) uses the fast chi2; it should match the GSL implementation'''
    rng = np.random.default_rng(5)
    c_funcs = gaiamock.read_in_C_functions()
    data = synthetic_epoch_astrometry(rng, c_funcs, period=300.0, ecc=0.3, a0_mas=0.3)
    np.random.seed(5)
    P, ecc, phi_p = gaiamock.generate_prior_samples(3000)
    lnL = gaiamock.get_astrometric_likelihoods_worker(*data, (P, ecc, phi_p))
    arrs = [np.ascontiguousarray(a, dtype=np.double) for a in list(data) + [P, phi_p, ecc]]
    lnL_gsl = np.empty(len(P))
    c_funcs.get_likelihoods_astrometry_gsl(ctypes.c_int(len(data[0])), ctypes.c_int(len(P)),
                                           *[ctypes.c_void_p(a.ctypes.data) for a in arrs],
                                           ctypes.c_void_p(lnL_gsl.ctypes.data))
    np.testing.assert_allclose(lnL, lnL_gsl, rtol=1e-9)


def test_grid_chi2_approximation():
    '''the approximate chi2 used to rank the v2 eccentric grid should be close to the exact chi2 at the grid eccentricities'''
    rng = np.random.default_rng(9)
    c_funcs = gaiamock.read_in_C_functions()
    data = synthetic_epoch_astrometry(rng, c_funcs, period=150.0, ecc=0.7, a0_mas=1.0)
    n = 3000
    P = np.exp(rng.uniform(np.log(10), np.log(1e4), n)); phi_p = rng.uniform(0, 2 * np.pi, n)
    ie = rng.integers(0, 5, n).astype(np.int32); ecc = np.array([0.2, 0.4, 0.6, 0.8, 0.95])[ie]
    arrs = [np.ascontiguousarray(a, dtype=np.double) for a in data]
    approx = np.zeros(n)
    c_funcs.get_chi2_astrometry_grid_approx(ctypes.c_int(len(arrs[0])), *[ctypes.c_void_p(a.ctypes.data) for a in arrs],
                                            ctypes.c_int(n), *[ctypes.c_void_p(a.ctypes.data) for a in (P, phi_p, ie)],
                                            ctypes.c_void_p(approx.ctypes.data))
    exact = fast_chi2_sequence(c_funcs, data, P, phi_p, ecc, False)
    ok = np.isfinite(approx)
    assert ok.mean() > 0.99
    np.testing.assert_allclose(approx[ok], exact[ok], rtol=2e-3)


@pytest.mark.parametrize('reject_outlier', [False, True])
def test_v2_approximate_grid_matches_exact_grid(reject_outlier):
    '''
    periodogram_multistart_v2 ranks its eccentric seed grid with an approximate chi2 (tabulated Kepler solutions) and
    rescores the best cells exactly; it should reach the same solution as scoring every cell exactly
    '''
    c_funcs = gaiamock.read_in_C_functions()
    L = np.array([10.0, 0.0, 0.0]); U = np.array([1e4, 2 * np.pi, 0.99])
    for k, (period, ecc, a0) in enumerate([(23.0, 0.1, 1.5), (300.0, 0.6, 1.0), (850.0, 0.9, 2.0), (2500.0, 0.4, 3.0),
                                           (60.0, 0.95, 0.5)]):
        data = synthetic_epoch_astrometry(np.random.default_rng(100 + k), c_funcs, period=period, ecc=ecc, a0_mas=a0)
        if reject_outlier:
            data[3][7] += 1.0
        arrs = [np.ascontiguousarray(a, dtype=np.double) for a in data]
        out = {}
        for name in ['run_astfit_periodogram_multistart_v2_nm', 'run_astfit_periodogram_multistart_v2_exactgrid']:
            r = np.zeros(4)
            getattr(c_funcs, name)(*[ctypes.c_void_p(a.ctypes.data) for a in arrs + [L, U, r]],
                                   ctypes.c_int(len(arrs[0])), ctypes.c_int(int(reject_outlier)))
            out[name] = r
        np.testing.assert_allclose(out['run_astfit_periodogram_multistart_v2_nm'],
                                   out['run_astfit_periodogram_multistart_v2_exactgrid'], rtol=1e-6, atol=1e-6)


def test_v2_levenberg_marquardt_matches_nelder_mead():
    '''
    periodogram_multistart_v2 refines its starts with Levenberg-Marquardt; on clear orbits it should reach the same
    minimum as refining every start with Nelder-Mead (periodogram_multistart_v2_nm)
    '''
    c_funcs = gaiamock.read_in_C_functions()
    for k, (period, ecc, a0) in enumerate([(23.0, 0.1, 1.5), (300.0, 0.6, 1.0), (850.0, 0.9, 2.0), (2500.0, 0.4, 3.0),
                                           (60.0, 0.95, 0.5), (140.0, 0.0, 0.8)]):
        data = synthetic_epoch_astrometry(np.random.default_rng(200 + k), c_funcs, period=period, ecc=ecc, a0_mas=a0)
        lm = gaiamock.fit_orbital_solution_nonlinear(*data, c_funcs)
        nm = gaiamock.fit_orbital_solution_nonlinear(*data, c_funcs, optimizer='periodogram_multistart_v2_nm')
        chi2_lm = gaiamock.get_astrometric_chi2(*data, *lm, c_funcs)[0]
        chi2_nm = gaiamock.get_astrometric_chi2(*data, *nm, c_funcs)[0]
        assert chi2_lm < chi2_nm + 1e-3
        assert abs(np.log(lm[0] / nm[0])) < 1e-3 and abs(lm[2] - nm[2]) < 1e-2


def test_clip_removes_outliers():
    '''
    reject_outlier="clip" removes several outliers, including one that the plain search absorbs into an e = 0.99
    spike (found by the transit-clump prescreen), and returns the fit to the remaining epochs
    '''
    c_funcs = gaiamock.read_in_C_functions()
    data = [np.array(a) for a in synthetic_epoch_astrometry(np.random.default_rng(11), c_funcs, period=300.0, ecc=0.3, a0_mas=0.5)]
    bad = [5, 30, 31, 60]
    data[3][bad] += np.array([1.5, -2.0, 1.0, -1.2])    # 20-40 sigma
    res, keep = gaiamock.fit_orbital_solution_nonlinear(*data, c_funcs, reject_outlier='clip', return_mask=True)
    np.testing.assert_array_equal(np.flatnonzero(~keep), bad)
    good = [a[keep] for a in data]
    np.testing.assert_allclose(res, gaiamock.fit_orbital_solution_nonlinear(*good, c_funcs), rtol=1e-10)


def test_clip_restores_good_epochs():
    '''
    on clean data clipping removes nothing, even when a fast, large orbit makes transits of the same visit disagree
    and the prescreen flags them: the clipping rounds restore every epoch that fits the orbit
    '''
    c_funcs = gaiamock.read_in_C_functions()
    for period, a0 in [(300.0, 1.0), (10.5, 5.0)]:
        data = [np.array(a) for a in synthetic_epoch_astrometry(np.random.default_rng(3), c_funcs, period=period, ecc=0.2, a0_mas=a0)]
        res, keep = gaiamock.fit_orbital_solution_nonlinear(*data, c_funcs, reject_outlier='clip', return_mask=True)
        assert keep.all()
        np.testing.assert_allclose(res, gaiamock.fit_orbital_solution_nonlinear(*data, c_funcs), rtol=1e-10)


def test_cascade_with_clip():
    '''the cascade's orbital solution uses the kept epochs and reports them'''
    c_funcs = gaiamock.read_in_C_functions()
    data = [np.array(a) for a in synthetic_epoch_astrometry(np.random.default_rng(11), c_funcs, period=300.0, ecc=0.3, a0_mas=1.0)]
    data[3][[5, 30]] += np.array([1.5, -2.0])
    res, keep = gaiamock.fit_full_astrometric_cascade(*data, c_funcs, reject_outlier='clip', skip_acceleration=True,
                                                     ruwe_min=0, return_mask=True)
    np.testing.assert_array_equal(np.flatnonzero(~keep), [5, 30])
    assert res[20] == keep.sum()                    # N_obs
    assert abs(res[10] / 300.0 - 1) < 0.01          # period
    with pytest.raises(ValueError):
        gaiamock.fit_orbital_solution_nonlinear(*data, c_funcs, reject_outlier='clip', optimizer='periodogram_multistart')


@pytest.mark.parametrize('reject_outlier', [False, True])
def test_previous_optimizer_matches_gsl_version(reject_outlier):
    '''
    periodogram_multistart runs the same search as before, only with the faster chi2, so it should land on the same
    solution as the GSL implementation
    '''
    rng = np.random.default_rng(7)
    c_funcs = gaiamock.read_in_C_functions()
    data = synthetic_epoch_astrometry(rng, c_funcs, period=450.0, ecc=0.5, a0_mas=1.0)
    fast = gaiamock.fit_orbital_solution_nonlinear(*data, c_funcs, reject_outlier=reject_outlier,
                                                   optimizer='periodogram_multistart')
    slow = gaiamock.fit_orbital_solution_nonlinear(*data, c_funcs, reject_outlier=reject_outlier,
                                                   optimizer='periodogram_multistart_gsl')
    np.testing.assert_allclose(fast, slow, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('fit_name', ['fit_orbital_solution_nonlinear', 'fit_full_astrometric_cascade'])
@pytest.mark.parametrize('kwargs', [{}, {'optimizer': None}, {'optimizer': 'default'}], ids=['omitted', 'none', 'default'])
def test_default_optimizer_uses_v2(fit_name, kwargs, monkeypatch):
    '''Both public entry points must dispatch all default spellings to the real compiled v2 search.'''
    c_funcs = gaiamock.read_in_C_functions()
    data = synthetic_epoch_astrometry(np.random.default_rng(42), c_funcs)
    run_v2 = c_funcs.run_astfit_periodogram_multistart_v2
    calls = []

    def record_v2(*args):
        calls.append(True)
        return run_v2(*args)

    monkeypatch.setattr(c_funcs, 'run_astfit_periodogram_multistart_v2', record_v2)
    options = dict(kwargs)
    if fit_name == 'fit_full_astrometric_cascade':
        options['skip_acceleration'] = True
    result = getattr(gaiamock, fit_name)(*data, c_funcs, **options)
    assert len(calls) == 1
    assert np.all(np.isfinite(result))


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
