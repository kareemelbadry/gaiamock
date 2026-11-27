'''
Optional portable GPU/TPU backend for low-SNR rejection-sampling likelihoods using JAX (FP64).

Runs on NVIDIA CUDA, AMD ROCm, Google TPU, or CPU whenever `jax` is installed, with zero
compiled GPU extensions. Ill-conditioned or ultra-high-SNR samples (typically <1%) fall back
automatically to the exact C Gram-Schmidt solver so results match GSL to machine precision.
'''
import numpy as np

_JAX_STATE = None


def _load_jax():
    global _JAX_STATE
    if _JAX_STATE is not None:
        return _JAX_STATE
    try:
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            'backend="gpu" requires JAX. Install with `pip install jax` (or `jax[cuda12]` for NVIDIA GPUs).'
        ) from exc

    pi = np.pi
    two_pi = 2.0 * np.pi
    en = jnp.arange(13, dtype=jnp.float64) * (pi / 12.0)
    sin_en = jnp.sin(en)
    cos_en = jnp.cos(en)

    def short_sin(x):
        x2 = x * x
        p = 1.0 / 6227020800.0 - x2 * (1.0 / 1307674368000.0)
        p = 1.0 / 39916800.0 - x2 * p
        p = 1.0 / 362880.0 - x2 * p
        p = 1.0 / 5040.0 - x2 * p
        p = 1.0 / 120.0 - x2 * p
        p = 1.0 / 6.0 - x2 * p
        return x * (1.0 - x2 * p)

    def kepler_orvara(a_arr, P, phi_p, e):
        bounds = en - e * sin_en
        d1 = 1.0 / (1.0 - e * cos_en)
        d2 = -0.5 * e * sin_en * d1 * d1 * d1
        idx = 1.0 / (bounds[1:] - bounds[:-1])
        a1, a2 = d1[:-1], d2[:-1]
        b0 = idx * (-a2 - idx * (a1 - idx * (pi / 12.0)))
        b1 = idx * (-2.0 * a2 - idx * (a1 - d1[1:]))
        b2 = idx * (d2[1:] - a2)
        c0 = en[:-1]
        c3 = b2 - 4.0 * b1 + 10.0 * b0
        c4 = (-2.0 * b2 + 7.0 * b1 - 15.0 * b0) * idx
        c5 = (b2 - 3.0 * b1 + 6.0 * b0) * idx * idx

        mi = a_arr / P - phi_p
        m = mi - two_pi * jnp.floor(mi / two_pi)
        lower = m > pi
        m = jnp.where(lower, two_pi - m, m)

        k = jnp.clip(jnp.sum(m[:, None] > bounds[None, 1:12], axis=1), 0, 11)
        x = m - bounds[k]
        e_poly = c0[k] + x * (a1[k] + x * (a2[k] + x * (c3[k] + x * (c4[k] + x * c5[k]))))

        ome = jnp.maximum(1.0 - e, 1e-15)
        sq_ome = jnp.sqrt(ome)
        chi = m / (sq_ome * ome)
        lam = jnp.sqrt(8.0 + 9.0 * chi * chi)
        s_cb = jnp.cbrt(lam + 3.0 * chi)
        s2 = s_cb * s_cb
        sigma = 6.0 * chi / (2.0 + s2 + 4.0 / s2)
        sig2 = sigma * sigma
        den_c = sig2 + 2.0
        poly_c = sig2 * (sig2 * (sig2 + 25.0) + 340.0) + 840.0
        series = (sig2 + 20.0) / (60.0 * den_c) + sig2 * ome * poly_c / (1400.0 * den_c * den_c * den_c)
        e_corner = sigma * (1.0 + sig2 * ome * series) * sq_ome

        ea = jnp.where(2.0 * m + (1.0 - e) <= 0.2, e_corner, e_poly)

        near0 = ea <= (pi / 4.0)
        nearpi = ea >= (3.0 * pi / 4.0)
        arg = jnp.where(near0, ea, jnp.where(nearpi, pi - ea, 0.5 * pi - ea))
        s_val = short_sin(arg)
        c_val = jnp.sqrt(jnp.maximum(1.0 - s_val * s_val, 0.0))
        sin_e = jnp.where(near0 | nearpi, s_val, c_val)
        cos_e = jnp.where(near0, c_val, jnp.where(nearpi, -c_val, s_val))

        num = m - ea + e * sin_e
        den = 1.0 - e * cos_e
        es, ec = e * sin_e, e * cos_e
        de = num * (den * den + 0.5 * num * es) / (den * den * den + num * (den * es + num * ec / 6.0))
        q = de * de / 6.0
        cd, sd = 1.0 - 3.0 * q, de * (1.0 - q)
        s2_rot = sin_e * cd + cos_e * sd
        c2_rot = cos_e * cd - sin_e * sd
        s_e = jnp.where(lower, -s2_rot, s2_rot)
        c_e = c2_rot

        is_zero = e == 0.0
        return jnp.where(is_zero, jnp.cos(mi), c_e), jnp.where(is_zero, jnp.sin(mi), s_e)

    def lnL_one(a_arr, W, s2, sc, c2, rr0, P, phi_p, e):
        c_e, s_e = kepler_orvara(a_arr, P, phi_p, e)
        x = c_e - e
        y = jnp.sqrt(jnp.maximum(1.0 - e * e, 0.0)) * s_e

        ax = x @ W
        ay = y @ W
        xx, xy, yy = x * x, x * y, y * y
        h0, h1, h2 = jnp.dot(xx, s2), jnp.dot(xy, s2), jnp.dot(yy, s2)
        h3, h4, h5 = jnp.dot(xx, sc), jnp.dot(xy, sc), jnp.dot(yy, sc)
        h6, h7, h8 = jnp.dot(xx, c2), jnp.dot(xy, c2), jnp.dot(yy, c2)

        H = jnp.array([[h0, h1, h3, h4], [h1, h2, h4, h5], [h3, h4, h6, h7], [h4, h5, h7, h8]])
        G = jnp.stack([ax[:5], ay[:5], ax[5:10], ay[5:10]], axis=1)
        b = jnp.array([ax[10], ay[10], ax[11], ay[11]])
        S = H - G.T @ G
        dh = jnp.diag(H)

        l00_sq = S[0, 0]
        ok0 = (dh[0] > 0.0) & (l00_sq > 1e-4 * dh[0])
        l00 = jnp.sqrt(jnp.where(ok0, l00_sq, 1.0))

        l10 = S[1, 0] / l00
        l11_sq = S[1, 1] - l10 * l10
        ok1 = (dh[1] > 0.0) & (l11_sq > 1e-4 * dh[1])
        l11 = jnp.sqrt(jnp.where(ok1, l11_sq, 1.0))

        l20 = S[2, 0] / l00
        l21 = (S[2, 1] - l20 * l10) / l11
        l22_sq = S[2, 2] - l20 * l20 - l21 * l21
        ok2 = (dh[2] > 0.0) & (l22_sq > 1e-4 * dh[2])
        l22 = jnp.sqrt(jnp.where(ok2, l22_sq, 1.0))

        l30 = S[3, 0] / l00
        l31 = (S[3, 1] - l30 * l10) / l11
        l32 = (S[3, 2] - l30 * l20 - l31 * l21) / l22
        l33_sq = S[3, 3] - l30 * l30 - l31 * l31 - l32 * l32
        ok3 = (dh[3] > 0.0) & (l33_sq > 1e-4 * dh[3])
        l33 = jnp.sqrt(jnp.where(ok3, l33_sq, 1.0))

        z0 = b[0] / l00
        z1 = (b[1] - l10 * z0) / l11
        z2 = (b[2] - l20 * z0 - l21 * z1) / l22
        z3 = (b[3] - l30 * z0 - l31 * z1 - l32 * z2) / l33
        chi2 = rr0 - (z0 * z0 + z1 * z1 + z2 * z2 + z3 * z3)
        ok = ok0 & ok1 & ok2 & ok3 & (chi2 >= 1e-6 * rr0)
        return jnp.where(ok, -0.5 * chi2, jnp.nan)

    batch_fn = jax.jit(jax.vmap(lnL_one, in_axes=(None, None, None, None, None, None, 0, 0, 0)))
    _JAX_STATE = (jax, jnp, batch_fn)
    return _JAX_STATE


def has_gpu_accelerator():
    try:
        jax, _, _ = _load_jax()
        return any(d.platform in ('gpu', 'tpu') for d in jax.devices())
    except Exception:
        return False


def _build_workspace(t_ast_yr, psi, plx_factor, ast_obs, ast_err):
    t = np.asarray(t_ast_yr, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    pf = np.asarray(plx_factor, dtype=np.float64)
    y = np.asarray(ast_obs, dtype=np.float64)
    err = np.asarray(ast_err, dtype=np.float64)
    iw = 1.0 / err
    a = 2.0 * np.pi * t * 365.25
    sw, cw, r0 = np.sin(psi) * iw, np.cos(psi) * iw, y * iw
    Q = np.zeros((5, len(t)), dtype=np.float64)
    nq = 0
    for col in (sw, t * sw, cw, t * cw, pf * iw):
        v = col.copy()
        nrm0 = np.linalg.norm(v)
        for _ in range(2):
            for k in range(nq):
                v -= np.dot(Q[k], v) * Q[k]
        nrm = np.linalg.norm(v)
        if nrm0 > 0.0 and (nrm / nrm0) > 1e-10:
            Q[nq] = v / nrm
            nq += 1
    for _ in range(2):
        for k in range(nq):
            r0 -= np.dot(Q[k], r0) * Q[k]
    W = np.zeros((len(t), 12), dtype=np.float64)
    for k in range(nq):
        W[:, k] = Q[k] * sw
        W[:, 5 + k] = Q[k] * cw
    W[:, 10] = r0 * sw
    W[:, 11] = r0 * cw
    return a, W, sw * sw, sw * cw, cw * cw, float(np.dot(r0, r0))


def get_astrometric_likelihoods_gpu(t_ast_yr, psi, plx_factor, ast_obs, ast_err, samples,
                                    chunk_size=65536, fallback_fn=None):
    '''
    Evaluate rejection-sampling log-likelihoods on a JAX accelerator (GPU/TPU) in FP64.
    Pads partial tail chunks to `chunk_size` so XLA compiles only once per `n_obs`.
    '''
    _, jnp, batch_fn = _load_jax()
    P, ecc, phi_p = (np.ascontiguousarray(x, dtype=np.float64) for x in samples)
    n = len(P)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    a, W, s2, sc, c2, rr0 = _build_workspace(t_ast_yr, psi, plx_factor, ast_obs, ast_err)
    a_j, W_j, s2_j, sc_j, c2_j = map(jnp.asarray, (a, W, s2, sc, c2))
    rr0_j = jnp.float64(rr0)

    out = np.empty(n, dtype=np.float64)
    csize = max(1, int(chunk_size))
    p_pad = np.empty(csize, dtype=np.float64)
    phi_pad = np.empty(csize, dtype=np.float64)
    e_pad = np.empty(csize, dtype=np.float64)

    for start in range(0, n, csize):
        end = min(start + csize, n)
        m = end - start
        if m == csize:
            res = batch_fn(a_j, W_j, s2_j, sc_j, c2_j, rr0_j,
                           jnp.asarray(P[start:end]), jnp.asarray(phi_p[start:end]), jnp.asarray(ecc[start:end]))
            out[start:end] = np.asarray(res)
        else:
            p_pad[:m], p_pad[m:] = P[start:end], 100.0
            phi_pad[:m], phi_pad[m:] = phi_p[start:end], 0.0
            e_pad[:m], e_pad[m:] = ecc[start:end], 0.0
            res = batch_fn(a_j, W_j, s2_j, sc_j, c2_j, rr0_j,
                           jnp.asarray(p_pad), jnp.asarray(phi_pad), jnp.asarray(e_pad))
            out[start:end] = np.asarray(res[:m])

    bad = np.where(~np.isfinite(out))[0]
    if len(bad) > 0 and fallback_fn is not None:
        out[bad] = fallback_fn(t_ast_yr, psi, plx_factor, ast_obs, ast_err, (P[bad], ecc[bad], phi_p[bad]))
    return out
