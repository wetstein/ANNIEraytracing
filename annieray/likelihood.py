"""Charge and time likelihood functions for track fitting."""

from __future__ import annotations

import math

import numpy as np


def poisson_charge_ll(
    observed: dict[int, int],
    expected: dict[int, float],
    all_pmt_indices: list[int],
    epsilon: float = 1e-10,
) -> float:
    """Poisson log-likelihood for per-PMT hit counts.

    ``LL = Σ_i [ n_i · ln(μ_i) − μ_i − ln(n_i!) ]``

    Parameters
    ----------
    observed : dict
        ``{pmt_index: n_hits}`` from ``pmt_responses``.
    expected : dict
        ``{pmt_index: expected_n_hits}`` from hypothesis raytracing.
    all_pmt_indices : list[int]
        Every PMT index in the detector (including zero-hit PMTs).
    epsilon : float
        Small offset to avoid ``ln(0)`` when a hypothesis predicts zero
        hits but the PMT observed hits (→ strongly penalised).

    PMTs absent from *observed* are treated as ``n_i = 0`` (penalising
    hypotheses that predict light where none was seen).  PMTs absent
    from *expected* are treated as ``μ_i = 0``.
    """
    ll = 0.0
    for idx in all_pmt_indices:
        n = observed.get(idx, 0)
        mu = expected.get(idx, 0.0)
        if n == 0:
            ll -= mu
        else:
            ll += n * math.log(max(mu, epsilon)) - mu - math.lgamma(n + 1)
    return ll


def time_residual_ll(
    observed_times: dict[int, float],
    expected_times_min: dict[int, float],
    hit_indices: list[int],
    sigma: float | dict[int, float] = 1.5,
) -> float:
    """Gaussian time-residual log-likelihood (omits constant terms).

    ``LL = −½ · Σ_i [ (t_i − t_exp_i)² / σ_i² ]``

    Only PMTs with *both* an observed and expected time contribute.
    """
    ll = 0.0
    for idx in hit_indices:
        t_obs = observed_times.get(idx)
        t_exp = expected_times_min.get(idx)
        if t_obs is None or t_exp is None:
            continue
        s = sigma.get(idx, 1.5) if isinstance(sigma, dict) else sigma
        r = t_obs - t_exp
        ll -= 0.5 * (r * r) / (s * s)
    return ll


def total_log_likelihood(
    charge_ll: float,
    time_ll: float,
    alpha: float = 1.0,
) -> float:
    """Combine charge and time log-likelihoods.

    ``LL = LL_charge + α · LL_time``
    """
    return charge_ll + alpha * time_ll


def compute_per_pmt_sigma(
    geometry,
    all_pmt_indices: list[int],
    fallback: float = 1.5,
) -> dict[int, float]:
    """Build a ``{pmt_index: transit_time_sigma}`` lookup from detector types.

    Uses ``PMT_TYPE_DEFAULTS`` when available; otherwise *fallback*.
    """
    from annieray.pmt_response import PMT_TYPE_DEFAULTS

    sigma: dict[int, float] = {}
    for d in geometry.detectors:
        if d.system == "pmt" and d.index in all_pmt_indices:
            defaults = PMT_TYPE_DEFAULTS.get(d.pmt_type)
            sigma[d.index] = defaults["transit_time_sigma"] if defaults else fallback
    for idx in all_pmt_indices:
        sigma.setdefault(idx, fallback)
    return sigma
