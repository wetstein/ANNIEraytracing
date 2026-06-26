"""Optical surface physics for multi-bounce ray tracing.

Provides per-material config loading, Fresnel equations, diffuse/specular
reflection models, and the hit-evaluation dispatcher used by
:func:`trace_with_optics` in *tracer.py*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from annieray.materials import MaterialID

# ---------------------------------------------------------------------------
# Material optical config
# ---------------------------------------------------------------------------


@dataclass
class OpticalMaterial:
    """Optical surface properties for a single material.

    Parameters
    ----------
    model:
        Interaction model: ``"fresnel"`` (dielectric interface with
        Snell's law + unpolarised Fresnel), ``"reflect"`` (probability-
        based reflection with specular/diffuse mix), or ``"absorb"``
        (always terminate).
    refractive_index:
        Real refractive index (only used for ``"fresnel"`` model).
    reflectivity:
        Reflection probability for the ``"reflect"`` model (0 = black,
        1 = perfect mirror).
    diffuse_fraction:
        Fraction of reflected light that follows Lambertian (cosine-
        weighted) distribution; the remainder is specular (mirror).
    is_sensitive:
        If ``True``, Fresnel transmission is recorded as a detected
        hit (photocathode-like).
    """
    model: str
    refractive_index: float | None = None
    reflectivity: float = 0.0
    diffuse_fraction: float = 0.0
    is_sensitive: bool = False


_BUILTIN_CONFIG: dict[int, OpticalMaterial] = {
    MaterialID.UNKNOWN:
        OpticalMaterial(model="absorb"),
    MaterialID.GLASS:
        OpticalMaterial(model="fresnel", refractive_index=1.50),
    MaterialID.PHOTOCATHODE:
        OpticalMaterial(model="fresnel", refractive_index=1.50, is_sensitive=True),
    MaterialID.PVC:
        OpticalMaterial(model="reflect", reflectivity=0.05, diffuse_fraction=0.50),
    MaterialID.TEFLON:
        OpticalMaterial(model="reflect", reflectivity=0.98, diffuse_fraction=0.80),
    MaterialID.ACRYLIC:
        OpticalMaterial(model="fresnel", refractive_index=1.49),
    MaterialID.STEEL:
        OpticalMaterial(model="reflect", reflectivity=0.30, diffuse_fraction=0.10),
    MaterialID.BLACK_SHEET:
        OpticalMaterial(model="reflect", reflectivity=0.02, diffuse_fraction=0.50),
}


def load_optics_config(path: str | Path | None = None) -> dict[int, OpticalMaterial]:
    """Load optical material config from a YAML file, or return built-in defaults.

    The YAML file should have the structure shown in
    ``optics_default.yaml``, with a top-level ``materials`` key mapping
    material names (matching the :class:`MaterialID` enum names) to
    per-material parameter dicts.

    Any material not present in the file keeps its built-in default.
    """
    if path is None:
        return dict(_BUILTIN_CONFIG)

    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = dict(_BUILTIN_CONFIG)
    materials = raw.get("materials", {})
    for name, params in materials.items():
        try:
            mid = MaterialID[name]
        except KeyError:
            continue  # unknown name — keep default

        model = params.get("model", cfg[mid].model)
        if model == "fresnel":
            cfg[mid] = OpticalMaterial(
                model=model,
                refractive_index=params.get("refractive_index", cfg[mid].refractive_index),
                is_sensitive=params.get("is_sensitive", cfg[mid].is_sensitive),
            )
        elif model == "reflect":
            cfg[mid] = OpticalMaterial(
                model=model,
                reflectivity=params.get("reflectivity", cfg[mid].reflectivity),
                diffuse_fraction=params.get("diffuse_fraction", cfg[mid].diffuse_fraction),
            )
        else:
            cfg[mid] = OpticalMaterial(model="absorb")
    return cfg


# ---------------------------------------------------------------------------
# Surface physics helpers
# ---------------------------------------------------------------------------


def fresnel_reflectance(n1: float, n2: float, cos_i: float) -> float:
    """Unpolarised Fresnel reflectance at a dielectric interface.

    Parameters
    ----------
    n1:
        Refractive index of the incident medium (water).
    n2:
        Refractive index of the transmitting medium (glass / acrylic).
    cos_i:
        Cosine of the incident angle  (always ≥ 0; cos_i = 1 is
        normal incidence).

    Returns
    -------
    R:
        Fraction of power reflected (0 … 1).  The transmitted fraction
        is ``1 - R``.
    """
    sin_t2 = (n1 / n2) ** 2 * (1.0 - cos_i * cos_i)
    if sin_t2 > 1.0:
        return 1.0  # total internal reflection
    cos_t = np.sqrt(1.0 - sin_t2)
    r_s = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2
    r_p = ((n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)) ** 2
    return 0.5 * (r_s + r_p)


def snell_refract_direction(
    incident: np.ndarray,
    normal: np.ndarray,
    n1: float,
    n2: float,
    cos_i: float,
) -> np.ndarray:
    """Compute transmitted (refracted) direction via Snell's law.

    ``incident`` points *toward* the surface; ``normal`` points
    *out of* the surface (away from the material).  Both are unit
    vectors.
    """
    sin_t2 = (n1 / n2) ** 2 * (1.0 - cos_i * cos_i)
    cos_t = np.sqrt(max(1.0 - sin_t2, 0.0))
    return (n1 / n2) * incident + ((n1 / n2) * cos_i - cos_t) * normal


def specular_reflect(incident: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Perfect mirror reflection direction."""
    return incident - 2.0 * np.dot(incident, normal) * normal


def cosine_weighted_hemisphere(normal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample a direction from the cosine-weighted (Lambertian) hemisphere.

    The returned direction is a unit vector distributed as
    ``f(theta) = cos(theta) / pi``.
    """
    u1, u2 = rng.uniform(0.0, 1.0, 2)
    theta = np.arccos(np.sqrt(1.0 - u1))
    phi = 2.0 * np.pi * u2

    local_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal, local_up)) > 0.99:
        local_up = np.array([1.0, 0.0, 0.0])
    tangent = np.cross(normal, local_up)
    t_len = np.linalg.norm(tangent)
    if t_len < 1e-12:
        tangent = np.array([1.0, 0.0, 0.0])
    else:
        tangent /= t_len
    bitangent = np.cross(normal, tangent)

    return (
        np.sin(theta) * np.cos(phi) * tangent
        + np.sin(theta) * np.sin(phi) * bitangent
        + np.cos(theta) * normal
    )


# ---------------------------------------------------------------------------
# Hit evaluator
# ---------------------------------------------------------------------------


def evaluate_hit(
    mat_opt: OpticalMaterial,
    incident_dir: np.ndarray,
    normal: np.ndarray,
    n_water: float,
    rng: np.random.Generator,
) -> tuple[str, np.ndarray | None]:
    """Decide what happens when a photon hits a material surface.

    Parameters
    ----------
    mat_opt:
        Optical properties of the struck material.
    incident_dir:
        Unit direction of the incoming photon (points *toward* surface).
    normal:
        Outward-facing unit normal at the hit point.
    n_water:
        Refractive index of water.
    rng:
        NumPy random generator.

    Returns
    -------
    (action, new_dir):
        ``action`` is one of ``"absorb"``, ``"detect"``, or ``"reflect"``.
        For ``"reflect"``, ``new_dir`` is the outgoing direction (unit);
        otherwise ``new_dir`` is ``None``.
    """
    if mat_opt.model == "absorb":
        return "absorb", None

    if mat_opt.model == "fresnel":
        n1 = n_water
        n2 = mat_opt.refractive_index or 1.5
        cos_i = -float(np.dot(incident_dir, normal))
        if cos_i < 0.0:
            normal = -normal
            cos_i = -cos_i
        R = fresnel_reflectance(n1, n2, cos_i)
        if rng.random() < R:
            new_dir = specular_reflect(incident_dir, normal)
            return "reflect", new_dir
        if mat_opt.is_sensitive:
            return "detect", None
        return "absorb", None

    if mat_opt.model == "reflect":
        if rng.random() < mat_opt.reflectivity:
            if rng.random() < mat_opt.diffuse_fraction:
                new_dir = cosine_weighted_hemisphere(normal, rng)
            else:
                new_dir = specular_reflect(incident_dir, normal)
            return "reflect", new_dir
        return "absorb", None

    return "absorb", None
