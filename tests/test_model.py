# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Tests of rivoli.model, the radial basis function models of BRDFs."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rivoli.colour import spectrum_to_rgb
from rivoli.dirichlet import dirichlet_matrix
from rivoli.distances import diameters
from rivoli.geometry import spherical_to_cartesian
from rivoli.kernels import collocation_matrix
from rivoli.model import RBFModel, fit
from rivoli.sampling import super_fibonacci_isotropic

ARGUMENTS = {"widths": (0.25, 0.4), "metric": "chordal", "isotropic": True}


def isotropic_sites(n):
    theta_h, theta_d, phi_d = super_fibonacci_isotropic(n)
    sites = (spherical_to_cartesian(theta_h, 0.0), spherical_to_cartesian(theta_d, phi_d))
    return sites, brdf(theta_h, theta_d)


def brdf(theta_h, theta_d):
    """A specular peak over a diffuse part, of a large dynamic range."""
    return 0.05 + 20 * np.exp(-((theta_h / 0.15) ** 2)) * np.cos(theta_d)


@pytest.mark.parametrize("log_space", [True, False])
def test_zero_regularization_interpolates_the_centres(log_space):
    centres, values = isotropic_sites(300)
    model = fit(centres, values, centres, regularization=0.0, log_space=log_space, **ARGUMENTS)
    assert_allclose(model(centres), values, rtol=1e-8)


def test_model_evaluates_the_exponential_of_the_expansion():
    centres, _ = isotropic_sites(100)
    sites, values = isotropic_sites(600)
    model = fit(sites, values, centres, regularization=0.1, **ARGUMENTS)
    assert model.log_offset == 1e-6
    collocation = collocation_matrix(sites, centres, widths=(0.25, 0.4), metric="chordal")
    assert_allclose(model(sites), np.exp(collocation @ model.weights) - 1e-6)
    linear = fit(sites, values, centres, regularization=0.1, log_space=False, **ARGUMENTS)
    assert linear.log_offset is None
    assert_allclose(linear(sites), collocation @ linear.weights)


def test_model_is_evaluated_by_chunks(monkeypatch):
    centres, _ = isotropic_sites(100)
    sites, values = isotropic_sites(600)
    model = fit(sites, values, centres, regularization=0.1, **ARGUMENTS)
    expected = model(sites)
    # Chunks of 7 sites, the last one being partial
    monkeypatch.setattr("rivoli.model._ENTRIES_PER_CHUNK", 700)
    assert_allclose(model(sites), expected, rtol=1e-14)


def test_log_space_fits_the_logarithm_of_the_values():
    centres, _ = isotropic_sites(100)
    sites, values = isotropic_sites(600)
    model = fit(sites, values, centres, regularization=0.1, **ARGUMENTS)
    linear = fit(
        sites, np.log(values + 1e-6), centres, regularization=0.1, log_space=False, **ARGUMENTS
    )
    assert_allclose(model.weights, linear.weights)


def test_regularization_smooths_the_fit():
    centres, _ = isotropic_sites(200)
    sites, values = isotropic_sites(1000)
    energy = dirichlet_matrix(centres, **ARGUMENTS)
    energies = []
    for regularization in (0.0, 1e-3, 0.1, 0.9):
        model = fit(sites, values, centres, regularization=regularization, **ARGUMENTS)
        energies.append(model.weights @ energy @ model.weights)
    assert np.all(np.diff(energies) < 0)


def test_given_energy_is_used():
    centres, _ = isotropic_sites(100)
    sites, values = isotropic_sites(500)
    scales = diameters(isotropic=True, metric="chordal")
    energy = dirichlet_matrix(centres, scales=scales, **ARGUMENTS)
    computed = fit(sites, values, centres, regularization=0.2, scales=scales, **ARGUMENTS)
    given = fit(sites, values, centres, regularization=0.2, energy=energy, **ARGUMENTS)
    assert_allclose(given.weights, computed.weights)
    unscaled = fit(sites, values, centres, regularization=0.2, **ARGUMENTS)
    assert not np.allclose(unscaled.weights, computed.weights)


def test_channels_are_fitted_independently():
    centres, _ = isotropic_sites(100)
    sites, values = isotropic_sites(500)
    channels = np.stack([values, 2 * values, values**2], axis=1)
    model = fit(sites, channels, centres, regularization=0.2, **ARGUMENTS)
    assert model.weights.shape == (100, 3)
    for c in range(3):
        alone = fit(sites, channels[:, c], centres, regularization=0.2, **ARGUMENTS)
        assert_allclose(model.weights[:, c], alone.weights, rtol=1e-10, atol=1e-12)


def test_spectral_model():
    angles = super_fibonacci_isotropic(400, wavelength_range=(380.0, 780.0))
    theta_h, theta_d, phi_d, wavelength = angles
    sites = (
        spherical_to_cartesian(theta_h, 0.0),
        spherical_to_cartesian(theta_d, phi_d),
        wavelength,
    )
    values = brdf(theta_h, theta_d) * (1 + np.sin(wavelength / 50))
    centres = tuple(x[::4] for x in sites)
    model = fit(
        sites,
        values,
        centres,
        widths=(0.3, 0.5, 60.0),
        metric="chordal",
        isotropic=True,
        regularization=1e-4,
        wavelength_range=(380.0, 780.0),
        scales=diameters(isotropic=True, metric="chordal", wavelength_range=(380.0, 780.0)),
    )
    assert len(model.centres) == 3
    assert model(sites).shape == (400,)


def spectral_model(channels=(), log_space=True):
    """A spectral model, with random weights."""
    angles = super_fibonacci_isotropic(60, wavelength_range=(380.0, 780.0))
    theta_h, theta_d, phi_d, wavelength = angles
    centres = (
        spherical_to_cartesian(theta_h, 0.0),
        spherical_to_cartesian(theta_d, phi_d),
        wavelength,
    )
    weights = np.random.default_rng(20261029).normal(scale=0.3, size=(60, *channels))
    return RBFModel(
        centres=centres,
        weights=weights,
        widths=(0.3, 0.5, 60.0),
        metric="chordal",
        kernel="gaussian",
        log_offset=1e-6 if log_space else None,
    )


@pytest.mark.parametrize("log_space", [True, False])
@pytest.mark.parametrize("channels", [(), (2,)])
def test_spectra_are_the_model_at_each_wavelength(monkeypatch, log_space, channels):
    model = spectral_model(channels, log_space)
    (h, d), _ = isotropic_sites(50)
    wavelengths = np.linspace(380.0, 780.0, 7)
    spectra = model.spectra((h, d), wavelengths)
    assert spectra.shape == (50, 7, *channels)
    for k, wavelength in enumerate(wavelengths):
        expected = model((h, d, np.full(50, wavelength)))
        assert_allclose(spectra[:, k], expected, rtol=1e-12, atol=1e-14)
    monkeypatch.setattr("rivoli.model._ENTRIES_PER_CHUNK", 1000)
    assert_allclose(model.spectra((h, d), wavelengths), spectra, rtol=1e-14)


def test_rgb_of_a_spectral_model():
    model = spectral_model()
    (h, d), _ = isotropic_sites(50)
    wavelengths = np.arange(380.0, 781.0, 5.0)
    rgb = model.rgb((h, d), wavelengths, illuminant="E")
    assert rgb.shape == (50, 3)
    expected = spectrum_to_rgb(wavelengths, model.spectra((h, d), wavelengths), illuminant="E")
    assert_allclose(rgb, expected, rtol=1e-12)


def test_invalid_spectral_evaluations():
    (h, d), _ = isotropic_sites(10)
    wavelengths = np.linspace(380.0, 780.0, 5)
    model = spectral_model()
    with pytest.raises(ValueError, match="without wavelengths, got 3 factors"):
        model.spectra((h, d, np.full(10, 500.0)), wavelengths)
    with pytest.raises(ValueError, match="shape \\(k,\\)"):
        model.spectra((h, d), wavelengths[:, np.newaxis])
    with pytest.raises(ValueError, match="several channels"):
        spectral_model((3,)).rgb((h, d), wavelengths)
    centres, values = isotropic_sites(20)
    rgb = fit(centres, values, centres, regularization=0.0, **ARGUMENTS)
    with pytest.raises(ValueError, match="not spectral"):
        rgb.spectra(centres, wavelengths)


def test_invalid_values():
    centres, values = isotropic_sites(20)
    with pytest.raises(ValueError, match="above -1e-06"):
        fit(centres, values - 1, centres, regularization=0.5, **ARGUMENTS)
    with pytest.raises(ValueError, match="expected values of shape \\(20,\\)"):
        fit(centres, values[1:], centres, regularization=0.5, **ARGUMENTS)
    with pytest.raises(ValueError, match="must be in \\[0, 1\\)"):
        fit(centres, values, centres, regularization=1.0, **ARGUMENTS)
