#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian Estimation Methods for iSCAT Axial Localization

This module provides Bayesian estimation functions for determining the 
defocus parameter (zf) in interferometric scattering (iSCAT) microscopy.

Author: weiliaoliao
License: MIT
"""

import numpy as np
import pymc as pm
from .parameters import iscat, sample


def estimate_zf_from_exp_ipsf(config_file, exp_ipsf,
                             wave_num, h_d, f_d, b, c, ampli_real, ampli_imag, 
                             E_r, noise_sd):
    """
    Estimate the zf parameter from input exp_ipsf using Bayesian fitting.

    Parameters:
        config_file (str): Path to configuration file
        exp_ipsf (numpy.ndarray): Input iPSF image data
        wave_num (float): Wave number
        h_d (ndarray): Coefficient for zf
        f_d (ndarray): Coefficient for zp
        b (float): Precomputed wave_num * f_d
        c (float): Precomputed wave_num * h_d
        ampli_real (float): Real part of amplitude
        ampli_imag (float): Imaginary part of amplitude
        E_r (float): Reference field amplitude
        noise_sd (float): Standard deviation of noise prior
        plot_trace (bool): Whether to plot posterior trace

    Returns:
        tuple: (zf_estimate, zp_estimate) - Estimated defocus and axial position
    """
    def approximate_model_pymc(zp, zf):
        """PyMC-compatible approximate model function that expands the complex values into real and imaginary parts"""
        deltax = wave_num * (np.std(f_d) * zp * 1e-6 + np.std(h_d) * zf * 1e-6)
        exp_real = np.cos(b * zp * 1e-6 + c * zf * 1e-6)
        exp_imag = np.sin(b * zp * 1e-6 + c * zf * 1e-6)
        term_real = -0.5 * deltax**2 + 1
        term_imag = deltax
        sca_real = (ampli_real * term_real - ampli_imag * term_imag) * exp_real - \
                   (ampli_real * term_imag + ampli_imag * term_real) * exp_imag
        sca_imag = (ampli_real * term_real - ampli_imag * term_imag) * exp_imag + \
                   (ampli_real * term_imag + ampli_imag * term_real) * exp_real
        sca_magnitude_squared = sca_real**2 + sca_imag**2
        interference_term = 2 * E_r * sca_real
        ref_intensity = E_r**2
        img = ref_intensity + sca_magnitude_squared + interference_term
        return img.flatten()
    
    
    m = iscat(config_file)
    s = sample(config_file)

    # Get center point values for later normalization
    exp_ipsf0 = exp_ipsf[0, :, :]
    nx = exp_ipsf0.shape[0]
    nx_half = nx // 2    
    exp_ipsf_center = exp_ipsf0[nx_half, nx_half]

    # Flatten and convert data for fitting
    exp_ipsf0 = np.array(exp_ipsf0).flatten().astype(np.float64)

    # Set up multiple chains with different random seeds
    n_chains = 4
    base_seed = 42
    random_seeds = [base_seed + i for i in range(n_chains)]
    
    with pm.Model() as model:        
        # Define priors
        zf_prior = pm.Normal("zf_estimated", mu=m.zf_initial, sigma=m.zf_range)  # System defocus (unit: um)
        zp_prior = pm.Normal("zp_estimated", mu=s.zp_initial, sigma=s.zp_range)  # Unit: um
        sigma_prior = pm.HalfNormal("sigma", sigma=noise_sd)  # Noise prior

        # Calculate the modeled iPSF pattern
        y_pred = approximate_model_pymc(zp_prior, zf_prior)
        
        # Calculate normalization factor based on center point ratio
        y_pred = y_pred.reshape([nx, nx])
        y_center = y_pred[nx_half, nx_half]
        normalization_factor = exp_ipsf_center / y_center
        y_pred = y_pred * normalization_factor
        y_pred = y_pred.flatten()
        
        # Define likelihood
        likelihood = -0.5 * pm.math.sum(((exp_ipsf0 - y_pred) / sigma_prior)**2)
        pm.Potential("likelihood", likelihood)

        # Sample the posterior distribution
        trace = pm.sample(10000, random_seed=random_seeds)

    # Extract posterior mean for zf
    zf_estimate = round(trace.posterior['zf_estimated'].mean().item(), 3)

    return zf_estimate

