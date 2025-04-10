#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fitting Functions for iSCAT Axial Localization

This module provides functions for fitting models to experimental iSCAT data.
It includes functions for:
- Creating radial weight matrices for fitting
- Computing residuals between model and experiment
- Recording fit history during optimization

Author: weiliaoliao
License: MIT
"""

from .forward_models import approximate_model
from .forward_models import scalar_model
import numpy as np


def create_radial_weights(nx, max_idx, max_weight=1.0, min_weight=0.0):
    """
    Create a radial weight matrix for an nx x nx grid.

    Parameters:
        nx (int): Grid size.
        max_idx (tuple): Center index of the radial weights.
        max_weight (float): Weight at the center.
        min_weight (float): Weight at the edges.

    Returns:
        np.ndarray: Radial weight matrix.
    """
    x, y = np.meshgrid(np.arange(nx), np.arange(nx))
    radius = np.sqrt((x - max_idx[0])**2 + (y - max_idx[1])**2)
    max_radius = np.max(radius)
    weights = max_weight - (max_weight - min_weight) * (radius / max_radius)
    return weights

def residuals(params, center_location, ampli, nop2, nx, nx_half, wave_num, f_d, h_d, zf, b, c, E_r):
    """
    Compute residuals between the simulated curve and experimental data.

    Parameters:
        params (list): Current parameter values.
        center_location (tuple): Center location of the pattern.
        ampli (complex): Amplitude.
        nop2 (np.ndarray): Experimental data.
        nx (int): Grid size.
        nx_half (int): Half grid size.
        wave_num (float): Wave number.
        f_d (float): Coefficient for zp.
        h_d (float): Coefficient for zf.
        zf (float): Defocus parameter.
        b (float): Precomputed wave_num * f_d.
        c (float): Precomputed wave_num * h_d * zf.
        E_r (float): Reference field.

    Returns:
        np.ndarray: Flattened residuals.
    """
    zp = params[0]
    nop2 = nop2.reshape([nx, nx])
    exp_mid_value = nop2[nx_half, nx_half]
    sim_curve = approximate_model(ampli, zp * 1e-6, wave_num, f_d, h_d, zf * 1e-6, b, c, E_r)
    sim_curve = sim_curve.reshape([nx, nx])
    sim_mid_value = sim_curve[nx_half, nx_half]
    normalization_factor = exp_mid_value / sim_mid_value
    sim_curve_normalized = sim_curve * normalization_factor    
    weights = create_radial_weights(nx, center_location)
    weighted_residuals = weights * (sim_curve_normalized - nop2)
    return weighted_residuals.flatten()

def residuals_with_loss_record(params, center_location, ampli, nop2, nx, nx_half, wave_num, f_d, h_d, zf, b, c, E_r, loss_history, params_history, residual_history):
    """
    Compute residuals with loss tracking.

    Same as `residuals` but also records loss history, parameter history,
    and residual history during optimization.

    Parameters:
        params (list): Current parameter values.
        center_location (tuple): Center location of the pattern.
        ampli (complex): Amplitude.
        nop2 (np.ndarray): Experimental data.
        nx (int): Grid size.
        nx_half (int): Half grid size.
        wave_num (float): Wave number.
        f_d (float): Coefficient for zp.
        h_d (float): Coefficient for zf.
        zf (float): Defocus parameter.
        b (float): Precomputed wave_num * f_d.
        c (float): Precomputed wave_num * h_d * zf.
        E_r (float): Reference field.
        loss_history (list): List to store loss values.
        params_history (list): List to store parameter values.
        residual_history (list): List to store residuals.

    Returns:
        np.ndarray: Flattened residuals.
    """
    residual = residuals(params, center_location, ampli, nop2, nx, nx_half, wave_num, f_d, h_d, zf, b, c, E_r)
    loss = np.sum(residual**2) 
    loss_history.append(loss)  
    params_history.append(params.copy())  
    residual_history.append(residual.copy())
    return residual

def residuals_scalar_model(zp, i, noisy_img, axial_location, d_theta, bessel_weight, wave_num, E_r, prefactor, coeff_zp, coeff_zf, const, zf, lambda_smooth=0.0e-2):
    """
    Compute residuals between scalar model predictions and experimental data.
    
    This function compares experimental iSCAT images with predictions from the scalar model,
    with an optional smoothness regularization term for temporal consistency.
    
    Parameters:
        zp (float): Axial position of the particle in micrometers.
        i (int): Current frame index.
        noisy_img (numpy.ndarray): Experimental iSCAT images.
        axial_location (numpy.ndarray): Previously estimated axial locations.
        d_theta (float): Angular integration step size.
        bessel_weight (numpy.ndarray): Precomputed Bessel function weights.
        wave_num (float): Wave number.
        E_r (float): Reference field amplitude.
        prefactor (complex): Prefactor for calculating the scattered field.
        coeff_zp (numpy.ndarray): Coefficients for zp in the phase term.
        coeff_zf (numpy.ndarray): Coefficients for zf in the phase term.
        const (numpy.ndarray): Constant terms in the phase term.
        zf (float): Defocus parameter in micrometers.
        lambda_smooth (float, optional): Smoothness regularization weight. Defaults to 0.0e-2.
        
    Returns:
        numpy.ndarray: Residuals between model and experiment, with optional smoothness penalty.
    """
    I_meas = noisy_img[i]
    I_sim = scalar_model(d_theta, wave_num, bessel_weight, E_r, prefactor, coeff_zp, coeff_zf, const, zf, zp)  
    # Normalization
    normalization_factor = np.mean(I_sim) / np.mean(I_meas) 
    I_meas = I_meas * normalization_factor
    residual = (I_sim - I_meas).flatten()
    if i > 0:
        smooth_penalty = np.sqrt(lambda_smooth) * (zp - axial_location[i-1])
        residual = np.append(residual, smooth_penalty)
    return residual
