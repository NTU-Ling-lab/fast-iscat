#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Functions for iSCAT Axial Localization

This module provides the main processing functions for axial localization
in interferometric scattering (iSCAT) microscopy.

Author: weiliaoliao
License: MIT
"""

import numpy as np
from tqdm import tqdm
from scipy.optimize import least_squares

# Import custom modules
from .utils import lateral_location_calibration, plot_experiment_vs_prediction
from .parameters import iscat, sample
from .bayesian_estimation import estimate_zf_from_exp_ipsf
from .fitting import residuals_with_loss_record, residuals_scalar_model
from .forward_models import approximate_model, vectorial_ipsf, precompute_scalar_model, scalar_model


def obtain_key_parameters(center_location, config_file):
    """
    Obtain key parameters for axial localization processing.
    
    Parameters:
        center_location (list): [x, y] coordinates of the center position
        config_file (str): Path to configuration JSON file
        
    Returns:
        dict: Dictionary containing key parameters for processing
    """
    m = iscat(config_file)
    s = sample(config_file)
    
    
    # Always run vectorial_ipsf
    ipsf, ES1, _, E_r, amplitude_baseline, f_d, h_d, prefactor = vectorial_ipsf(
        center_location, m=m, s=s
    )

    result = {
        "ipsf": ipsf,
        "ES1": ES1,
        "f_d": f_d,
        "h_d": h_d,
        "ampli": amplitude_baseline,
        "E_r": E_r,
        "prefactor": prefactor,
        "wave_num": m.wave_num,
        "zf": m.zf_initial,
        "zp_min": s.zp_min,
        "zp_max": s.zp_max,
        "zp_initial": s.initial_guess,
        "M": m.M,
        "pixel_physicalsize": m.pixel_physicalsize,
        "noise_sd": m.noise_sd
    }

    if s.zp_range >= 3:
        wave_num, d_theta, bessel_weight, E_r,prefactor,coeff_zp, coeff_zf, const = precompute_scalar_model(
            center_location, m=m, s=s
        )
        result.update({
            "d_theta": d_theta,
            "bessel_weight": bessel_weight,
            "coeff_zp": coeff_zp,
            "coeff_zf": coeff_zf,
            "const": const
        })
    return result
    

def estimate_axial_location(exp_ipsf, config_file, plot_trace=False):
    """
    Process interferometric scattering (iSCAT) data for axial localization.
    
    Parameters:
        exp_ipsf (numpy.ndarray): 3D array containing normalized interference patterns
        config_file (str): Path to configuration JSON file
        plot_trace (bool): Whether to plot posterior trace
    
    Returns:
        numpy.ndarray: Normalized axial location measurements
    """
    # Initialize array to store axial locations
    axial_location = np.empty(len(exp_ipsf))
    
    # Initialize array to store predicted images
    nx = len(exp_ipsf[0])
    predicted_images = np.zeros((len(exp_ipsf), nx, nx))
    
    # Set up grid for processing
    x = np.linspace(0, nx-1, nx)
    y = np.linspace(0, nx-1, nx)
    X, Y = np.meshgrid(x, y)
    
    # Calculate lateral location from the first frame
    center_location = lateral_location_calibration(exp_ipsf[0])
    
    # Obtain parameters for processing
    key_parameters = obtain_key_parameters(center_location, config_file)
    
    # Initial estimation of defocus parameter
    print("Initial guess for defocus using Bayesian estimation...")
    zf_estimate = estimate_defocus(
        key_parameters, config_file, exp_ipsf
    )

    print(f"Estimated defocus from Bayesian estimation: {zf_estimate:.2f} μm")
    print("Starting axial localization...")
    zp_range = key_parameters['zp_max'] - key_parameters['zp_min']
    # Perform fitting for each frame
    if zp_range < 3:
        for j in tqdm(range(len(exp_ipsf)), desc="Fitting Progress", unit="frame"):
            nop = np.array(exp_ipsf[j]).ravel()
            result_list = fit_axial_positions(
                zf_estimate, center_location, key_parameters, 
                exp_ipsf, nop
            )
            axial_location[j] = result_list["zp_fit"]
            predicted_images[j] = result_list["predicted_image"]
    else:
        for j in tqdm(range(len(exp_ipsf)), desc="Fitting Progress", unit="frame"):
            nop = np.array(exp_ipsf[j]).ravel()
            result_list = fit_axial_locations_scalar_model(zf_estimate, j, axial_location, key_parameters, exp_ipsf)
            axial_location[j] = result_list["zp_fit"]
            predicted_images[j] = result_list["predicted_image"]
        
    # If we want to plot, generate time-matched predictions and show comparison
    if plot_trace:
        # Extract necessary parameters for visualization
        nx_half = nx // 2
        pixel_physicalsize = key_parameters["pixel_physicalsize"]
        M = key_parameters["M"]
                      
        plot_experiment_vs_prediction(
            exp_ipsf, predicted_images, nx, nx_half,
            pixel_physicalsize, M
        )
    
    return axial_location * 1000  # convert to nm


def estimate_defocus(key_parameters, config_file, cropped_video_data):
    """
    Estimate the defocus parameter (zf) from the given data.
    
    Parameters:
        key_parameters (dict): Key parameters from obtain_parameter
        config_file (str): Path to configuration JSON file
        cropped_video_data (numpy.ndarray): Video data
    
    Returns:
        float: Estimated zf value
    """
    # Extract necessary parameters from key_parameters
    I = key_parameters["ipsf"]
    noise_sd = key_parameters["noise_sd"]
    wave_num = key_parameters["wave_num"]
    h_d = key_parameters["h_d"]
    f_d = key_parameters["f_d"]
    amplitude_baseline = key_parameters["ampli"]
    prefactor = key_parameters['prefactor']
    ampli = amplitude_baseline * prefactor
    ampli_real = np.real(ampli).astype(np.float64)
    ampli_imag = np.imag(ampli).astype(np.float64)
    E_r = key_parameters["E_r"].real
    
    # Calculate coefficients
    f = np.mean(f_d)   # Coefficient of zp
    h = np.mean(h_d)   # Coefficient of zf
    b = wave_num * f
    c = wave_num * h

    # Call Bayesian estimation function
    zf_estimate = estimate_zf_from_exp_ipsf(
        config_file, cropped_video_data,
        wave_num, h_d, f_d, b, c, ampli_real, ampli_imag, E_r, noise_sd
    )

    return zf_estimate


def fit_axial_positions(zf_estimated, center_location, key_parameters, exp_ipsf, nop2):
    """
    Fit axial positions (zp) for a given interferometric pattern.
    
    Parameters:
        zf_estimated (float): Estimated defocus parameter
        center_location (list): [x, y] coordinates of the center position
        key_parameters (dict): Key parameters from obtain_parameter
        exp_ipsf (numpy.ndarray): Normalized interferometric patterns
        nop2 (numpy.ndarray): Experimental data (flattened)
    
    Returns:
        dict: A dictionary containing the fitting results and predicted image
    """
    # Extract necessary parameters from key_parameters
    nx = len(exp_ipsf[0])
    nx_half = nx // 2
    wave_num = key_parameters["wave_num"]
    h_d = key_parameters["h_d"]
    f_d = key_parameters["f_d"]
    amplitude_baseline = key_parameters["ampli"]
    prefactor = key_parameters['prefactor']
    ampli = amplitude_baseline * prefactor
    E_r = key_parameters["E_r"].real
    
    # Calculate coefficients
    f = np.mean(f_d)   # Coefficient of zp
    h = np.mean(h_d)   # Coefficient of zf
    b = wave_num * f
    c = wave_num * h * zf_estimated
   
    # Define initial guess and bounds
    initial_guess = ([key_parameters["zp_initial"]])
    param_bounds = ([key_parameters["zp_min"]], [key_parameters["zp_max"]])

    # Initialize history trackers
    loss_history = []
    params_history = []
    residual_history = []

    # Define a closure for residuals function
    def residuals_closure(params):
        return residuals_with_loss_record(
            params, center_location, ampli, nop2, nx, nx_half, 
            wave_num, f_d, h_d, zf_estimated, b, c, E_r,
            loss_history, params_history, residual_history
        )

    # Call least_squares for fitting
    result = least_squares(
        residuals_closure, initial_guess, bounds=param_bounds,
        max_nfev=1000, verbose=0
    )

    # Calculate the predicted image for this frame
    nop_predicted = approximate_model(
        ampli, result.x[0] * 1e-6, wave_num, f_d, h_d, 
        zf_estimated * 1e-6, b, c, E_r
    )
    predicted_image = nop_predicted.reshape((nx, nx))
    
    # Return fitting results and history
    return {
        "zp_fit": result.x[0],
        "predicted_image": predicted_image
    } 


def fit_axial_locations_scalar_model(zf_estimated, frame_index, axial_location, key_parameters, exp_ipsf):
    """
    Fit axial positions using the scalar model approach.
    
    Parameters:
        zf_estimated (float): Estimated defocus parameter
        frame_index (int): Current frame index
        axial_location (numpy.ndarray): Array of axial locations
        key_parameters (dict): Key parameters from obtain_parameter
        exp_ipsf (numpy.ndarray): Normalized interferometric patterns
    
    Returns:
        dict: A dictionary containing the fitting results and predicted image
    """
    nx = len(exp_ipsf[0])
    initial_guess = key_parameters["zp_initial"]         
    zp_range = key_parameters["zp_max"] - key_parameters["zp_min"]
    d_theta = key_parameters['d_theta']
    bessel_weight = key_parameters['bessel_weight']
    E_r = key_parameters['E_r']
    wave_num = key_parameters['wave_num']
    prefactor = key_parameters['prefactor']
    coeff_zp = key_parameters['coeff_zp']
    coeff_zf = key_parameters['coeff_zf']
    const = key_parameters['const']
    
    if frame_index > 0:
        initial_guess = axial_location[frame_index - 1] 
    
    lower_bound = max(initial_guess - zp_range/10, 0)
    upper_bound = initial_guess + zp_range/10
    param_bounds = ([lower_bound], [upper_bound])
    
    def residuals_closure(params):
        return residuals_scalar_model(
            params, frame_index, exp_ipsf, axial_location, 
            d_theta, bessel_weight, wave_num, E_r, prefactor, coeff_zp, 
            coeff_zf, const, zf_estimated, lambda_smooth=1e-3
        )

    result = least_squares(
        residuals_closure, initial_guess, bounds=param_bounds,
        max_nfev=1000, verbose=0
    )
    
    # Calculate the predicted image for this frame
    nop_predicted = scalar_model(
        d_theta, wave_num, bessel_weight, E_r, prefactor, coeff_zp, coeff_zf, 
        const, zf_estimated, result.x[0] 
    )
    
    predicted_image = nop_predicted.reshape((nx, nx))
    
    return {
        "zp_fit": result.x[0],
        "predicted_image": predicted_image
    } 


