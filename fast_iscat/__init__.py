#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Axial Localization Package
==========================

This package provides functions and classes for axial localization in 
interferometric scattering (iSCAT) microscopy.

Modules:
--------
- parameters: Configuration and parameter-related classes
- forward_models: Physical modeling of iSCAT process
- bayesian_estimation: Bayesian methods for parameter estimation
- fitting: Optimization and fitting routines
- utils: General utility functions
- processing: Data processing utilities
- visualization: Plotting and visualization tools
- core: Main processing workflow

License:
--------
MIT License. See LICENSE.txt for the complete license text.
"""

# Parameter and configuration classes
from .parameters import iscat, sample

# Core physical models
from .forward_models import get_Es0Er0, vectorial_ipsf, approximate_model

# Estimation and fitting
from .bayesian_estimation import estimate_zf_from_exp_ipsf
from .fitting import residuals, residuals_with_loss_record, create_radial_weights

# Data processing and utilities
from .utils import (
    lateral_location_calibration,
    plot_experiment_vs_prediction
)

# Main processing functions
from .core import (
    obtain_key_parameters,
    estimate_axial_location,
    estimate_defocus,
    fit_axial_positions
)

# Define the public API
__all__ = [
    # Parameter-related classes and functions
    "iscat",
    "sample",
    
    # Core physics models
    "get_Es0Er0",
    "vectorial_ipsf",
    "approximate_model",
    
    # Estimation and fitting
    "estimate_zf_from_exp_ipsf",
    "residuals",
    "residuals_with_loss_record",
    "create_radial_weights",
    
    # Utilities and processing
    "lateral_location_calibration",
    
    # Visualization
    "plot_experiment_vs_prediction",
    
    # Main processing workflow
    "obtain_key_parameters",
    "estimate_axial_location",
    "estimate_defocus",
    "fit_axial_positions"
]
