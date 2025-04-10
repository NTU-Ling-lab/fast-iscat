#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter classes for iSCAT Axial Localization

This module provides data classes for storing and accessing configuration parameters
for interferometric scattering (iSCAT) microscopy analysis.

Classes:
    iscat: Configuration for the iSCAT microscope
    sample: Configuration for the sample being imaged

Authors: weiliaoliao, Tong Ling
License: MIT
"""

import numpy as np
import json
from dataclasses import dataclass


@dataclass
class iscat:
    """
    Class to store iSCAT microscope parameters.
    
    Attributes are loaded from a JSON configuration file.
    """
    
    def __init__(self, config_file):
        """
        Initialize with parameters from a configuration file.
        
        Parameters:
            config_file (str): Path to JSON configuration file
        """
        with open(config_file) as f:
            config = json.load(f)
            iscat_params = config['iscat']
            self.M = iscat_params['M']
            self.pixel_physicalsize = iscat_params['pixel_physicalsize']
            self.nThetas = iscat_params['nThetas']
            self.wavelength = iscat_params['wavelength']
            self.zf_initial = iscat_params['zf_initial']
            self.adjust4Er = iscat_params['adjust4Er']
            self.nx = iscat_params['nx']
            self.ny = self.nx               # for simplicity, assume nx = ny
            self.na = iscat_params['na']
            self.coverslip_ri = iscat_params['coverslip_ri']
            self.coverslip_ri_spec = iscat_params['coverslip_ri_spec']
            self.immersion_medium_ri = iscat_params['immersion_medium_ri']
            self.immersion_medium_ri_spec = iscat_params['immersion_medium_ri_spec']
            self.working_distance = iscat_params['working_distance']
            self.coverslip_thickness = iscat_params['coverslip_thickness']
            self.coverslip_thickness_spec = iscat_params['coverslip_thickness_spec']
            self.frame_interval = iscat_params['frame_interval']
            self.noise_sd = iscat_params['noise_sd']
            self.nz = iscat_params['nz']
            self.zf_range = iscat_params['zf_range']

    @property
    def dxy(self):
        """Calculate pixel size in sample space."""
        return (self.pixel_physicalsize * 1e-6) / self.M

    @property
    def wave_num(self):
        """Calculate wave number k = 2π/λ in 1/m."""
        return 2 * np.pi / (self.wavelength * 1e-9)

    @property
    def NA(self):
        """Numerical aperture of the objective."""
        return self.na

    @property
    def ng(self):
        """Refractive index of the coverslip."""
        return self.coverslip_ri

    @property
    def ng0(self):
        """Design refractive index of the coverslip."""
        return self.coverslip_ri_spec

    @property
    def ni(self):
        """Refractive index of the immersion medium."""
        return self.immersion_medium_ri

    @property
    def ni0(self):
        """Design refractive index of the immersion medium."""
        return self.immersion_medium_ri_spec

    @property
    def ti0(self):
        """Working distance in meters."""
        return self.working_distance * 1e-6

    @property
    def tg(self):
        """Coverslip thickness in meters."""
        return self.coverslip_thickness * 1e-6

    @property
    def tg0(self):
        """Design coverslip thickness in meters."""
        return self.coverslip_thickness_spec * 1e-6

    @property
    def half_angle(self):
        """Half-angle of the objective cone."""
        return np.arcsin(self.na / self.ni)

   

@dataclass
class sample:
    """
    Class to store sample parameters.
    
    Attributes are loaded from a JSON configuration file.
    """
    
    def __init__(self, config_file):
        """
        Initialize with parameters from a configuration file.
        
        Parameters:
            config_file (str): Path to JSON configuration file
        """
        self.config_file = config_file  # Store the config file path
        with open(config_file) as f:
            config = json.load(f)
            sample_params = config['sample']
            self.radius = sample_params['radius']
            self.zp_initial = sample_params['zp_initial']
            self.zp_range = sample_params['zp_range']
            self.p_permittivity_real = sample_params['p_permittivity_real']
            self.p_permittivity_img = sample_params['p_permittivity_img']
            self.adjust4Es = sample_params['adjust4Es']
            self.sample_medium_ri = sample_params['sample_medium_ri']

    @property
    def r(self):
        """Radius of the particle in meters."""
        return self.radius * 1e-9

    @property
    def ns(self):
        """Refractive index of the sample medium."""
        return self.sample_medium_ri

    @property
    def p_permittivity(self):
        """Complex permittivity of the particle."""
        return complex(self.p_permittivity_real, self.p_permittivity_img)

    @property
    def s_permittivity(self):
        """Permittivity of the sample medium."""
        return self.ns ** 2
        
    @property
    def zp(self):
        """Axial position range of the particle in meters."""
        if self.nz == 1:
            return np.full(1, self.zp_initial * 1e-6)
        else:
            return np.linspace(self.zp_initial - self.zp_range/2, self.zp_initial + self.zp_range/2, self.nz) * 1e-6
            
    @property
    def nz(self):
        """Number of axial positions."""
        with open(self.config_file) as f:
            config = json.load(f)
            return config['iscat']['nz']
            
    @property
    def zp_min(self):
        """Minimum axial position in μm."""
        return self.zp_initial - self.zp_range/2
        
    @property
    def zp_max(self):
        """Maximum axial position in μm."""
        return self.zp_initial + self.zp_range/2
    
    @property
    def initial_guess(self):
        """Initial guess for the axial position in μm."""
        return self.zp_initial



