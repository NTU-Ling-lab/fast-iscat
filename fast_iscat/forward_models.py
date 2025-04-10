#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forward Models for iSCAT Axial Localization

This module implements the forward physical models for interferometric scattering
(iSCAT) microscopy. It provides functions to calculate the expected observed
patterns based on physical parameters of the setup and sample.

Authors: weiliaoliao, Haitao Nie, Tong Ling
License: MIT
"""

import numpy as np
from scipy.special import j0, jv


def get_Es0Er0(m, s):
    """
    Calculate scattered (Es0) and reference (Er0) electric field amplitudes.
    
    Parameters:
        m (iscat): Microscope parameters
        s (sample): Sample parameters
        
    Returns:
        tuple: (E_s0, E_r0) - scattered and reference field amplitudes
    """
    # Calculate transmission angles using Snell's Law
    thetag = np.arcsin(m.ni / m.ng * 0)
    thetas = np.arcsin(m.ng / s.ns * np.sin(thetag))
    
    # Calculate reflectance and transmittance at the coverslip-sample interface
    R_gs = ((m.ng * np.cos(thetas) - s.ns * np.cos(thetag)) / 
            (m.ng * np.cos(thetas) + s.ns * np.cos(thetag))) ** 2
    T_gs = 1 - R_gs
    
    # Calculate reflectance and transmittance at the coverslip-immersion oil interface
    R_gi = ((m.ng * np.cos(0) - m.ni * np.cos(thetag)) / 
            (m.ng * np.cos(0) + m.ni * np.cos(thetag))) ** 2
    T_gi = 1 - R_gi
    
    # Calculate the amplitude of the scattered electric field
    wave_num_s = s.ns * m.wave_num                      # wave_num in sample
    alpha = (4 * np.pi * s.r ** 3 * 
            (s.p_permittivity - s.s_permittivity) / 
            (s.p_permittivity + 2 * s.s_permittivity))
    C = wave_num_s ** 4 / (6 * np.pi) * abs(alpha) ** 2
    mu = 1 / np.pi * np.arcsin(m.NA / m.ni)
    E_s0 = (s.adjust4Es * mu * 
           np.sqrt(C) * np.sqrt(T_gs) * np.exp(1j * np.angle(alpha)))
    
    # Calculate the amplitude of the reference electric field
    E_r0 = m.adjust4Er * np.sqrt(R_gs) * np.sqrt(T_gi)
    
    return E_s0, E_r0


def vectorial_ipsf(center_idx, m, s):
    """
    Generate iPSF(s) using Richards-Wolf vectorial diffraction model.
    
    Parameters:
        center_idx (list): Center coordinates [y, x]
        m (iscat): Microscope parameters
        s (sample): Sample parameters
        
    Returns:
        tuple: Multiple outputs including iPSF intensity and phase information
    """
    zf = m.zf_initial

    # ti has to be calculated according to Gibson-Lanni equation
    ti = s.zp - zf - m.ni * (s.zp / s.ns + m.tg / m.ng -
                            m.tg0 / m.ng0 - m.ti0 / m.ni0)

    # Calculate the amplitudes of the scattered and reference waves
    E_s0, E_r = get_Es0Er0(m, s)
    prefactor = -1j * m.wave_num / 2 * E_s0
    
    max_angle = np.arcsin(m.NA / m.ni)
    d_theta = max_angle / (m.nThetas - 1)
    thetas = np.linspace(0, max_angle, m.nThetas)
    sintheta = np.sin(thetas)
    costheta = np.cos(thetas)
    ni2sin2theta = (m.ni ** 2 * sintheta ** 2).astype("complex")
    nsroot = np.sqrt(s.ns ** 2 - ni2sin2theta)
    ngroot = np.sqrt(m.ng ** 2 - ni2sin2theta)
    ts1ts2 = (4.0 * nsroot * ngroot).astype("complex")
    tp1tp2 = ts1ts2.copy()
    tp1tp2 /= ((m.ng * costheta + m.ni / m.ng * ngroot) * 
              (s.ns / m.ng * ngroot + m.ng / s.ns * nsroot))
    ts1ts2 /= (m.ni * costheta + ngroot) * (ngroot + nsroot)
    sqrtcostheta = np.sqrt(costheta).astype("complex")
    sincos = np.sin(thetas) * sqrtcostheta
    A0 = ts1ts2 + tp1tp2 / s.ns * nsroot
    A2 = ts1ts2 - tp1tp2 / s.ns * nsroot

    # Camera parameters
    x = np.arange(-(m.nx - 1) / 2, (m.nx + 1) / 2, 1)
    y = np.arange(-(m.ny - 1) / 2, (m.ny + 1) / 2, 1)
        
    xi_map, yi_map = np.meshgrid(x, y)
    r_d = np.sqrt((xi_map + ((m.nx-1)/2 - center_idx[0]))**2 + 
                 (yi_map + ((m.nx-1)/2 - center_idx[1]))**2) * m.dxy
    phi_d = np.angle((xi_map + ((m.nx-1)/2 - center_idx[0])) + 
                    1j * (yi_map + ((m.nx-1)/2 - center_idx[1])))
    
    # Calculating I0 and I2
    I0 = np.empty(shape=(s.nz, m.nx, m.ny), dtype="complex")
    I2 = np.empty(shape=(s.nz, m.nx, m.ny), dtype="complex")
    I_2 = np.empty(shape=(s.nz, m.nx, m.ny), dtype="complex")
    bessel0 = np.repeat(r_d[np.newaxis, :, :], m.nThetas,
                       axis=0).astype("complex")
    bessel2 = np.repeat(r_d[np.newaxis, :, :], m.nThetas,
                       axis=0).astype("complex")
    
    b0 = np.repeat(r_d[np.newaxis, :, :], m.nThetas, axis=0).astype("complex")
    b2 = np.repeat(r_d[np.newaxis, :, :], m.nThetas, axis=0).astype("complex")
    f_d = np.empty([m.nThetas])
    h_d = np.empty([m.nThetas])
    I = np.empty(shape=(s.nz, m.nx, m.ny))
    E_s1 = np.empty(shape=(s.nz, m.nx, m.ny), dtype="complex")
    E_s2 = np.empty(shape=(s.nz, m.nx, m.ny), dtype="complex")
    phase_s1 = np.empty(shape=(s.nz, m.nx, m.ny))

    for t in range(m.nThetas):
        bessel_arg = m.wave_num * m.ni * np.sin(thetas[t]) * r_d
        bessel0[t] = j0(bessel_arg) * sincos[t] * A0[t]
        bessel2[t] = jv(2, bessel_arg) * sincos[t] * A2[t]
        f_d[t] = (s.ns * (np.sqrt(1-(m.ni*sintheta[t])**2/s.ns**2) + 1) + 
                 m.ni*(1-m.ni/s.ns)*(costheta[t]-1))
        h_d[t] = -m.ni*(costheta[t]-1)
    
    for i in range(s.nz):
        OPD = (ti[i] * m.ni * costheta + s.zp[i] * nsroot + m.tg * ngroot - 
              m.tg0 * np.sqrt(m.ng0 ** 2 - ni2sin2theta) - 
              m.ti0 * np.sqrt(m.ni0 ** 2 - ni2sin2theta) +
              (s.ns * s.zp[i] + m.ng * m.tg + m.ni * ti[i]) - 
              (m.ng0 * m.tg0 + m.ni0 * m.ti0) - 
              2*(m.ni*ti[i]-m.ni0*m.ti0) - 2*(m.ng*m.tg-m.ng0*m.tg0))
        
        # OPD correction term
        aberration_factor = np.exp(1j * m.wave_num * OPD)
        
        for t in range(len(thetas)):
            b0[t] = bessel0[t] * aberration_factor[t]
            b2[t] = bessel2[t] * aberration_factor[t]
            
        I0[i] = b0.sum(0) * d_theta
        I2[i] = b2.sum(0) * d_theta

        E_s1 = prefactor * (I0[i] + I2[i] * np.cos(2 * phi_d))
        phase_s1[i] = np.angle(E_s1)
        E_s2 = prefactor * I2[i] * np.sin(2 * phi_d)
        I[i] = abs(E_r + E_s1) ** 2 + abs(E_s2) ** 2
        I_2[i] = E_s1
                
    return I, I_2, phase_s1, E_r, bessel0.sum(0) * d_theta, f_d, h_d, prefactor


def approximate_model(ampli, zp, wave_num, f_d, h_d, zf, b, c, E_r):
    """
    Calculate approximated iSCAT image using a simplified analytical model.
    
    This model is derived from the vectorial model as shown in Eq. (S33) in the 
    supplementary material of the paper "Fast 3D localization of nano-objects in 
    wide-field interferometric scattering microscopy via vectorial diffraction 
    model-derived analytical fitting".
    
    Parameters:
        ampli (complex): Amplitude coefficient.
        zp (float): Axial position of the particle in meters.
        wave_num (float): Wave number.
        f_d (ndarray): Coefficient for zp.
        h_d (ndarray): Coefficient for zf.
        zf (float): Defocus parameter in meters.
        b (float): Precomputed wave_num * f_d.
        c (float): Precomputed wave_num * h_d * zf.
        E_r (float): Reference field amplitude.
        
    Returns:
        ndarray: Flattened intensity distribution.
    """
    deltax = wave_num * (np.std(f_d) * zp + np.std(h_d) * zf)
    sca = ampli * np.exp(1j*(b*zp + c * zf)) * (-0.5*(deltax)**2 + 1j*(deltax) + 1)
    img = abs(E_r + sca) ** 2

    return img.flatten()

def precompute_scalar_model(center_idx, m, s):
    x = np.arange(-(m.nx - 1) / 2, (m.nx + 1) / 2, 1)
    y = np.arange(-(m.ny - 1) / 2, (m.ny + 1) / 2, 1)
        
    xi_map, yi_map = np.meshgrid(x, y)   
    r_d = np.sqrt((xi_map + ((m.nx-1)/2 - center_idx[0]))**2 + 
                 (yi_map + ((m.nx-1)/2 - center_idx[1]))**2) * m.dxy
    thetas = np.linspace(0, np.arcsin(m.NA / m.ni), m.nThetas)
    sintheta = np.sin(thetas)
    costheta = np.cos(thetas)
    d_theta = thetas[1] - thetas[0]
    ni2sin2theta = (m.ni * sintheta)**2
    nsroot = np.sqrt(s.ns**2 - ni2sin2theta)
    ngroot = np.sqrt(m.ng**2 - ni2sin2theta)
    ng0root = np.sqrt(m.ng0**2 - ni2sin2theta)
    ni0root = np.sqrt(m.ni0**2 - ni2sin2theta)
    a_z = 1 - m.ni / s.ns
    a_c = -m.ni * (m.tg / m.ng - m.tg0 / m.ng0 - m.ti0 / m.ni0)
    # ---- Calculate coefficients for zp and zf----
    coeff_zp = (
        m.ni * a_z * (costheta - 1)
        + nsroot + s.ns
        - m.ni * a_z
    )
    coeff_zf = m.ni * (1 - costheta)
    const = (
        m.ni * a_c * costheta
        + m.tg * ngroot
        - m.tg0 * ng0root
        - m.ti0 * ni0root
        + m.ng * m.tg
        + m.ni * a_c
        - m.ng0 * m.tg0
        - m.ni0 * m.ti0
        - 2 * (m.ni * a_c - m.ni0 * m.ti0)
        - 2 * (m.ng * m.tg - m.ng0 * m.tg0)
    )

    # Angular weighting
    A0 = np.ones_like(thetas)
    sincos = sintheta * np.sqrt(costheta)
    A_stack = (sincos * A0)[:, np.newaxis, np.newaxis]
    r_d_stack = r_d[np.newaxis, :, :] 
    sintheta_stack = sintheta[:, np.newaxis, np.newaxis]
    bessel_args = m.wave_num * m.ni * sintheta_stack * r_d_stack
    bessel0 = j0(bessel_args)
    bessel_weight = bessel0 * A_stack  
    E_s0, E_r = get_Es0Er0(m, s)            
    prefactor = -1j * m.wave_num / 2 * E_s0  

    return m.wave_num, d_theta, bessel_weight, E_r, prefactor, coeff_zp, coeff_zf, const


def scalar_model(d_theta, wave_num, bessel_weight, E_r, prefactor, coeff_zp, coeff_zf, const, zf, zp):
    """
    The scalar model is an approximation of Richards-Wolf vectorial diffraction model
    until Eq. (S20) in the supplementary material of the paper.
    """
    
    coeff_zp = coeff_zp[:, np.newaxis, np.newaxis]
    coeff_zf = coeff_zf[:, np.newaxis, np.newaxis]
    const = const[:, np.newaxis, np.newaxis]
    OPD = zp * 1e-6 * coeff_zp + zf * 1e-6 * coeff_zf + const      
    phase = wave_num * OPD                            
    exp_phase = np.exp(1j * phase)                      
    weighted = bessel_weight * exp_phase                
    I0 = np.sum(weighted, axis=0) * d_theta             
    E_total = E_r + prefactor * I0
    I_i = np.abs(E_total)**2
    return I_i.astype("float64")
