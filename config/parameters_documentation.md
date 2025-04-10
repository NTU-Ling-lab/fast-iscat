# Fast-iSCAT Parameter Documentation

This document describes the parameters used in the Fast-iSCAT package and their units.

## ISCAT Parameters

| Parameter | Unit | Description |
|-----------|------|-------------|
| M | dimensionless | Magnification of the microscope |
| pixel_physicalsize | micrometers (µm) | Physical size of camera pixel |
| nThetas | dimensionless | Number of angles for numerical integration |
| nx | dimensionless | Number of pixels in x direction |
| nz | dimensionless | Number of z-stack slices |
| zf_initial | micrometers (µm) | Initial guess for the system defocus |
| zf_range | micrometers (µm) | Range of defocus values to search |
| wavelength | nanometers (nm) | Illumination wavelength |
| adjust4Er | dimensionless | Adjustment factor for reference wave |
| na | dimensionless | Numerical aperture of the objective |
| coverslip_ri | dimensionless | Coverslip refractive index |
| coverslip_ri_spec | dimensionless | Coverslip refractive index specification |
| immersion_medium_ri | dimensionless | Immersion medium refractive index |
| immersion_medium_ri_spec | dimensionless | Immersion medium refractive index specification |
| working_distance | micrometers (µm) | Working distance of the objective |
| coverslip_thickness | micrometers (µm) | Thickness of the coverslip |
| coverslip_thickness_spec | micrometers (µm) | Specified thickness of the coverslip |
| frame_interval | seconds (s) | Time between consecutive frames |
| noise_sd | dimensionless | Standard deviation of noise between model prediction and observed value |


## Sample Parameters

| Parameter | Unit | Description |
|-----------|------|-------------|
| radius | nanometers (nm) | Radius of the particle |
| zp_initial | micrometers (µm) | Initial axial position of the particle |
| zp_range | micrometers (µm) | Range of axial positions to search |
| p_permittivity_real | dimensionless | Real part of particle permittivity |
| p_permittivity_img | dimensionless | Imaginary part of particle permittivity |
| adjust4Es | dimensionless | Adjustment factor for scattered wave |
| sample_medium_ri | dimensionless | Sample medium refractive index |

## Parameter Files

There are three parameter files in the config directory:

1. `parameters_static.json`: Used for static samples with minimal axial movement.
2. `parameters_dynamic.json`: Used for dynamic samples with thermal expansion-induced axial movement over time.
3. `parameters_diffusion.json`: Used for Brownian diffusion samples with stochastic movement in aqueous environments.
