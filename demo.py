# Fast Axial Localization for wide-field iSCAT
#
# This script demonstrates the workflow for analyzing interferometric scattering (iSCAT) 
# microscopy data following the approximate model introduced in "Fast 3D localization 
# of nano-objects in wide-field interferometric scattering microscopy via vectorial 
# diffraction model-derived analytical fitting" to perform axial localization of nano-objects. 
# The workflow includes:
#
# 1. Loading and initializing data from `.mat` files
# 2. Parameter configuration via JSON files
# 3. Processing for axial localization
# 4. Visualization of results
#
# We will examine different types of samples to showcase the performance of the method:
# - Static sample: Minimal motion, used to establish baseline precision
# - Dynamic sample: Thermal expansion-induced displacement
# - Diffusion sample: Free Brownian motion in aqueous environment

# ## 1. Import Required Libraries
#
# First, we need to import the necessary libraries for data processing, visualization, 
# and our custom axial localization package.

import numpy as np
import matplotlib.pyplot as plt
import os

# Import packages
import fast_iscat

# Configure plot styling for better visuals
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'figure.figsize': (10, 6),
    'axes.linewidth': 1.5
})

# ## 2. Processing Functions
#
# Define a function to process samples for axial localization

def process_sample(mat_filename, config_filename, plot_trace=True, sample_type="Static"):
    """   
    Process a sample for axial localization and plot the results.
    
    Parameters:
        mat_filename (str): Path to the .mat file
        config_filename (str): Path to the configuration JSON file
        plot_trace (bool): Whether to plot traces during processing
        sample_type (str): Type of sample (for plot title)
        
    Returns:
        numpy.ndarray: Axial location measurements
    """
    # Load data
    exp_ipsf = fast_iscat.utils.load_and_preprocess_data(mat_filename)
    
    # Load configuration parameters
    m = fast_iscat.parameters.iscat(config_filename)
    frame_interval = m.frame_interval
    
    # Process data to get axial location
    print(f"Processing {sample_type} sample...")
    axial_location = fast_iscat.core.estimate_axial_location(exp_ipsf, config_filename, plot_trace=plot_trace)
    
    # Plot the results
    fast_iscat.utils.plot_axial_location(
        axial_location, 
        frame_interval, 
        len(exp_ipsf),
        title=f"{sample_type} Sample: Axial Location vs Time"
    )
    
    return axial_location

# ## 3. Static Sample Analysis
#
# First, we'll analyze a static sample to demonstrate the basic workflow. 
# Static samples have little or no motion in the axial direction, providing 
# a good baseline for understanding the measurement precision.

def analyze_static_sample():
    # File paths for static sample
    static_mat_filename = "./data/static.mat"
    static_config_filename = "./config/parameters_static.json"
    
    # Check if files exist
    if os.path.exists(static_mat_filename) and os.path.exists(static_config_filename):
        # Process static sample
        static_axial_location = process_sample(static_mat_filename, static_config_filename, sample_type="Static")
        
        # Display statistical metrics
        print("\nStatistical Analysis for Static Sample:")
        print(f"Mean axial location: {np.mean(static_axial_location):.2f} nm")
        print(f"Standard deviation: {np.std(static_axial_location):.2f} nm")
        print(f"Range (max - min): {np.max(static_axial_location) - np.min(static_axial_location):.2f} nm")
        
        return static_axial_location
    else:
        print(f"Warning: Could not find static sample files. Please ensure '{static_mat_filename}' and '{static_config_filename}' exist in the current directory.")
        return None

# ## 4. Dynamic Sample Analysis (Thermal Expansion)
#
# Now, we'll analyze a dynamic sample where the nano-object displacement is caused by 
# thermal expansion of the surrounding material. This demonstrates the capability of our 
# method to track axial displacements with high precision in response to temperature changes.

def analyze_dynamic_sample():
    # File paths for thermal expansion-induced displacement sample
    dynamic_mat_filename = "./data/dynamic.mat"
    dynamic_config_filename = "./config/parameters_dynamic.json"
    
    # Check if files exist
    if os.path.exists(dynamic_mat_filename) and os.path.exists(dynamic_config_filename):
        # Process thermal expansion sample
        dynamic_axial_location = process_sample(dynamic_mat_filename, dynamic_config_filename, sample_type="Thermal Expansion")
        
        # Display statistical metrics
        print("\nStatistical Analysis for Thermal Expansion Sample:")
        print(f"Mean axial location: {np.mean(dynamic_axial_location):.2f} nm")
        print(f"Standard deviation: {np.std(dynamic_axial_location):.2f} nm")
        print(f"Range (max - min): {np.max(dynamic_axial_location) - np.min(dynamic_axial_location):.2f} nm")
        
        return dynamic_axial_location
    else:
        print(f"Warning: Could not find thermal expansion sample files. Please ensure '{dynamic_mat_filename}' and '{dynamic_config_filename}' exist in the current directory.")
        return None


# ## 5. Diffusion Sample Analysis (Brownian Motion)
#
# Finally, we'll analyze a sample where the nano-object undergoes Brownian motion as it
# diffuses freely in water. This showcases the ability of our method to track the
# stochastic 3D movement of nanoparticles in liquid environments. Note that to achieve a
# large dynamic range, this section relies on the approximate model provided by Eq. (S20) 
# in the supplementary material and additional regularization is used to ensure temporal
# consistency.

def analyze_diffusion_sample():
    # File paths for diffusing nanoparticle sample
    diffusion_mat_filename = "./data/diffusion.mat"
    diffusion_config_filename = "./config/parameters_diffusion.json"
    
    # Check if files exist
    if os.path.exists(diffusion_mat_filename) and os.path.exists(diffusion_config_filename):
        # Process Brownian motion sample
        diffusion_axial_location = process_sample(diffusion_mat_filename, diffusion_config_filename, sample_type="Brownian Diffusion")
        
        # Display statistical metrics
        print("\nStatistical Analysis for Brownian Diffusion Sample:")
        print(f"Mean axial location: {np.mean(diffusion_axial_location):.2f} nm")
        print(f"Standard deviation: {np.std(diffusion_axial_location):.2f} nm") 
        print(f"Range (max - min): {np.max(diffusion_axial_location) - np.min(diffusion_axial_location):.2f} nm")
        
        return diffusion_axial_location
    else:
        print(f"Warning: Could not find diffusion sample files. Please ensure '{diffusion_mat_filename}' and '{diffusion_config_filename}' exist in the current directory.")
        return None

# Main execution
if __name__ == "__main__":
    print("Running Fast Axial Localization for wide-field iSCAT analysis...")
    # Analyze static sample
    static_result = analyze_static_sample()
    
    # Analyze thermal expansion sample
    dynamic_result = analyze_dynamic_sample()

    # Analyze Brownian diffusion sample
    diffusion_result = analyze_diffusion_sample()
    
    print("Analysis complete!") 
