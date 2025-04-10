#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility Functions for iSCAT Axial Localization

This module provides utility functions for the processing, visualization,
and analysis of interferometric scattering (iSCAT) microscopy data.

Author: weiliaoliao
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import imgrvt as rvt
from scipy.io import loadmat
from matplotlib.colors import Normalize

def lateral_location_calibration(image, rmin=0):
    """
    Calculate the center position of the bright spot in the given image.

    Parameters:
        image (numpy.ndarray): Input 2D image.
        rmin (int): Minimum range for normalization, default is 0.

    Returns:
        max_idx (list): Center coordinates of the bright spot [mean_x, mean_y].
    """
    nx = len(image[0])
    transformed = rvt.rvt(image, rmin=rmin, rmax=round(nx/4))
    flat_transformed = transformed.flatten()
    top_4_idx = np.argsort(flat_transformed)[-4:]
    top_4_coords = np.array([np.unravel_index(idx, transformed.shape) for idx in top_4_idx])
    mean_x = np.mean(top_4_coords[:, 0])
    mean_y = np.mean(top_4_coords[:, 1])
    max_idx = [mean_y, mean_x]
    
    return max_idx

def plot_axial_location(axial_location, frame_interval, frames, title="Axial Location vs Time"):
    """
    Plot the axial location data with a clean, publication-ready style.
    
    Parameters:
        axial_location (numpy.ndarray): Axial location data
        frame_interval (float): Time interval between frames in milliseconds
        frames (int): Total number of frames
        title (str): Title of the plot
    """
    # Create timeline based on frame interval
    t = np.linspace(0, frames * frame_interval, frames)
    
    # Create plot with clean styling
    plt.figure(figsize=(10, 6))
    plt.plot(t, axial_location - axial_location[0], linewidth=2)
    
    # Set labels and title
    plt.xlabel("Time (ms)", fontsize=16)
    plt.ylabel("Axial location (nm)", fontsize=16)
    plt.title(title, fontsize=18)
    
    # Set tick parameters
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Remove top and right spines for cleaner look
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def load_and_preprocess_data(mat_filename):
    """
    Load data from a .mat file and preprocess it for axial localization.
    
    Parameters:
        mat_filename (str): Path to the .mat file
        
    Returns:
        numpy.ndarray: Preprocessed exp_ipsf data
    """
    # Load data from the .mat file
    file = loadmat(mat_filename)
    for key, value in file.items():
        if isinstance(value, np.ndarray) and not key.startswith('__'):
            exp_ipsf = value
            break
    
    # Transpose exp_ipsf to get correct shape (frames, height, width)
    if exp_ipsf.ndim == 3:
        exp_ipsf = np.transpose(exp_ipsf, (2, 0, 1))
    
    print(f"Loaded data with shape: {exp_ipsf.shape}")
    return exp_ipsf

def plot_experiment_vs_prediction(exp_ipsf, predicted_images, nx, nx_half, pixel_physicalsize, M, k=0):
    """
    Plot experimental and predicted images side by side with a slider for frame selection.
    
    Parameters:
        exp_ipsf (numpy.ndarray): 3D array of experimental images
        predicted_images (numpy.ndarray): 3D array of predicted images
        nx (int): Image width/height in pixels
        nx_half (int): Half of image width/height
        pixel_physicalsize (float): Physical size of pixel in μm
        M (float): Magnification factor
        k (int, optional): Initial frame index to display. Defaults to 0.
        
    Returns:
        None
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from matplotlib.colors import Normalize
    
    frame_number = len(exp_ipsf)
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    
    # Create a 2x2 grid for subplots
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    
    # Add subplots
    ax_exp = fig.add_subplot(gs[0, 0])     # Top left for experimental image
    ax_pred = fig.add_subplot(gs[0, 1])    # Top right for predicted image
    ax_line = fig.add_subplot(gs[1, :])    # Bottom for line plot
    ax_slider = plt.axes([0.1, 0.03, 0.8, 0.03])  # Slider at the bottom
    
    # Define function to calculate reference intensity (using boundary pixels)
    def calculate_Ir(image):
        # Extract the boundary pixels
        boundary_pixels = np.concatenate([
            image[0, :],      # Top edge
            image[-1, :],     # Bottom edge
            image[1:-1, 0],   # Left edge (excluding corners)
            image[1:-1, -1]   # Right edge (excluding corners)
        ])
        # Return median as a robust estimate of background
        return np.median(boundary_pixels)
    
    # Define function to convert intensity to contrast
    def calculate_contrast(image, Ir):
        return (image - Ir) / Ir
    
    # Extract the current frame for display
    nop_intensity = exp_ipsf[k, :, :].reshape([nx, nx])
    nop_predicted_intensity = predicted_images[k]
    
    # Calculate Ir for both experimental and predicted data
    Ir_exp = calculate_Ir(nop_intensity)
    Ir_pred = calculate_Ir(nop_predicted_intensity)
    
    # Convert intensity to contrast
    nop_exp_contrast = calculate_contrast(nop_intensity, Ir_exp)
    nop_pred_contrast = calculate_contrast(nop_predicted_intensity, Ir_pred)
    
    # Extract intensity profiles along x and y directions
    exp_x_intensity = exp_ipsf[k, nx_half, :]
    pred_x_intensity = predicted_images[k, nx_half, :]
    
    # Convert 1D slices to contrast
    exp_x_contrast = calculate_contrast(exp_x_intensity, Ir_exp)
    pred_x_contrast = calculate_contrast(pred_x_intensity, Ir_pred)
    
    # Set up colormap normalization for contrast values
    # Use the same scale for both experimental and predicted contrast for easier comparison
    min_contrast = min(np.min(nop_exp_contrast), np.min(nop_pred_contrast))
    max_contrast = max(np.max(nop_exp_contrast), np.max(nop_pred_contrast))
    
    norm_contrast = Normalize(vmin=min_contrast, vmax=max_contrast)
    
    # Plot images
    im_exp = ax_exp.imshow(nop_exp_contrast, norm=norm_contrast, cmap='gray')
    ax_exp.set_title('Experimental Image', fontsize=14)
    ax_exp.set_xticks([])
    ax_exp.set_yticks([])
    
    im_pred = ax_pred.imshow(nop_pred_contrast, norm=norm_contrast, cmap='gray')
    ax_pred.set_title('Predicted Image', fontsize=14)
    ax_pred.set_xticks([])
    ax_pred.set_yticks([])
    
    # Add colorbar for contrast (shared between both images) - positioned in the middle
    cbar_ax = fig.add_axes([0.1, 0.39, 0.8, 0.02])
    cbar = fig.colorbar(im_exp, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Contrast')
    
    # Frame counter text
    frame_text = fig.text(0.5, 0.96, f'Frame: {k+1}/{frame_number}', 
                         ha='center', fontsize=14)
    
    # Calculate line plot data
    wid = (pixel_physicalsize) / M * nx
    x_w = np.linspace(-0.5*wid, 0.5*wid, nx)
    
    # Plot 1D slices on the same axes (since now both are in contrast units)
    line_exp, = ax_line.plot(x_w, exp_x_contrast, 'b-', linewidth=2, label='Experimental')
    line_pred, = ax_line.plot(x_w, pred_x_contrast, 'r-', linewidth=2, label='Predicted')
    
    # Configure line plot axes
    ax_line.set_xlabel(r'Position ($\mu$m)', fontsize=12)
    ax_line.set_ylabel('Contrast', fontsize=12)
    ax_line.tick_params(axis='y')
    
    # Add a legend
    ax_line.legend(loc='upper right')
    
    # Create slider for frame selection
    slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=0,
        valmax=frame_number-1,
        valinit=k,
        valstep=1
    )
    
    # Define update function for slider
    def update(val):
        frame_idx = int(slider.val)
        
        # Update experimental and predicted images
        nop_intensity = exp_ipsf[frame_idx, :, :].reshape([nx, nx])
        nop_predicted_intensity = predicted_images[frame_idx]
        
        # Calculate Ir for both experimental and predicted data
        Ir_exp = calculate_Ir(nop_intensity)
        Ir_pred = calculate_Ir(nop_predicted_intensity)
        
        # Convert intensity to contrast
        nop_exp_contrast = calculate_contrast(nop_intensity, Ir_exp)
        nop_pred_contrast = calculate_contrast(nop_predicted_intensity, Ir_pred)
        
        # Extract updated 1D profiles
        exp_x_intensity = exp_ipsf[frame_idx, nx_half, :]
        pred_x_intensity = predicted_images[frame_idx, nx_half, :]
        
        # Convert 1D slices to contrast
        exp_x_contrast = calculate_contrast(exp_x_intensity, Ir_exp)
        pred_x_contrast = calculate_contrast(pred_x_intensity, Ir_pred)
        
        # Update the image plots
        im_exp.set_data(nop_exp_contrast)
        im_pred.set_data(nop_pred_contrast)
        
        # Update shared normalization limits
        min_contrast = min(np.min(nop_exp_contrast), np.min(nop_pred_contrast))
        max_contrast = max(np.max(nop_exp_contrast), np.max(nop_pred_contrast))
        
        norm_contrast = Normalize(vmin=min_contrast, vmax=max_contrast)
        
        im_exp.set_norm(norm_contrast)
        im_pred.set_norm(norm_contrast)
        
        # Update colorbar
        cbar.update_normal(im_exp)
        
        # Update line plots
        line_exp.set_ydata(exp_x_contrast)
        line_pred.set_ydata(pred_x_contrast)
        
        # Update y-axis limits for line plot to accommodate both lines
        min_y = min(np.min(exp_x_contrast), np.min(pred_x_contrast)) * 0.95
        max_y = max(np.max(exp_x_contrast), np.max(pred_x_contrast)) * 1.05
        ax_line.set_ylim(min_y, max_y)
        
        # Update the frame counter and background info
        frame_text.set_text(f'Frame: {frame_idx+1}/{frame_number}')
        
        fig.canvas.draw_idle()
    
    # Set initial y-axis limits for line plot
    min_y = min(np.min(exp_x_contrast), np.min(pred_x_contrast)) * 0.95
    max_y = max(np.max(exp_x_contrast), np.max(pred_x_contrast)) * 1.05
    ax_line.set_ylim(min_y, max_y)
    
    # Connect the slider to the update function
    slider.on_changed(update)
    
    # Add keyboard event handler for left/right arrow keys
    def on_key_press(event):
        if event.key == 'right':
            new_val = min(slider.val + 1, slider.valmax)
            slider.set_val(new_val)
        elif event.key == 'left':
            new_val = max(slider.val - 1, slider.valmin)
            slider.set_val(new_val)
    
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Add scroll event handler
    def on_scroll(event):
        if event.button == 'up':
            new_val = min(slider.val + 1, slider.valmax)
        else:
            new_val = max(slider.val - 1, slider.valmin)
        slider.set_val(new_val)
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    # Add click event handler for the slider
    def on_click(event):
        if ax_slider.in_axes(event):
            # Calculate the new value based on click position
            val_range = slider.valmax - slider.valmin
            val = slider.valmin + val_range * (event.xdata - ax_slider.get_position().x0) / ax_slider.get_position().width
            val = min(max(val, slider.valmin), slider.valmax)
            slider.set_val(int(val))
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Add a single horizontal line showing the slice position (instead of crosshair)
    ax_exp.axhline(y=nx_half, color='r', linestyle='--', alpha=0.7)
    ax_pred.axhline(y=nx_half, color='r', linestyle='--', alpha=0.7)
        
    # Use a cleaner layout and explicitly avoid tightly packed elements
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, hspace=0.4)
    
    # Add a text note about keyboard controls
    fig.text(0.02, 0.01, "Navigation: Use slider, mouse wheel, or left/right arrow keys", 
             fontsize=10, ha='left', va='bottom')
        
    plt.show()

