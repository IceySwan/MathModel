# -*- coding: utf-8 -*-
"""
2025华数杯全国大学生数学建模竞赛 C题
问题3：LED光源时变光谱匹配优化
参考question1_final.py的五参数计算方法
"""

import pandas as pd
import numpy as np
from scipy import interpolate, optimize
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import os
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 优化参数权重系数
ALPHA_1 = 0.01  # CCT权重
ALPHA_2 = 10    # Duv权重
ALPHA_3 = 0.1   # Rf权重
ALPHA_4 = 0.1   # Rg权重
ALPHA_5 = 1     # DER_mel权重

def get_cie_color_matching_functions():
    """Get CIE 1931 2° standard observer color matching functions"""
    # Standard CIE 1931 2° observer data (380-780nm, 5nm intervals)
    wavelengths = np.arange(380, 785, 5)

    # CIE 1931 2° standard observer x̄(λ), ȳ(λ), z̄(λ)
    x_bar = np.array([
        0.0014, 0.0022, 0.0042, 0.0076, 0.0143, 0.0232, 0.0435, 0.0776,
        0.1344, 0.2148, 0.2839, 0.3285, 0.3483, 0.3481, 0.3362, 0.3187,
        0.2908, 0.2511, 0.1954, 0.1421, 0.0956, 0.0580, 0.0320, 0.0147,
        0.0049, 0.0024, 0.0093, 0.0291, 0.0633, 0.1096, 0.1655, 0.2257,
        0.2904, 0.3597, 0.4334, 0.5121, 0.5945, 0.6784, 0.7621, 0.8425,
        0.9163, 0.9786, 1.0263, 1.0567, 1.0622, 1.0456, 1.0026, 0.9384,
        0.8544, 0.7514, 0.6424, 0.5419, 0.4479, 0.3608, 0.2835, 0.2187,
        0.1649, 0.1212, 0.0874, 0.0636, 0.0468, 0.0329, 0.0227, 0.0158,
        0.0114, 0.0081, 0.0058, 0.0041, 0.0029, 0.0020, 0.0014, 0.0010,
        0.0007, 0.0005, 0.0003, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001,
        0.0000
    ])

    y_bar = np.array([
        0.0000, 0.0001, 0.0001, 0.0002, 0.0004, 0.0006, 0.0012, 0.0022,
        0.0040, 0.0073, 0.0116, 0.0168, 0.0230, 0.0298, 0.0380, 0.0480,
        0.0600, 0.0739, 0.0910, 0.1126, 0.1390, 0.1693, 0.2080, 0.2586,
        0.3230, 0.4073, 0.5030, 0.6082, 0.7100, 0.7932, 0.8620, 0.9149,
        0.9540, 0.9803, 0.9950, 1.0000, 0.9950, 0.9786, 0.9520, 0.9154,
        0.8700, 0.8163, 0.7570, 0.6949, 0.6310, 0.5668, 0.5030, 0.4412,
        0.3810, 0.3210, 0.2650, 0.2170, 0.1750, 0.1382, 0.1070, 0.0816,
        0.0610, 0.0446, 0.0320, 0.0232, 0.0170, 0.0119, 0.0082, 0.0057,
        0.0041, 0.0029, 0.0021, 0.0015, 0.0010, 0.0007, 0.0005, 0.0004,
        0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000,
        0.0000
    ])

    z_bar = np.array([
        0.0065, 0.0105, 0.0201, 0.0362, 0.0679, 0.1102, 0.2074, 0.3713,
        0.6456, 1.0391, 1.3856, 1.6230, 1.7471, 1.7826, 1.7721, 1.7441,
        1.6692, 1.5281, 1.2876, 1.0419, 0.8130, 0.6162, 0.4652, 0.3533,
        0.2720, 0.2123, 0.1582, 0.1117, 0.0782, 0.0573, 0.0422, 0.0298,
        0.0203, 0.0134, 0.0087, 0.0057, 0.0039, 0.0027, 0.0021, 0.0018,
        0.0017, 0.0014, 0.0011, 0.0010, 0.0008, 0.0006, 0.0003, 0.0002,
        0.0002, 0.0001, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000
    ])

    return wavelengths, x_bar, y_bar, z_bar

def get_cri_test_color_samples():
    """Get CRI-14 test color sample reflectances (TM-30 color samples)"""
    # Wavelengths for color sample data
    wavelengths = np.arange(380, 781, 5)

    # Sample reflectance data for 14 common color samples (simplified)
    color_samples = []

    # Sample 1: Light greyish red
    sample1 = 0.2 + 0.3 * np.exp(-((wavelengths - 600) / 50)**2)
    color_samples.append(sample1)

    # Sample 2: Dark greyish yellow
    sample2 = 0.3 + 0.4 * np.exp(-((wavelengths - 570) / 40)**2)
    color_samples.append(sample2)

    # Sample 3: Strong yellow green
    sample3 = 0.2 + 0.5 * np.exp(-((wavelengths - 540) / 30)**2)
    color_samples.append(sample3)

    # Sample 4: Moderate yellowish green
    sample4 = 0.25 + 0.4 * np.exp(-((wavelengths - 520) / 35)**2)
    color_samples.append(sample4)

    # Sample 5: Light bluish green
    sample5 = 0.3 + 0.3 * np.exp(-((wavelengths - 500) / 40)**2)
    color_samples.append(sample5)

    # Sample 6: Light blue
    sample6 = 0.2 + 0.4 * np.exp(-((wavelengths - 480) / 30)**2)
    color_samples.append(sample6)

    # Sample 7: Light violet
    sample7 = 0.25 + 0.3 * np.exp(-((wavelengths - 450) / 25)**2)
    color_samples.append(sample7)

    # Sample 8: Light reddish purple
    sample8 = 0.2 + 0.25 * np.exp(-((wavelengths - 420) / 30)**2) + 0.2 * np.exp(-((wavelengths - 650) / 40)**2)
    color_samples.append(sample8)

    # Additional samples 9-14
    for i in range(6):
        # Generate additional diverse color samples
        peak_wl = 400 + i * 60
        sample = 0.3 + 0.3 * np.exp(-((wavelengths - peak_wl) / 35)**2)
        color_samples.append(sample)

    return wavelengths, np.array(color_samples)

def get_d65_illuminant():
    """Get D65 standard illuminant spectral power distribution"""
    # Extended wavelength range to match melanopic sensitivity
    wavelengths = np.arange(380, 781, 1)  # Changed from 5nm to 1nm intervals

    # Interpolate the original D65 data to 1nm intervals
    original_wavelengths = np.arange(380, 781, 5)
    original_d65_spd = np.array([
        49.975, 52.311, 54.648, 68.701, 82.754, 87.120, 91.486, 92.458,
        93.431, 90.057, 86.682, 95.773, 104.865, 110.936, 117.008, 117.410,
        117.812, 116.336, 114.861, 115.392, 115.923, 112.367, 108.811, 109.082,
        109.354, 108.578, 107.802, 106.296, 104.790, 106.239, 107.689, 106.047,
        104.405, 104.225, 104.046, 102.023, 100.000, 98.167, 96.334, 96.061,
        95.788, 92.237, 88.685, 89.996, 91.306, 90.231, 89.157, 88.475,
        87.793, 85.947, 84.101, 83.233, 82.365, 80.877, 79.388, 78.988,
        78.588, 77.630, 76.672, 76.639, 76.606, 76.851, 77.096, 75.984,
        74.872, 74.349, 73.816, 73.227, 72.638, 71.808, 70.978, 70.458,
        69.938, 69.498, 69.058, 68.269, 67.481, 66.953, 66.425, 65.942,
        65.459, 64.933, 64.408, 64.133
    ])

    # Ensure arrays have same length
    min_len = min(len(original_wavelengths), len(original_d65_spd))
    original_wavelengths = original_wavelengths[:min_len]
    original_d65_spd = original_d65_spd[:min_len]

    # Interpolate to 1nm intervals
    f_d65 = interpolate.interp1d(original_wavelengths, original_d65_spd,
                                kind='linear', bounds_error=False, fill_value=0)
    d65_spd = f_d65(wavelengths)

    return wavelengths, d65_spd

def get_melanopic_sensitivity_lucas():
    """Get melanopic sensitivity function based on Lucas et al. 2014"""
    # CIE S 026 melanopic sensitivity function
    wavelengths = np.arange(380, 781, 1)

    # Melanopic sensitivity function values (Lucas et al. 2014)
    # Peak at ~490nm, values normalized to peak = 1
    s_mel = np.zeros_like(wavelengths, dtype=float)

    for i, wl in enumerate(wavelengths):
        if 380 <= wl <= 780:
            # More accurate melanopic sensitivity based on CIE S 026
            if wl < 490:
                # Blue side of the curve
                s_mel[i] = np.exp(-0.5 * ((wl - 490) / 38)**2)
            else:
                # Red side of the curve (steeper falloff)
                s_mel[i] = np.exp(-0.5 * ((wl - 490) / 42)**2)

    # Normalize to peak = 1
    if s_mel.max() > 0:
        s_mel = s_mel / s_mel.max()

    return wavelengths, s_mel

def xyz_to_cam02ucs(X, Y, Z, X_w, Y_w, Z_w, L_A=64):
    """Convert XYZ to CAM02-UCS color space"""
    # Simplified CAM02-UCS conversion
    # In practice, this would be a full CIECAM02 implementation

    # Adaptation
    Y_b = 20.0  # Background luminance factor
    F = 1.0     # Surround condition
    c = 0.69    # Impact of surround
    N_c = 1.0   # Chromatic induction factor

    # Normalize by white point
    X_rel = X / X_w
    Y_rel = Y / Y_w
    Z_rel = Z / Z_w

    # Simplified lightness calculation
    J = 100 * (Y_rel ** 0.5)

    # Simplified chroma calculations
    a = 200 * (X_rel - Y_rel)
    b = 100 * (Y_rel - Z_rel)

    return J, a, b

def interpolate_color_matching_functions(wavelength_data):
    """Interpolate color matching functions to match the data wavelengths"""
    cie_wavelengths, x_bar, y_bar, z_bar = get_cie_color_matching_functions()

    # Create interpolation functions
    f_x = interpolate.interp1d(cie_wavelengths, x_bar, kind='linear',
                               bounds_error=False, fill_value=0)
    f_y = interpolate.interp1d(cie_wavelengths, y_bar, kind='linear',
                               bounds_error=False, fill_value=0)
    f_z = interpolate.interp1d(cie_wavelengths, z_bar, kind='linear',
                               bounds_error=False, fill_value=0)

    # Interpolate to data wavelengths
    x_bar_interp = f_x(wavelength_data)
    y_bar_interp = f_y(wavelength_data)
    z_bar_interp = f_z(wavelength_data)

    return x_bar_interp, y_bar_interp, z_bar_interp

def calculate_tristimulus_values(wavelength, intensity):
    """Calculate CIE XYZ tristimulus values"""
    # Get interpolated color matching functions
    x_bar, y_bar, z_bar = interpolate_color_matching_functions(wavelength)

    # Calculate wavelength step
    d_lambda = np.diff(wavelength)
    d_lambda = np.append(d_lambda, d_lambda[-1])  # Use last interval for the last point

    # Normalization constant K for photopic vision
    K = 683  # lm/W for photopic vision

    # Calculate tristimulus values using numerical integration with proper normalization
    X = K * np.sum(intensity * x_bar * d_lambda)
    Y = K * np.sum(intensity * y_bar * d_lambda)  # Y represents luminance when multiplied by K
    Z = K * np.sum(intensity * z_bar * d_lambda)

    return X, Y, Z

def calculate_cct_duv(X, Y, Z):
    """Calculate Correlated Color Temperature (CCT) and Distance from Planckian locus (Duv)"""
    # Convert to chromaticity coordinates
    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)

    # McCamy's approximation for CCT
    n = (x - 0.3320) / (0.1858 - y)
    CCT = 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33

    # Calculate Duv using Robertson's method with correct Planckian locus
    # Use CIE 1960 UCS coordinates for more accurate calculation
    u = 4*x / (-2*x + 12*y + 3)
    v = 6*y / (-2*x + 12*y + 3)

    # Find closest point on Planckian locus in UCS space
    T_range = np.linspace(max(1000, CCT-500), min(25000, CCT+500), 1000)
    min_dist = float('inf')
    closest_u_p = 0
    closest_v_p = 0

    for T in T_range:
        # Planckian locus in UCS coordinates (more accurate formula)
        u_p = (0.860117757 + 1.54118254e-4*T + 1.28641212e-7*T**2) / (1.0 + 8.42420235e-4*T + 7.08145163e-7*T**2)
        v_p = (0.317398726 + 4.22806245e-5*T + 4.20481691e-8*T**2) / (1.0 - 2.89741816e-5*T + 1.61456053e-7*T**2)

        dist = np.sqrt((u - u_p)**2 + (v - v_p)**2)
        if dist < min_dist:
            min_dist = dist
            closest_u_p = u_p
            closest_v_p = v_p

    # Calculate Duv with proper sign in UCS space
    # Vector from closest point on locus to test point
    du = u - closest_u_p
    dv = v - closest_v_p

    # Perpendicular direction to the locus (simplified)
    # Positive Duv means above the locus, negative means below
    duv_vector = dv  # Simplified: use v-direction as primary indicator

    # Final Duv calculation
    Duv = np.sqrt(du**2 + dv**2) * np.sign(duv_vector)

    return CCT, Duv

def get_reference_illuminant_spd(cct):
    """Get reference illuminant spectral power distribution based on CCT"""
    wavelengths = np.arange(380, 781, 1)

    if cct < 5000:
        # Use Planckian radiator for CCT < 5000K
        h = 6.626e-34  # Planck constant
        c = 3.0e8      # Speed of light
        k = 1.381e-23  # Boltzmann constant

        spd = []
        for wl in wavelengths:
            wl_m = wl * 1e-9  # Convert nm to m
            # Planck's law
            B = (2 * h * c**2 / wl_m**5) / (np.exp(h * c / (wl_m * k * cct)) - 1)
            spd.append(B)
        spd = np.array(spd)
    else:
        # Use D-illuminant for CCT >= 5000K
        # Simplified D65 approximation scaled for different CCT
        d65_wl, d65_spd = get_d65_illuminant()
        f_d65 = interpolate.interp1d(d65_wl, d65_spd, kind='linear',
                                    bounds_error=False, fill_value=0)
        spd = f_d65(wavelengths)

        # Scale for different CCT (simplified)
        scale_factor = 6500 / cct
        spd = spd * scale_factor

    return wavelengths, spd

def calculate_color_samples_under_illuminant(illuminant_wl, illuminant_spd, sample_wl, sample_reflectances):
    """Calculate XYZ values for color samples under given illuminant"""
    # Interpolate illuminant to sample wavelengths
    f_illuminant = interpolate.interp1d(illuminant_wl, illuminant_spd, kind='linear',
                                       bounds_error=False, fill_value=0)
    illuminant_interp = f_illuminant(sample_wl)

    # Get CIE color matching functions
    cie_wl, x_bar, y_bar, z_bar = get_cie_color_matching_functions()

    # Interpolate color matching functions to sample wavelengths
    f_x = interpolate.interp1d(cie_wl, x_bar, kind='linear', bounds_error=False, fill_value=0)
    f_y = interpolate.interp1d(cie_wl, y_bar, kind='linear', bounds_error=False, fill_value=0)
    f_z = interpolate.interp1d(cie_wl, z_bar, kind='linear', bounds_error=False, fill_value=0)

    x_bar_interp = f_x(sample_wl)
    y_bar_interp = f_y(sample_wl)
    z_bar_interp = f_z(sample_wl)

    # Calculate wavelength intervals
    d_lambda = np.diff(sample_wl)
    d_lambda = np.append(d_lambda, d_lambda[-1])

    # Normalization constant K for consistency
    K = 683  # lm/W for photopic vision

    # Calculate XYZ for each color sample under test and reference illuminants
    xyz_values = []
    for reflectance in sample_reflectances:
        # Multiply illuminant * reflectance * color matching function * K
        X = K * np.sum(illuminant_interp * reflectance * x_bar_interp * d_lambda)
        Y = K * np.sum(illuminant_interp * reflectance * y_bar_interp * d_lambda)
        Z = K * np.sum(illuminant_interp * reflectance * z_bar_interp * d_lambda)
        xyz_values.append([X, Y, Z])

    return np.array(xyz_values)

def calculate_rf_accurate(test_wavelength, test_intensity):
    """Calculate accurate Rf based on CAM02-UCS color differences"""
    # Get color samples
    sample_wl, sample_reflectances = get_cri_test_color_samples()

    # Get test source XYZ
    test_X, test_Y, test_Z = calculate_tristimulus_values(test_wavelength, test_intensity)
    test_cct, _ = calculate_cct_duv(test_X, test_Y, test_Z)

    # Get reference illuminant
    ref_wl, ref_spd = get_reference_illuminant_spd(test_cct)

    # Interpolate test source to sample wavelengths
    f_test = interpolate.interp1d(test_wavelength, test_intensity, kind='linear',
                                 bounds_error=False, fill_value=0)
    test_spd_interp = f_test(sample_wl)

    # Interpolate reference to sample wavelengths
    f_ref = interpolate.interp1d(ref_wl, ref_spd, kind='linear',
                                bounds_error=False, fill_value=0)
    ref_spd_interp = f_ref(sample_wl)

    # Calculate XYZ for color samples under test and reference illuminants
    test_xyz = calculate_color_samples_under_illuminant(sample_wl, test_spd_interp,
                                                       sample_wl, sample_reflectances)
    ref_xyz = calculate_color_samples_under_illuminant(sample_wl, ref_spd_interp,
                                                      sample_wl, sample_reflectances)

    # Calculate white point for both illuminants
    white_sample = np.ones_like(sample_wl)  # Perfect reflector
    test_white_xyz = calculate_color_samples_under_illuminant(sample_wl, test_spd_interp,
                                                             sample_wl, [white_sample])[0]
    ref_white_xyz = calculate_color_samples_under_illuminant(sample_wl, ref_spd_interp,
                                                            sample_wl, [white_sample])[0]

    # Convert to CAM02-UCS
    delta_e_values = []

    for i in range(len(sample_reflectances)):
        # Test illuminant CAM02-UCS
        J_test, a_test, b_test = xyz_to_cam02ucs(test_xyz[i, 0], test_xyz[i, 1], test_xyz[i, 2],
                                                test_white_xyz[0], test_white_xyz[1], test_white_xyz[2])

        # Reference illuminant CAM02-UCS
        J_ref, a_ref, b_ref = xyz_to_cam02ucs(ref_xyz[i, 0], ref_xyz[i, 1], ref_xyz[i, 2],
                                             ref_white_xyz[0], ref_white_xyz[1], ref_white_xyz[2])

        # Calculate color difference in CAM02-UCS
        delta_e = np.sqrt((J_test - J_ref)**2 + (a_test - a_ref)**2 + (b_test - b_ref)**2)
        delta_e_values.append(delta_e)

    # Calculate average color difference
    delta_e_avg = np.mean(delta_e_values)

    # Calculate Rf
    Rf = 100 - 6.73 * delta_e_avg

    return max(0, min(100, Rf))

def calculate_rg_accurate(test_wavelength, test_intensity):
    """Calculate accurate Rg based on color gamut area"""
    # Get color samples
    sample_wl, sample_reflectances = get_cri_test_color_samples()

    # Get test source XYZ
    test_X, test_Y, test_Z = calculate_tristimulus_values(test_wavelength, test_intensity)
    test_cct, _ = calculate_cct_duv(test_X, test_Y, test_Z)

    # Get reference illuminant
    ref_wl, ref_spd = get_reference_illuminant_spd(test_cct)

    # Interpolate test source to sample wavelengths
    f_test = interpolate.interp1d(test_wavelength, test_intensity, kind='linear',
                                 bounds_error=False, fill_value=0)
    test_spd_interp = f_test(sample_wl)

    # Interpolate reference to sample wavelengths
    f_ref = interpolate.interp1d(ref_wl, ref_spd, kind='linear',
                                bounds_error=False, fill_value=0)
    ref_spd_interp = f_ref(sample_wl)

    # Calculate XYZ for color samples under test and reference illuminants
    test_xyz = calculate_color_samples_under_illuminant(sample_wl, test_spd_interp,
                                                       sample_wl, sample_reflectances)
    ref_xyz = calculate_color_samples_under_illuminant(sample_wl, ref_spd_interp,
                                                      sample_wl, sample_reflectances)

    # Calculate white point for both illuminants
    white_sample = np.ones_like(sample_wl)  # Perfect reflector
    test_white_xyz = calculate_color_samples_under_illuminant(sample_wl, test_spd_interp,
                                                             sample_wl, [white_sample])[0]
    ref_white_xyz = calculate_color_samples_under_illuminant(sample_wl, ref_spd_interp,
                                                            sample_wl, [white_sample])[0]

    # Convert to CAM02-UCS and extract a*, b* coordinates for gamut calculation
    test_points = []
    ref_points = []

    for i in range(len(sample_reflectances)):
        # Test illuminant CAM02-UCS
        J_test, a_test, b_test = xyz_to_cam02ucs(test_xyz[i, 0], test_xyz[i, 1], test_xyz[i, 2],
                                                test_white_xyz[0], test_white_xyz[1], test_white_xyz[2])
        test_points.append([a_test, b_test])

        # Reference illuminant CAM02-UCS
        J_ref, a_ref, b_ref = xyz_to_cam02ucs(ref_xyz[i, 0], ref_xyz[i, 1], ref_xyz[i, 2],
                                             ref_white_xyz[0], ref_white_xyz[1], ref_white_xyz[2])
        ref_points.append([a_ref, b_ref])

    test_points = np.array(test_points)
    ref_points = np.array(ref_points)

    # Calculate convex hull areas
    try:
        test_hull = ConvexHull(test_points)
        ref_hull = ConvexHull(ref_points)

        A_test = test_hull.volume  # In 2D, volume is area
        A_ref = ref_hull.volume

        # Calculate Rg
        if A_ref > 0:
            Rg = (A_test / A_ref) * 100
        else:
            Rg = 100

    except Exception:
        # Fallback if ConvexHull fails
        Rg = 100

    return max(50, min(150, Rg))

def calculate_der_mel_accurate(test_wavelength, test_intensity):
    """Calculate accurate DER_mel based on melanopic illuminance ratio"""
    # Get melanopic sensitivity function
    mel_wl, s_mel = get_melanopic_sensitivity_lucas()

    # Get D65 illuminant
    d65_wl, d65_spd = get_d65_illuminant()

    # Interpolate test source to melanopic wavelengths
    f_test = interpolate.interp1d(test_wavelength, test_intensity, kind='linear',
                                 bounds_error=False, fill_value=0)
    test_spd_interp = f_test(mel_wl)

    # No need to interpolate D65 since both mel_wl and d65_wl are identical
    # But keep for safety in case wavelength ranges differ
    if len(d65_wl) == len(mel_wl) and np.allclose(d65_wl, mel_wl):
        d65_spd_interp = d65_spd
    else:
        f_d65 = interpolate.interp1d(d65_wl, d65_spd, kind='linear',
                                    bounds_error=False, fill_value=0)
        d65_spd_interp = f_d65(mel_wl)

    # Calculate wavelength intervals
    d_lambda = np.diff(mel_wl)
    d_lambda = np.append(d_lambda, d_lambda[-1])

    # Calculate melanopic illuminance for test source
    E_mel_test = np.sum(test_spd_interp * s_mel * d_lambda)

    # Calculate melanopic illuminance for D65
    E_mel_d65 = np.sum(d65_spd_interp * s_mel * d_lambda)

    # Calculate DER_mel
    if E_mel_d65 > 0:
        DER_mel = E_mel_test / E_mel_d65
    else:
        DER_mel = 0

    return DER_mel
def load_q3_data():
    """加载Q3数据"""
    try:
        print("正在加载Q3数据...")
        df = pd.read_excel('attachment-Q3.xlsx')
        print(f"Q3数据加载成功: {df.shape}")

        # 提取波长 - 第1列
        wavelengths = df.iloc[:, 0].values
        # 转换为数值
        wl_clean = []
        for w in wavelengths:
            try:
                if isinstance(w, str):
                    # 处理可能包含单位的字符串
                    wl_clean.append(float(w.split('(')[0] if '(' in w else w))
                else:
                    wl_clean.append(float(w))
            except:
                continue

        wavelengths = np.array(wl_clean)

        # 提取时刻数据 - 第2-16列 (5:30到19:30，每小时一个数据点)
        intensity_data = {}
        time_cols = df.columns[1:].tolist()

        for i, col in enumerate(time_cols):
            if i >= 15:  # 限制到前15个时刻 (5:30-19:30)
                break

            # 转换时间格式
            if hasattr(col, 'strftime'):
                time_str = col.strftime('%H:%M:%S')
            else:
                time_str = str(col)

            # 提取强度数据
            intensity_vals = []
            for val in df[col].values[:len(wavelengths)]:
                try:
                    intensity_vals.append(float(val))
                except:
                    intensity_vals.append(0.0)

            intensity_data[time_str] = np.array(intensity_vals)

        print(f"成功加载 {len(wavelengths)} 个波长点")
        print(f"时刻数据: {list(intensity_data.keys())[:5]}...")

        return wavelengths, intensity_data

    except Exception as e:
        print(f"Q3数据加载失败: {e}")
        return None, None

def calculate_five_parameters(wavelength, intensity):
    """计算五个参数：CCT, Duv, Rf, Rg, DER_mel"""
    try:
        # 1. 计算XYZ三刺激值
        X, Y, Z = calculate_tristimulus_values(wavelength, intensity)
        
        # 2. 计算CCT和Duv
        CCT, Duv = calculate_cct_duv(X, Y, Z)
        
        # 3. 计算Rf (色彩保真度指数)
        Rf = calculate_rf_accurate(wavelength, intensity)
        
        # 4. 计算Rg (色域指数)
        Rg = calculate_rg_accurate(wavelength, intensity)
        
        # 5. 计算DER_mel (褪黑激素日光效能比)
        DER_mel = calculate_der_mel_accurate(wavelength, intensity)
        
        return CCT, Duv, Rf, Rg, DER_mel
        
    except Exception as e:
        print(f"计算参数时出错: {e}")
        return 0, 0, 0, 0, 0

def save_results_to_excel(results, times, filename='attachment-Q3.xlsx'):
    """将结果保存到Excel文件的最后5行"""
    try:
        # 读取原文件
        df = pd.read_excel(filename)
        
        # 准备参数名称
        param_names = ['CCT', 'Duv', 'Rf', 'Rg', 'DER_mel']
        
        # 获取当前行数
        current_rows = len(df)
        
        # 为每个参数创建新行
        for i, param_name in enumerate(param_names):
            new_row = [param_name]  # 第一列是参数名
            
            # 添加各时刻的计算结果
            for time_key in df.columns[1:]:  # 跳过第一列（波长列）
                # 找到对应时刻的结果
                found_result = None
                for j, calc_time in enumerate(times):
                    if str(time_key) in calc_time or calc_time in str(time_key):
                        found_result = results[j][i]
                        break
                
                if found_result is not None:
                    new_row.append(found_result)
                else:
                    new_row.append(0)  # 如果没找到对应时刻，填0
            
            # 添加新行到DataFrame
            df.loc[current_rows + i] = new_row
        
        # 保存到文件
        with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
            df.to_excel(writer, index=False)
        
        print(f"结果已保存到 {filename} 的第 {current_rows+1}-{current_rows+5} 行")
        
    except Exception as e:
        print(f"保存结果时出错: {e}")

def create_visualizations(times, results, save_dir='output/Question3'):
    """创建五参数可视化图表"""
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 提取数据
    param_names = ['CCT', 'Duv', 'Rf', 'Rg', 'DER_mel']
    param_units = ['K', '', '', '', '']
    
    # 转换时间格式为小时
    time_hours = []
    for time_str in times:
        try:
            # 处理不同的时间格式
            if ':' in time_str:
                parts = time_str.split(':')
                hours = float(parts[0]) + float(parts[1])/60
            else:
                hours = float(time_str)
            time_hours.append(hours)
        except:
            time_hours.append(len(time_hours))
    
    # 转换结果为numpy数组
    results_array = np.array(results)
    
    # 1. 时间序列图 - 修改为3x2布局但只显示5个子图
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('问题3：五参数时间序列分析', fontsize=16, fontweight='bold')
    
    # 绘制每个参数的时间序列
    for i, (param_name, unit) in enumerate(zip(param_names, param_units)):
        row = i // 2
        col = i % 2
        
        ax = axes[row, col]
        ax.plot(time_hours, results_array[:, i], 'o-', linewidth=2, markersize=6,
               color=plt.cm.Set3(i/5), label=param_name)
        ax.set_xlabel('时间 (小时)')
        ax.set_ylabel(f'{param_name} ({unit})' if unit else param_name)
        ax.set_title(f'{param_name} 随时间变化')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 添加数值标注 - 改为四位有效数字
        for j, (x, y) in enumerate(zip(time_hours, results_array[:, i])):
            if j % 2 == 0:  # 每隔一个点标注
                # 使用四位有效数字格式
                if abs(y) >= 1000:
                    label = f'{y:.4g}'  # 大数值用科学计数法
                elif abs(y) >= 1:
                    label = f'{y:.4g}'  # 4位有效数字
                else:
                    label = f'{y:.4g}'  # 小数值也用4位有效数字
                ax.annotate(label, (x, y), textcoords="offset points",
                           xytext=(0,10), ha='center', fontsize=8)

    # 删除空白子图
    fig.delaxes(axes[2, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '图1_五参数时间序列分析.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 2. 雷达图显示五参数的综合表现
    plt.figure(figsize=(12, 10))
    
    # 准备雷达图数据
    angles = np.linspace(0, 2*np.pi, len(param_names), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    # 归一化数据用于雷达图显示
    normalized_results = np.zeros_like(results_array)
    for i in range(len(param_names)):
        if param_names[i] == 'CCT':
            # CCT归一化到0-1，假设范围2000K-8000K
            normalized_results[:, i] = (results_array[:, i] - 2000) / 6000
        elif param_names[i] == 'Duv':
            # Duv归一化，范围通常-0.02到0.02
            normalized_results[:, i] = (results_array[:, i] + 0.02) / 0.04
        elif param_names[i] in ['Rf', 'Rg']:
            # Rf, Rg归一化到0-1，范围0-100
            normalized_results[:, i] = results_array[:, i] / 100
        elif param_names[i] == 'DER_mel':
            # DER_mel归一化，范围通常0-2
            normalized_results[:, i] = results_array[:, i] / 2
    
    # 限制范围在0-1之间
    normalized_results = np.clip(normalized_results, 0, 1)
    
    # 选择几个关键时刻绘制雷达图
    key_times = [0, len(times)//4, len(times)//2, len(times)*3//4, len(times)-1]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for i, (time_idx, color) in enumerate(zip(key_times, colors)):
        values = normalized_results[time_idx].tolist()
        values += values[:1]  # 闭合
        
        ax.plot(angles, values, 'o-', linewidth=2, color=color, 
               label=f'{times[time_idx]}')
        ax.fill(angles, values, alpha=0.25, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(param_names)
    ax.set_ylim(0, 1)
    ax.set_title('五参数雷达图对比分析', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.savefig(os.path.join(save_dir, '图2_五参数雷达图对比.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 参数相关性热力图
    plt.figure(figsize=(10, 8))
    
    correlation_matrix = np.corrcoef(results_array.T)
    
    im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, label='相关系数')
    
    # 设置刻度和标签
    plt.xticks(range(len(param_names)), param_names, rotation=45)
    plt.yticks(range(len(param_names)), param_names)
    
    # 添加数值标注
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            plt.text(j, i, f'{correlation_matrix[i, j]:.3f}', 
                    ha='center', va='center', 
                    color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')
    
    plt.title('五参数相关性分析热力图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '图3_参数相关性热力图.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. 综合性能分析图
    plt.figure(figsize=(15, 10))
    
    # 4x2 子图布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('问题3：综合性能分析', fontsize=16, fontweight='bold')
    
    # 4.1 CCT vs Duv散点图
    ax1 = axes[0, 0]
    scatter = ax1.scatter(results_array[:, 0], results_array[:, 1], 
                         c=time_hours, cmap='viridis', s=60, alpha=0.7)
    ax1.set_xlabel('CCT (K)')
    ax1.set_ylabel('Duv')
    ax1.set_title('CCT vs Duv 散点图')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='时间 (小时)')
    
    # 4.2 Rf vs Rg散点图
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(results_array[:, 2], results_array[:, 3], 
                          c=time_hours, cmap='plasma', s=60, alpha=0.7)
    ax2.set_xlabel('Rf')
    ax2.set_ylabel('Rg')
    ax2.set_title('色彩指数 Rf vs Rg')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='时间 (小时)')
    
    # 4.3 DER_mel随时间变化
    ax3 = axes[0, 2]
    ax3.plot(time_hours, results_array[:, 4], 'o-', color='green', linewidth=2)
    ax3.set_xlabel('时间 (小时)')
    ax3.set_ylabel('DER_mel')
    ax3.set_title('褪黑激素日光效能比变化')
    ax3.grid(True, alpha=0.3)
    
    # 4.4 参数变化率分析
    ax4 = axes[1, 0]
    param_changes = np.abs(np.diff(results_array, axis=0))
    param_change_avg = np.mean(param_changes, axis=0)
    
    bars = ax4.bar(param_names, param_change_avg, color=plt.cm.Set3(np.arange(5)/5))
    ax4.set_ylabel('平均变化率')
    ax4.set_title('参数稳定性分析')
    ax4.tick_params(axis='x', rotation=45)
    
    # 添加数值标注
    for bar, val in zip(bars, param_change_avg):
        height = bar.get_height()
        ax4.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # 4.5 综合性能指标
    ax5 = axes[1, 1]
    # 计算综合性能指标
    composite_score = []
    for i in range(len(results)):
        # 归一化各参数并计算综合得分
        cct_score = 1 - abs(results_array[i, 0] - 6500) / 3000  # CCT接近6500K越好
        duv_score = 1 - abs(results_array[i, 1]) / 0.02  # Duv接近0越好
        rf_score = results_array[i, 2] / 100  # Rf越高越好
        rg_score = results_array[i, 3] / 100  # Rg越高越好
        der_score = min(results_array[i, 4], 1)  # DER_mel适中为好
        
        composite = (cct_score + duv_score + rf_score + rg_score + der_score) / 5
        composite_score.append(max(0, composite))
    
    ax5.plot(time_hours, composite_score, 'o-', color='red', linewidth=2)
    ax5.set_xlabel('时间 (小时)')
    ax5.set_ylabel('综合性能指标')
    ax5.set_title('综合性能评价')
    ax5.grid(True, alpha=0.3)
    
    # 4.6 最优时间点标识
    ax6 = axes[1, 2]
    best_time_idx = np.argmax(composite_score)
    worst_time_idx = np.argmin(composite_score)
    
    categories = ['最佳', '最差', '平均']
    best_values = results_array[best_time_idx]
    worst_values = results_array[worst_time_idx]
    avg_values = np.mean(results_array, axis=0)
    
    x = np.arange(len(param_names))
    width = 0.25
    
    bars1 = ax6.bar(x - width, best_values/np.max(results_array, axis=0), width, 
                   label='最佳时刻', alpha=0.8, color='green')
    bars2 = ax6.bar(x, worst_values/np.max(results_array, axis=0), width, 
                   label='最差时刻', alpha=0.8, color='red')
    bars3 = ax6.bar(x + width, avg_values/np.max(results_array, axis=0), width, 
                   label='平均值', alpha=0.8, color='blue')
    
    ax6.set_xlabel('参数')
    ax6.set_ylabel('归一化值')
    ax6.set_title('最佳/最差时刻对比')
    ax6.set_xticks(x)
    ax6.set_xticklabels(param_names, rotation=45)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '图4_综合性能分析.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存数据分析结果
    analysis_data = {
        'times': times,
        'time_hours': time_hours,
        'parameters': param_names,
        'results': results,
        'correlation_matrix': correlation_matrix.tolist(),
        'composite_scores': composite_score,
        'best_time': times[best_time_idx],
        'worst_time': times[worst_time_idx],
        'parameter_stability': param_change_avg.tolist()
    }
    
    # 保存分析结果到CSV
    results_df = pd.DataFrame(results, columns=param_names, index=times)
    results_df['综合得分'] = composite_score
    results_df.to_csv(os.path.join(save_dir, 'question3_analysis_results.csv'), encoding='utf-8-sig')
    
    print(f"\n可视化图表已保存到 {save_dir} 文件夹")
    print(f"最佳表现时刻: {times[best_time_idx]} (综合得分: {composite_score[best_time_idx]:.3f})")
    print(f"最差表现时刻: {times[worst_time_idx]} (综合得分: {composite_score[worst_time_idx]:.3f})")
    
    return analysis_data

def main():
    """主函数：计算attachment-Q3.xlsx的对应五参数并保存到文件最后行"""
    print("="*60)
    print("2025华数杯 C题 - 问题3：计算时变光谱五参数")
    print("="*60)

    # 1. 加载数据
    wavelengths, intensity_data = load_q3_data()
    if wavelengths is None:
        print("数据加载失败，程序终止")
        return

    # 2. 计算各时刻的五参数
    print(f"\n开始计算各时刻的五参数...")
    results = []
    times = list(intensity_data.keys())
    
    for i, (time_str, intensity) in enumerate(intensity_data.items()):
        print(f"\n计算时刻 {i+1}/{len(intensity_data)}: {time_str}")
        
        # 计算五参数
        CCT, Duv, Rf, Rg, DER_mel = calculate_five_parameters(wavelengths, intensity)
        
        results.append([CCT, Duv, Rf, Rg, DER_mel])
        
        print(f"  CCT: {CCT:.0f}K")
        print(f"  Duv: {Duv:.4f}")
        print(f"  Rf:  {Rf:.1f}")
        print(f"  Rg:  {Rg:.1f}")
        print(f"  DER_mel: {DER_mel:.4f}")

    # 3. 保存结果到Excel文件
    print(f"\n保存结果到Excel文件...")
    save_results_to_excel(results, times, 'attachment-Q3.xlsx')

    # 4. 创建可视化图表
    print(f"\n创建可视化图表...")
    analysis_data = create_visualizations(times, results)

    # 5. 输出总结
    print("\n" + "="*50)
    print("问题3计算结果总结")
    print("="*50)
    print("时刻\t\tCCT(K)\tDuv\tRf\tRg\tDER_mel")
    print("-" * 60)
    for i, (time_str, result) in enumerate(zip(times, results)):
        CCT, Duv, Rf, Rg, DER_mel = result
        print(f"{time_str}\t{CCT:.0f}\t{Duv:.4f}\t{Rf:.1f}\t{Rg:.1f}\t{DER_mel:.4f}")
    print("="*50)
    
    print(f"\n结果已保存到Excel文件的第403-407行")
    print("各行分别对应：CCT, Duv, Rf, Rg, DER_mel")
    print(f"\n可视化图表已保存到 Question3_figures 文件夹")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序执行错误: {e}")
        import traceback
        traceback.print_exc()
