#!/usr/bin/python3
#
# functions.py
#
# Functions for regenopt (regenopt.py).
#
# Uses parameters.py.
#
# MIT License
#
# Copyright (c) 2024 Natural Resources Institute Finland and Eero
# Holmström (eero.holmstrom@luke.fi)
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#



#
# Import the necessary modules
#



import parameters
import coordinates
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import alphashape
from shapely.geometry import Polygon, Point, MultiPolygon, shape
from shapely.ops import unary_union
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib import colors
import glob



#
# Set the default matplotlib fontsize globally
#

import matplotlib as mpl

mpl.rcParams["font.size"] = parameters.default_fontsize



#
# Read in the simulation data used for the regression models.
#

def read_in_simulation_data():


    #
    # Pure Motti
    #

    pure_motti_data_points_birch_blv_mt = np.loadtxt(parameters.simulation_results_root + "/mt/pure_motti_mt_birch_tsum_and_blv_at_interest_rate_of_" + str(parameters.blv_interest_rate) + "_percent.txt.tsum_unit_1000_degree_days_blv_unit_1000_eur_per_ha")
    pure_motti_data_points_birch_blv_omt = np.loadtxt(parameters.simulation_results_root + "/omt/pure_motti_omt_birch_tsum_and_blv_at_interest_rate_of_" + str(parameters.blv_interest_rate) + "_percent.txt.tsum_unit_1000_degree_days_blv_unit_1000_eur_per_ha")

    pure_motti_data_points_pine_blv_mt = np.loadtxt(parameters.simulation_results_root + "/mt/pure_motti_mt_pine_tsum_and_blv_at_interest_rate_of_" + str(parameters.blv_interest_rate) + "_percent.txt.tsum_unit_1000_degree_days_blv_unit_1000_eur_per_ha")
    pure_motti_data_points_pine_blv_omt = np.loadtxt(parameters.simulation_results_root + "/omt/pure_motti_omt_pine_tsum_and_blv_at_interest_rate_of_" + str(parameters.blv_interest_rate) + "_percent.txt.tsum_unit_1000_degree_days_blv_unit_1000_eur_per_ha")

    pure_motti_data_points_birch_carbon_mt = np.loadtxt(parameters.simulation_results_root + "/mt/pure_motti_mt_birch_tsum_and_carbon.txt.tsum_unit_1000_degree_days_carbon_unit_100_tCO2_per_ha")
    pure_motti_data_points_birch_carbon_omt = np.loadtxt(parameters.simulation_results_root + "/omt/pure_motti_omt_birch_tsum_and_carbon.txt.tsum_unit_1000_degree_days_carbon_unit_100_tCO2_per_ha")

    pure_motti_data_points_pine_carbon_mt = np.loadtxt(parameters.simulation_results_root + "/mt/pure_motti_mt_pine_tsum_and_carbon.txt.tsum_unit_1000_degree_days_carbon_unit_100_tCO2_per_ha")
    pure_motti_data_points_pine_carbon_omt = np.loadtxt(parameters.simulation_results_root + "/omt/pure_motti_omt_pine_tsum_and_carbon.txt.tsum_unit_1000_degree_days_carbon_unit_100_tCO2_per_ha")



    #
    # Motti+Hmodel
    #

    motti_plus_hmodel_data_points_spruce_blv_mt = np.loadtxt(parameters.simulation_results_root + "/mt/motti_plus_hmodel_mt_spruce_tsum_frot_and_blv_at_interest_rate_of_" + str(parameters.blv_interest_rate) + "_percent.txt.tsum_unit_1000_degree_days_blv_unit_1000_eur_per_ha")
    motti_plus_hmodel_data_points_spruce_blv_omt = np.loadtxt(parameters.simulation_results_root + "/omt/motti_plus_hmodel_omt_spruce_tsum_frot_and_blv_at_interest_rate_of_" + str(parameters.blv_interest_rate) + "_percent.txt.tsum_unit_1000_degree_days_blv_unit_1000_eur_per_ha")

    motti_plus_hmodel_data_points_spruce_birch_blv_mt = np.loadtxt(parameters.simulation_results_root + "/mt/motti_plus_hmodel_mt_spruce_birch_tsum_frot_and_blv_at_interest_rate_of_" + str(parameters.blv_interest_rate) + "_percent.txt.tsum_unit_1000_degree_days_blv_unit_1000_eur_per_ha")
    motti_plus_hmodel_data_points_spruce_birch_blv_omt = np.loadtxt(parameters.simulation_results_root + "/omt/motti_plus_hmodel_omt_spruce_birch_tsum_frot_and_blv_at_interest_rate_of_" + str(parameters.blv_interest_rate) + "_percent.txt.tsum_unit_1000_degree_days_blv_unit_1000_eur_per_ha")

    motti_plus_hmodel_data_points_spruce_carbon_mt = np.loadtxt(parameters.simulation_results_root + "/mt/motti_plus_hmodel_mt_spruce_tsum_frot_and_carbon.txt.tsum_unit_1000_degree_days_carbon_unit_100_tCO2_per_ha")
    motti_plus_hmodel_data_points_spruce_carbon_omt = np.loadtxt(parameters.simulation_results_root + "/omt/motti_plus_hmodel_omt_spruce_tsum_frot_and_carbon.txt.tsum_unit_1000_degree_days_carbon_unit_100_tCO2_per_ha")

    motti_plus_hmodel_data_points_spruce_birch_carbon_mt = np.loadtxt(parameters.simulation_results_root + "/mt/motti_plus_hmodel_mt_spruce_birch_tsum_frot_and_carbon.txt.tsum_unit_1000_degree_days_carbon_unit_100_tCO2_per_ha")
    motti_plus_hmodel_data_points_spruce_birch_carbon_omt = np.loadtxt(parameters.simulation_results_root + "/omt/motti_plus_hmodel_omt_spruce_birch_tsum_frot_and_carbon.txt.tsum_unit_1000_degree_days_carbon_unit_100_tCO2_per_ha")


    return  pure_motti_data_points_birch_blv_mt, pure_motti_data_points_birch_blv_omt, pure_motti_data_points_pine_blv_mt, pure_motti_data_points_pine_blv_omt, pure_motti_data_points_birch_carbon_mt, pure_motti_data_points_birch_carbon_omt, pure_motti_data_points_pine_carbon_mt, pure_motti_data_points_pine_carbon_omt, motti_plus_hmodel_data_points_spruce_blv_mt, motti_plus_hmodel_data_points_spruce_blv_omt, motti_plus_hmodel_data_points_spruce_birch_blv_mt, motti_plus_hmodel_data_points_spruce_birch_blv_omt, motti_plus_hmodel_data_points_spruce_carbon_mt, motti_plus_hmodel_data_points_spruce_carbon_omt, motti_plus_hmodel_data_points_spruce_birch_carbon_mt, motti_plus_hmodel_data_points_spruce_birch_carbon_omt



#
# Read in the models for BLV (in units of kEUR/ha) and carbon (in
# units of 100 tCO2/ha) as a function of temperature sum (in units of
# 1000 d.d) and rot fraction.
#
# The format for the 1D models must be the following:
#
# <Tsum_1>   <BLV or carbon 1>
# <Tsum_2>   <BLV or carbon 2>
#   ...            ...
#
# The format for the 2D models must be:
#
# <Tsum>   <f_rot>   <BLV or carbon>
#   ...      ...          ...
#
# where the three vectors (Tsum, f_rot, BLV or carbon) are each
# "flattened" from a meshgrid representation. See the code for
# creating the final models for details.
#

def read_in_models():


    #
    # Pure motti
    #

    birch_blv_vs_tsum_mt = np.loadtxt(parameters.blv_and_carbon_models_root + "/mt/blv/interest_rate_" + str(parameters.blv_interest_rate) + "_percent/birch/final_model/loess_fit.txt")
    birch_blv_vs_tsum_omt = np.loadtxt(parameters.blv_and_carbon_models_root + "/omt/blv/interest_rate_" + str(parameters.blv_interest_rate) + "_percent/birch/final_model/loess_fit.txt")

    pine_blv_vs_tsum_mt = np.loadtxt(parameters.blv_and_carbon_models_root + "/mt/blv/interest_rate_" + str(parameters.blv_interest_rate) + "_percent/pine/final_model/loess_fit.txt")
    pine_blv_vs_tsum_omt = np.loadtxt(parameters.blv_and_carbon_models_root + "/omt/blv/interest_rate_" + str(parameters.blv_interest_rate) + "_percent/pine/final_model/loess_fit.txt")

    birch_carbon_vs_tsum_mt = np.loadtxt(parameters.blv_and_carbon_models_root + "/mt/carbon/birch/final_model/loess_fit.txt")
    birch_carbon_vs_tsum_omt = np.loadtxt(parameters.blv_and_carbon_models_root + "/omt/carbon/birch/final_model/loess_fit.txt")

    pine_carbon_vs_tsum_mt = np.loadtxt(parameters.blv_and_carbon_models_root + "/mt/carbon/pine/final_model/loess_fit.txt")
    pine_carbon_vs_tsum_omt = np.loadtxt(parameters.blv_and_carbon_models_root + "/omt/carbon/pine/final_model/loess_fit.txt")



    #
    # Motti+Hmodel
    #

    spruce_blv_vs_tsum_and_frot_mt = np.loadtxt(parameters.blv_and_carbon_models_root + "/mt/blv/interest_rate_" + str(parameters.blv_interest_rate) + "_percent/spruce/final_model/loess_fit.txt")
    spruce_blv_vs_tsum_and_frot_omt = np.loadtxt(parameters.blv_and_carbon_models_root + "/omt/blv/interest_rate_" + str(parameters.blv_interest_rate) + "_percent/spruce/final_model/loess_fit.txt")

    spruce_birch_blv_vs_tsum_and_frot_mt = np.loadtxt(parameters.blv_and_carbon_models_root + "/mt/blv/interest_rate_" + str(parameters.blv_interest_rate) + "_percent/spruce_birch/final_model/loess_fit.txt")
    spruce_birch_blv_vs_tsum_and_frot_omt = np.loadtxt(parameters.blv_and_carbon_models_root + "/omt/blv/interest_rate_" + str(parameters.blv_interest_rate) + "_percent/spruce_birch/final_model/loess_fit.txt")

    spruce_carbon_vs_tsum_and_frot_mt = np.loadtxt(parameters.blv_and_carbon_models_root + "/mt/carbon/spruce/final_model/loess_fit.txt")
    spruce_carbon_vs_tsum_and_frot_omt = np.loadtxt(parameters.blv_and_carbon_models_root + "/omt/carbon/spruce/final_model/loess_fit.txt")

    spruce_birch_carbon_vs_tsum_and_frot_mt = np.loadtxt(parameters.blv_and_carbon_models_root + "/mt/carbon/spruce_birch/final_model/loess_fit.txt")
    spruce_birch_carbon_vs_tsum_and_frot_omt = np.loadtxt(parameters.blv_and_carbon_models_root + "/omt/carbon/spruce_birch/final_model/loess_fit.txt")


    return birch_blv_vs_tsum_mt, birch_blv_vs_tsum_omt, pine_blv_vs_tsum_mt, pine_blv_vs_tsum_omt, birch_carbon_vs_tsum_mt, birch_carbon_vs_tsum_omt, pine_carbon_vs_tsum_mt, pine_carbon_vs_tsum_omt, spruce_blv_vs_tsum_and_frot_mt, spruce_blv_vs_tsum_and_frot_omt, spruce_birch_blv_vs_tsum_and_frot_mt, spruce_birch_blv_vs_tsum_and_frot_omt, spruce_carbon_vs_tsum_and_frot_mt, spruce_carbon_vs_tsum_and_frot_omt, spruce_birch_carbon_vs_tsum_and_frot_mt, spruce_birch_carbon_vs_tsum_and_frot_omt


    
#
# Plot a given 1D regression function function along with the corresponding
# raw simulation data points. Save the plot into a file.
#

def visualize_1D_regression_function(xy_tabulated_data, raw_data_points, color, xlabel, ylabel, title, filename):
    
    
    fig, ax = plt.subplots(figsize = parameters.figure_size)

    h_regression_function, = ax.plot(xy_tabulated_data[:, 0], xy_tabulated_data[:, 1], '-', color = color, linewidth = parameters.linewidth_for_1D_regression_function_plots)

    h_raw_data, = ax.plot(raw_data_points[:, 0], raw_data_points[:, 1], 'ko', markersize = parameters.markersize_for_1D_plot_data_points, markeredgewidth = parameters.markeredgewidth_for_1D_plot_data_points)

    if not (parameters.ymin_buffer is None or parameters.ymax_buffer is None):
    
        ax.set_ylim(np.min(raw_data_points[:, 1]) - parameters.ymin_buffer, np.max(raw_data_points[:, 1]) + parameters.ymax_buffer)

    ax.set_xlabel(xlabel, labelpad = parameters.labelpad_1D_regression_function)

    ax.set_ylabel(ylabel, labelpad = parameters.labelpad_1D_regression_function)

    ax.tick_params(axis = 'x', which = 'major', pad = parameters.tickpad_1D_regression_function)

    ax.tick_params(axis = 'y', which = 'major', pad = parameters.tickpad_1D_regression_function)
    
    ax.legend([h_raw_data, h_regression_function], ['Simulation data', 'Fit to data'])
    
    ax.set_title(title)

    fig.savefig(filename)

    plt.close(fig)
    


#
# Plot a given 2D regression function function along with the corresponding
# simulation data points. Save the plot into a file.
#

def visualize_2D_regression_function(xyz_tabulated_data, simulation_data_points, xlabel, ylabel, zlabel, title, filename_basename):


    fig = plt.figure(figsize = parameters.figure_size)

    ax = fig.add_subplot(projection = '3d', proj_type = 'ortho')

    
    #
    # First, plot the simulation data
    #

    h_simulation_data_points = ax.scatter(simulation_data_points[:, 0], simulation_data_points[:, 1], simulation_data_points[:, 2], c = 'k', marker = 'x', s = parameters.markersize_simulation_data_points_in_2D_plot, linewidth = parameters.linewidth_simulation_data_points_in_2D_plot, alpha = 1.0)



    #
    # Then, visualize the 3D surface. To do this, set the data onto a
    # mesh grid. Reconstruct the mesh grid from the "flattened"
    # (x,y,z) data format in which the model was imported.
    #


    #
    # Find the length of a single row in the x meshgrid
    #

    previous_x = -np.Inf

    i_x = 0
    
    while xyz_tabulated_data[i_x, 0] > previous_x:
        
        previous_x = xyz_tabulated_data[i_x, 0]
        
        i_x = i_x + 1
        
    length_of_row_in_x = i_x
        
        
    #
    # Find the number of rows in the x meshgrid
    #

    number_of_rows_in_x = int(xyz_tabulated_data.shape[0] / length_of_row_in_x)

    
    #
    # Re-create the x and y meshgrids, and create the corresponding z
    # matrix.
    # 
    
    x_meshgrid = np.zeros((number_of_rows_in_x, length_of_row_in_x))

    y_meshgrid = np.zeros((number_of_rows_in_x, length_of_row_in_x))
    
    z_to_visualize = np.zeros(x_meshgrid.shape)

    
    for i_row in np.arange(0, number_of_rows_in_x):

        x_meshgrid[i_row, :] = xyz_tabulated_data[ i_row*length_of_row_in_x : (i_row + 1)*length_of_row_in_x, 0 ]

        y_meshgrid[i_row, :] = xyz_tabulated_data[ i_row*length_of_row_in_x : (i_row + 1)*length_of_row_in_x, 1 ]
        
        z_to_visualize[i_row, :] = xyz_tabulated_data[ i_row*length_of_row_in_x : (i_row + 1)*length_of_row_in_x, 2 ]

        
    surf = ax.plot_surface(x_meshgrid, y_meshgrid, z_to_visualize, cmap = 'rainbow', rstride = 1, cstride = 1, linewidth = 0, antialiased = False, alpha = parameters.alpha_3D_surface)

    if not (parameters.zmin_buffer is None or parameters.zmax_buffer is None):
    
        ax.set_zlim([np.min(simulation_data_points[:, 2]) - parameters.zmin_buffer, np.max(simulation_data_points[:, 2]) + parameters.zmax_buffer])

    ax.grid(False)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    ax.zaxis.line.set_lw(0.)
    ax.set_zticks([])

    cbar = fig.colorbar(surf, shrink = 0.40, aspect = 8, pad = 0.1)
    cbar.set_label(zlabel, labelpad = parameters.labelpad, rotation = 90.0)

    ax.set_xlabel(xlabel, labelpad = parameters.labelpad_2D_regression_function)

    ax.set_ylabel(ylabel, labelpad = parameters.labelpad_2D_regression_function)

    ax.tick_params(axis = 'x', which = 'major', pad = parameters.tickpad_2D_regression_function)

    ax.tick_params(axis = 'y', which = 'major', pad = parameters.tickpad_2D_regression_function)
    
    ax.legend([h_simulation_data_points], ['Simulation data'])

    ax.set_title(title, pad = parameters.titlepad)

    ax.view_init(elev = 20.0, azim = 45.0)
    
    fig.savefig(filename_basename + '_sideview.png')

    ax.view_init(elev = 90.0, azim = 0.0)
    
    fig.savefig(filename_basename + '_topview.png')
    
    plt.close(fig)



#
# Convert the given cabin positions latitude, longitude readings into
# easting, northing.
#

def convert_cabin_positions_from_latlon_to_easting_northing(cabin_position_latitude, cabin_position_longitude):

    
    easting_cabin_position = np.zeros(cabin_position_latitude.shape)

    northing_cabin_position = np.zeros(cabin_position_latitude.shape)
    
    
    for i in range(0, cabin_position_latitude.shape[0]):
    
        this_latitude = cabin_position_latitude[i]

        this_longitude = cabin_position_longitude[i]

        etrs_tm35fin_result = coordinates.WGS84lalo_to_ETRSTM35FINxy({'La': this_latitude, 'Lo': this_longitude})
    
        this_easting = etrs_tm35fin_result['E']

        this_northing = etrs_tm35fin_result['N']
    
        easting_cabin_position[i] = this_easting

        northing_cabin_position[i] = this_northing

        
    print("Converted a total of %d latitude, longitude pairs." % (i + 1))


    return easting_cabin_position, northing_cabin_position



#
# Read in a Finnish Forest Centre lattice data raster (here fertility
# class or soil type) and slice it to just encompass the stem
# positions of the stand being processed.
#

def read_in_ffc_raster(data_root, feature_name, stem_positions_min_easting, stem_positions_max_easting, stem_positions_min_northing, stem_positions_max_northing):

    
    #
    # Set the paths so that the lattice data for this province will be
    # loaded correctly
    #
    
    feature_data_file = data_root + "Hila_" + parameters.stand_province + ".gpkg." + feature_name + ".png"
    
    world_file = data_root + "Hila_" + parameters.stand_province + ".gpkg." + feature_name + ".wld"
    
    
    #
    # Set this parameter so that you can read in arbitrarily large .png
    # files
    #
    
    Image.MAX_IMAGE_PIXELS = None


    #
    # Read in the georeferenced PNG as a PIL image
    #

    print("")
    print("Now reading in the " + feature_name + " data from file %s..." % feature_data_file)

    feature_as_image = Image.open(feature_data_file)
    
    
    #
    # Print out some stats on the image. The mode "L" means 8-bit pixel
    # values and grayscale, i.e., each pixel has a value in the range
    # 0...255.
    #

    print("")
    print("Done. Here are some stats on the image:")
    print("")
    print("Format:", feature_as_image.format)
    print("Size (width, height):", feature_as_image.size)
    print("Mode:", feature_as_image.mode)
    print("Bands:", feature_as_image.getbands())
    print("Colors (count, pixel value):", feature_as_image.getcolors())

    

    #
    # Read in the world file
    #

    print("")
    print("Now reading in the world file %s for the " % world_file + feature_name + " data...")

    world_file_data = np.loadtxt(world_file)

    pixel_size_x = np.abs(world_file_data[0])

    pixel_size_y = np.abs(world_file_data[3])

    x_coordinate_of_ul_pixel_center = world_file_data[4]

    y_coordinate_of_ul_pixel_center = world_file_data[5]


    print("Done. Found the following parameter values:")
    print("")
    print("Pixel size in x-direction (m): %f" % pixel_size_x)
    print("Pixel size in y-direction (m): %f" % pixel_size_y)
    print("Upper left pixel center x-coordinate (easting): %f" % x_coordinate_of_ul_pixel_center)
    print("Upper left pixel center y-coordinate (northing): %f" % y_coordinate_of_ul_pixel_center)
    print("")

    

    #
    # Convert the image into a numpy array
    #

    feature_as_array = np.array(feature_as_image)

    feature_as_array_y_shape = feature_as_array.shape[0]

    feature_as_array_x_shape = feature_as_array.shape[1]


    print("Converted the " + feature_name + " image data into a numpy array. Here are some stats on the array:")
    print("")
    print("Shape:", feature_as_array.shape)
    print("Minimum value:", np.min(feature_as_array))
    print("Maximum value:", np.max(feature_as_array))
    print("Mean value for cells with pixel value greater than zero:", np.mean(feature_as_array[feature_as_array > 0]))
    print("Standard deviation of values for cells with pixel value greater than zero:", np.std(feature_as_array[feature_as_array > 0]))
    

    #
    # Get a histogram of pixel values for the array
    #

    print("")
    print("Here is a histogram of pixel values of the array (value, count):")
    print("")
    
    values, counts = np.unique(feature_as_array, return_counts = True)
    
    for i in np.arange(0, len(values)):

        print("(%d, %d)" % (values[i], counts[i]))

        
    #
    # Get the easting-northing limits of the given data in terms of pixel centers
    #

    feature_data_min_easting = x_coordinate_of_ul_pixel_center

    feature_data_max_easting = x_coordinate_of_ul_pixel_center + pixel_size_x*(feature_as_array_x_shape - 1)

    feature_data_max_northing = y_coordinate_of_ul_pixel_center

    feature_data_min_northing = y_coordinate_of_ul_pixel_center - pixel_size_y*(feature_as_array_y_shape - 1)

    print("")
    print("The full " + feature_name + " data pixel centers run from easting %f m to %f m and northing %f m to %f m" % (feature_data_min_easting, feature_data_max_easting, feature_data_min_northing, feature_data_max_northing))

    

    #
    # Slice out the data for the stand we are processing. Before
    # slicing, make sure the data encompasses the stem positions for
    # this stand. Just use the raster pixel centers for this check.
    #
    
    if stem_positions_max_easting > feature_data_max_easting or stem_positions_min_easting < feature_data_min_easting or stem_positions_max_northing > feature_data_max_northing or stem_positions_min_northing < feature_data_min_northing:

        print("")
        print("ERROR! Stem positional data reaches outside of the " + feature_name + " data spatial range. Exiting.")
        exit(1)

    else:

        print("")
        print("The given " + feature_name + " data encompasses the stem positions.")


        
    #
    # Compute the slicing indeces. Do the slicing so that you include
    # the previous / next pixel as the lower / upper limit in each
    # dimension.
    #

    i_x_max_slicing = (np.floor((stem_positions_max_easting - feature_data_min_easting) / pixel_size_x) + 1).astype(int)

    i_x_min_slicing = (np.floor((stem_positions_min_easting - feature_data_min_easting) / pixel_size_x)).astype(int)

    i_y_max_slicing = (np.floor(np.abs(stem_positions_min_northing - feature_data_max_northing) / pixel_size_y) + 1).astype(int)

    i_y_min_slicing = (np.floor(np.abs(stem_positions_max_northing - feature_data_max_northing) / pixel_size_y)).astype(int)

    print("")
    print("The slicing indeces for the " + feature_name + " data run from i_x = %d to %d, i_y = %d to %d" % (i_x_min_slicing, i_x_max_slicing, i_y_min_slicing, i_y_max_slicing))
    print("")


    #
    # Slice the data
    #

    sliced_feature_as_array = feature_as_array[i_y_min_slicing: i_y_max_slicing + 1, i_x_min_slicing: i_x_max_slicing + 1]

    sliced_feature_as_array_y_shape = feature_as_array.shape[0]

    sliced_feature_as_array_x_shape = feature_as_array.shape[1]

    print("Sliced the " + feature_name + " data to a tight fit of the stem positional data. Here are some stats:")
    print("")
    print("Shape:", sliced_feature_as_array.shape)
    print("Minimum value:", np.min(sliced_feature_as_array))
    print("Maximum value:", np.max(sliced_feature_as_array))
    print("Mean value:", np.mean(sliced_feature_as_array))
    print("Standard deviation:", np.std(sliced_feature_as_array))


    #
    # Get a histogram of pixel values for the array
    #

    print("")
    print("Here is a histogram of pixel values of the sliced array (value, count):")
    print("")
    
    values, counts = np.unique(sliced_feature_as_array, return_counts = True)
    
    for i in np.arange(0, len(values)):

        print("(%d, %d)" % (values[i], counts[i]))

    
    #
    # Get the minimum and maximum coordinate of the sliced data in
    # both dimensions in terms of pixel centers
    #

    sliced_feature_data_min_easting = feature_data_min_easting + pixel_size_x*i_x_min_slicing

    sliced_feature_data_max_easting = feature_data_min_easting + pixel_size_x*i_x_max_slicing

    sliced_feature_data_min_northing = feature_data_max_northing - pixel_size_y*i_y_max_slicing

    sliced_feature_data_max_northing = feature_data_max_northing - pixel_size_y*i_y_min_slicing

    print("")
    print("The sliced " + feature_name + " data pixel centers run from easting %f m to %f m and northing %f m to %f m" % (sliced_feature_data_min_easting, sliced_feature_data_max_easting, sliced_feature_data_min_northing, sliced_feature_data_max_northing))


    
    #
    # Return what is needed later
    #
    
    return sliced_feature_as_array, sliced_feature_data_min_easting, sliced_feature_data_max_easting, sliced_feature_data_min_northing, sliced_feature_data_max_northing, pixel_size_x, pixel_size_y



#
# Find the temperature sum for this stand (in units of d.d.). Do this
# by finding the closest (easting, northing) point to the stand in the
# pre-created temperature sum data.
#

def find_temperature_sum_for_stand(easting_crane_tip, northing_crane_tip):


    #
    # Read in the pre-created temperature sum data of the format
    #
    # <easting (m)> <northing (m)> <temperature sum (d.d.)>
    #

    tsum_data = np.loadtxt(parameters.temperature_sum_data_file)
    
    
    
    #
    # Find the mean position of the stems in the stand
    #

    mean_position_of_stems_in_stand = np.array([np.mean(easting_crane_tip), np.mean(northing_crane_tip)])


    print("")
    print("Mean position of stems in the stand is (%f, %f) m" % (mean_position_of_stems_in_stand[0], mean_position_of_stems_in_stand[1]))



    #
    # Find the closest (easting, northing) point in the temperature
    # sum data to the mean position of the stems
    #
    
    distances_from_center_of_stand_to_tsum_data_points = np.linalg.norm(mean_position_of_stems_in_stand - tsum_data[:, 0:2], axis = 1)

    index_of_closest_point = np.argmin(distances_from_center_of_stand_to_tsum_data_points)
        
    easting_of_the_closest_point = tsum_data[index_of_closest_point, 0]
    
    northing_of_the_closest_point = tsum_data[index_of_closest_point, 1]
    
    tsum_of_the_closest_point = tsum_data[index_of_closest_point, 2]

    
    print("")
    print("The closest point in the temperature sum data to the mean position of stems in the stand was (%f, %f) m, which has a temperature sum of %f d.d." % (easting_of_the_closest_point, northing_of_the_closest_point, tsum_of_the_closest_point))

    
    
    return tsum_of_the_closest_point



#
# Segment the harvest area into rotten and healthy microstands using
# alpha shapes and standard operations on geometric objects in the
# plane, considering clusters of stems as found using DBSCAN.
#

def delineate_microstands_using_dbscan(position_and_cluster_id_data_for_each_cluster_including_outliers, species_group_id, theoretical_sawlog_volume, rotten_200_with_sawlog, easting_crane_tip, northing_crane_tip):
    
    
    #
    # First, for each rot cluster, including the set of outliers,
    # create a list of the rotten spruce stem positions, for
    # delineation via alpha complexes. Combine this with the cluster
    # ID of the cluster to create a list with each item a tuple of the
    # format
    #
    # (list of stem position tuples, cluster ID)
    #
    
    print("")
    print("Now breaking the stand down into microstands.")

    
    position_and_cluster_id_data_for_delineation = []
    
    for this_cluster in position_and_cluster_id_data_for_each_cluster_including_outliers:

        
        this_cluster_easting_values = this_cluster[:, 0]

        this_cluster_northing_values = this_cluster[:, 1]

        this_cluster_id = this_cluster[0, 2].astype(int)

        
        this_cluster_rotten_spruce_stem_positions = []

        
        for i in np.arange(0, this_cluster_easting_values.shape[0]):

            this_easting = this_cluster_easting_values[i]

            this_northing = this_cluster_northing_values[i]

            this_cluster_rotten_spruce_stem_positions.append((this_easting, this_northing))

            
        position_and_cluster_id_data_for_delineation.append((this_cluster_rotten_spruce_stem_positions, this_cluster_id))


    
    #
    # Then, delineate each rot cluster using an alpha complex. Also,
    # create a buffer zone for each delineation. Store the delineating
    # shapes, each of which will be a shapely Polygon, into a list
    # with each item a tuple of the format
    #
    # (delineation shape, cluster ID)
    #

    
    print("")
    print("Now creating delineations for the rot clusters.")

    
    cluster_delineations = []

    
    for this_cluster in position_and_cluster_id_data_for_delineation:

        
        this_cluster_position_tuples = this_cluster[0]

        this_cluster_id = this_cluster[1]

        
        #
        # Skip the individual rotten stems, these will be handled
        # separately later on
        #

        if this_cluster_id == -1:

            continue

        
        #
        # Find the optimal alpha parameter for producing the alpha
        # complex for this cluster, i.e., the value for alpha which
        # wraps the points as tightly as possible without losing any
        # points (which will happen when alpha grows sufficiently
        # large).
        #

        this_optimal_alpha_parameter = alphashape.optimizealpha(this_cluster_position_tuples, max_iterations = parameters.max_iterations_for_alpha_optimization)
        
        print("---> The optimal alpha parameter for cluster of ID %d was %f" % (this_cluster_id, this_optimal_alpha_parameter))

        if this_optimal_alpha_parameter == 0.0:

            print("*** NB! This means that the convex hull will be created for this cluster.")
        
        
        #
        # Get the alpha complex using the optimized value for the
        # alpha parameter. The returned alphashape will be either a
        # shapely Polygon, LineString, or Point, depending on the
        # number of points being delineated.
        #
        
        this_delineation = alphashape.alphashape(this_cluster_position_tuples, alpha = this_optimal_alpha_parameter)

    
        #
        # Buffer the alpha complex delineation with a "safety buffer"
        # of the desired size. This line should work in the desired
        # way, regardless of whether the delineation being buffered is
        # a Polygon, LineString, or Point. After the buffering, the
        # delineation will be a Polygon.
        #
    
        this_delineation = this_delineation.buffer(parameters.rot_area_buffer_size, resolution = parameters.rot_area_buffer_resolution, cap_style = parameters.rot_area_buffer_cap_style, join_style = parameters.rot_area_buffer_join_style, single_sided = False)


        #
        # Print out the area and rot fraction of this delineation
        #

        print("---> The area of this delineation is %f m**2 or %f ha" % (this_delineation.area, this_delineation.area / 1e4))
        print("---> The rot fraction of this delineation is %f" % (get_rot_fraction_within_given_delineation(this_delineation, species_group_id, theoretical_sawlog_volume, rotten_200_with_sawlog, easting_crane_tip, northing_crane_tip)))
    
        #
        # Save this delineation in the list described above
        #
    
        cluster_delineations.append((this_delineation, this_cluster_id))



    
    #
    # Set a buffer zone around each rotten outlier spruce stem. Each
    # resulting delineation will be a shapely Polygon. Store these
    # into a list with each item a tuple of the format
    #
    # (delineation shape, cluster ID (here equal to -1))
    #


    individual_stem_delineations = []


    for this_cluster in position_and_cluster_id_data_for_delineation:


        this_cluster_position_tuples = this_cluster[0]

        this_cluster_id = this_cluster[1]


        if this_cluster_id == -1:

        
            print("")
            print("Now producing delineations for the individual rotten stems outside of the clusters.")


            #
            # Loop over the individual rotten stems, creating a
            # circular "safety buffer" delineation for each. After the
            # buffering, each delineation will be a shapely Polygon.
            #

            for i_stem in np.arange(0, len(this_cluster_position_tuples)):


                this_easting = this_cluster_position_tuples[i_stem][0]

                this_northing = this_cluster_position_tuples[i_stem][1]

                this_delineation = Point(this_easting, this_northing).buffer(parameters.rot_area_buffer_size, resolution = parameters.rot_area_buffer_resolution, cap_style = parameters.rot_area_buffer_cap_style, join_style = parameters.rot_area_buffer_join_style, single_sided = False)

                
                #
                # Print out the area and rot fraction of this delineation
                #

                print("---> The area of this delineation is %f m**2 or %f ha" % (this_delineation.area, this_delineation.area / 1e4))
                print("---> The rot fraction of this delineation is %f" % (get_rot_fraction_within_given_delineation(this_delineation, species_group_id, theoretical_sawlog_volume, rotten_200_with_sawlog, easting_crane_tip, northing_crane_tip)))

            
                individual_stem_delineations.append((this_delineation, this_cluster_id))
            

            break



    #
    # Then, delineate the full harvest area
    #

    print("")
    print("Now delineating the full harvest area, i.e., the set of all stem positions.")
    
    
    #
    # First, save the position tuples of all stems into a list
    #

    position_data_for_full_harvest_area_delineation = []

    
    for i in np.arange(0, easting_crane_tip.shape[0]):

        this_easting = easting_crane_tip[i]

        this_northing = northing_crane_tip[i]
    
        position_data_for_full_harvest_area_delineation.append((this_easting, this_northing))

    
    #
    # Then, delineate the full harvest area. As with the rot clusters,
    # find the optimal value for the parameter alpha first, and then
    # use this to create the alpha complex. The delineation will be
    # either a shapely Polygon, LineString, or Point, depending on the
    # number of points being delineated.
    #

    this_optimal_alpha_parameter = alphashape.optimizealpha(position_data_for_full_harvest_area_delineation, max_iterations = parameters.max_iterations_for_alpha_optimization)

    print("---> The optimal alpha parameter for the full harvest area delineation was %f" % this_optimal_alpha_parameter)

    if this_optimal_alpha_parameter == 0.0:

        print("*** NB! This means that the convex hull will be created for the full harvest area.")

        
    full_harvest_area_delineation = alphashape.alphashape(position_data_for_full_harvest_area_delineation, alpha = this_optimal_alpha_parameter)

    #
    # Buffer the alpha complex delineation. After the buffering, the
    # delineation will be a shapely Polygon.
    #
   
    full_harvest_area_delineation = full_harvest_area_delineation.buffer(parameters.full_harvest_area_buffer_size, resolution = parameters.full_harvest_area_buffer_resolution, cap_style = parameters.full_harvest_area_buffer_cap_style, join_style = parameters.full_harvest_area_buffer_join_style, single_sided = False)


    #
    # Print out the area and rot fraction of this delineation
    #

    print("---> The area of the full harvest area delineation is %f m**2 or %f ha" % (full_harvest_area_delineation.area, full_harvest_area_delineation.area / 1e4))
    print("---> The rot fraction of the full harvest area delineation is %f" % (get_rot_fraction_within_given_delineation(full_harvest_area_delineation, species_group_id, theoretical_sawlog_volume, rotten_200_with_sawlog, easting_crane_tip, northing_crane_tip)))

    #
    # Get mean stump density as a sanity check
    #

    print("---> The mean stump density is %f stumps per ha" % ( easting_crane_tip.shape[0] / (full_harvest_area_delineation.area / 1e4) ))


    
    #
    # Then, delineate the total rotten area in this harvest area,
    # i.e., get the union of the cluster delineations and the
    # individual stem delineations
    #

    print("")
    print("Now delineating the total rotten area in this harvest area.")

    
    #
    # Create a list of all the delineating Polygons of the rot
    # clusters and the individual rotten stems
    #

    list_of_rot_area_delineations = [this_cluster[0] for this_cluster in cluster_delineations] + [this_individual_stem[0] for this_individual_stem in individual_stem_delineations]

    
    #
    # Then, get their union. The result will be a Polygon or
    # MultiPolygon.
    #
    
    total_rot_area_delineation = unary_union(list_of_rot_area_delineations)


    #
    # Cut out parts of the total rot area delineation that are outside
    # of the full harvest area delineation
    #

    total_rot_area_delineation = total_rot_area_delineation.intersection(full_harvest_area_delineation)
    

    #
    # Print out the area and rot fraction of this delineation
    #

    print("---> The area of the total rotten area delineation is %f m**2 or %f ha" % (total_rot_area_delineation.area, total_rot_area_delineation.area / 1e4))
    print("---> The rot fraction of the total rotten area delineation is %f" % (get_rot_fraction_within_given_delineation(total_rot_area_delineation.buffer(0), species_group_id, theoretical_sawlog_volume, rotten_200_with_sawlog, easting_crane_tip, northing_crane_tip)))
    
    
    
    #
    # Then, delineate the total healthy area in this harvest area,
    # i.e., the harvest area outside of the total rotten area
    #

    print("")
    print("Now delineating the total healthy area in this harvest area.")


    #
    # Do the delineation. The result will be a Polygon or MultiPolygon.
    #
    
    total_healthy_area_delineation = full_harvest_area_delineation.difference(total_rot_area_delineation)

    
    #
    # Print out the area and rot fraction of this delineation
    #

    print("---> The area of the total healthy area delineation is %f m**2 or %f ha" % (total_healthy_area_delineation.area, total_healthy_area_delineation.area / 1e4))
    print("---> The rot fraction of the total healthy area delineation is %f" % (get_rot_fraction_within_given_delineation(total_healthy_area_delineation.buffer(0), species_group_id, theoretical_sawlog_volume, rotten_200_with_sawlog, easting_crane_tip, northing_crane_tip)))


    #
    # Return what is needed later on for performing the optimization and creating plots
    #
    
    return cluster_delineations, individual_stem_delineations, full_harvest_area_delineation, total_rot_area_delineation, total_healthy_area_delineation



#
# Segment the stand into microstands using the pre-computed
# delineations for rotten and healthy areas as found in the external
# .geojson file.
#

def delineate_microstands_using_external_geojson(species_group_id, theoretical_sawlog_volume, rotten_200_with_sawlog, easting_crane_tip, northing_crane_tip):

    
    #
    # First, delineate the full harvest area
    #

    print("")
    print("Now delineating the full harvest area, i.e., the set of all stem positions.")
    
    
    #
    # First, save the position tuples of all stems into a list
    #

    position_data_for_full_harvest_area_delineation = []

    
    for i in np.arange(0, easting_crane_tip.shape[0]):

        this_easting = easting_crane_tip[i]

        this_northing = northing_crane_tip[i]
    
        position_data_for_full_harvest_area_delineation.append((this_easting, this_northing))

    
    #
    # Then, delineate the full harvest area. Find the optimal value
    # for the parameter alpha first, and then use this to create the
    # alpha complex. The delineation will be either a shapely Polygon,
    # LineString, or Point, depending on the number of points being
    # delineated.
    #

    this_optimal_alpha_parameter = alphashape.optimizealpha(position_data_for_full_harvest_area_delineation, max_iterations = parameters.max_iterations_for_alpha_optimization)

    print("---> The optimal alpha parameter for the full harvest area delineation was %f" % this_optimal_alpha_parameter)

    if this_optimal_alpha_parameter == 0.0:

        print("*** NB! This means that the convex hull will be created for the full harvest area.")

        
    full_harvest_area_delineation = alphashape.alphashape(position_data_for_full_harvest_area_delineation, alpha = this_optimal_alpha_parameter)

    #
    # Buffer the alpha complex delineation. After the buffering, the
    # delineation will be a shapely Polygon.
    #
   
    full_harvest_area_delineation = full_harvest_area_delineation.buffer(parameters.full_harvest_area_buffer_size, resolution = parameters.full_harvest_area_buffer_resolution, cap_style = parameters.full_harvest_area_buffer_cap_style, join_style = parameters.full_harvest_area_buffer_join_style, single_sided = False)


    #
    # Print out the area and rot fraction of this delineation
    #

    print("---> The area of the full harvest area delineation is %f m**2 or %f ha" % (full_harvest_area_delineation.area, full_harvest_area_delineation.area / 1e4))
    print("---> The rot fraction of the full harvest area delineation is %f" % (get_rot_fraction_within_given_delineation(full_harvest_area_delineation, species_group_id, theoretical_sawlog_volume, rotten_200_with_sawlog, easting_crane_tip, northing_crane_tip)))
    
    #
    # Get mean stump density as a sanity check
    #

    print("---> The mean stump density is %f stumps per ha" % ( easting_crane_tip.shape[0] / (full_harvest_area_delineation.area / 1e4) ))
    
    
    
    #
    # Then, delineate the total rotten area in this harvest area,
    # i.e., get the union of the rotten, i.e., unsafe areas
    #

    print("")
    print("Now delineating the total rotten area in this harvest area.")

    
    #
    # Read in the rotten area delineations from the GeoJSON file into
    # Shapely Polygons and MultiPolygons.
    #

    
    print("")
    print("Now reading in the GeoJSON file %s..." % parameters.external_geojson_file_for_delineations)

    f = open(parameters.external_geojson_file_for_delineations, 'r')

    delineations_as_json = json.load(f)

    f.close()
        
    print("Done.")
    

    print("")
    print("Now finding the rotten area delineations, i.e., the Polygons and MultiPolygons in this FeatureCollection that delineate the areas unsafe for planting spruce.")

    
    list_of_rot_area_delineations = []

    
    #
    # Loop over the Feature objects, i.e., the delineations in the
    # FeatureCollection
    #
    
    
    for this_feature_object in delineations_as_json["features"]:


        #
        # Get the value of the property "lyr.1" for this object, which
        # tells us whether this delineation shape describes a safe or
        # unsafe area
        #
            
        this_lyr_one = this_feature_object["properties"]["lyr.1"]

        print("")
        print("The value of lyr.1 is %d" % this_lyr_one)


        #
        # Get the corresponding geometry object. Use .buffer(0) to
        # fix cases where Polygons have overlapping rings.
        #
            
        this_geometry_object = shape(this_feature_object["geometry"]).buffer(0)
    
        print("The type of this geometry object is", type(this_geometry_object))


        #
        # See whether this object is a rotten area delineation or not
        #
            
        if this_lyr_one == 0:

                
            print("This object is a rotten area delineation. Now adding this geometry object to the list of rotten area delineations.")

            list_of_rot_area_delineations.append(this_geometry_object)

                
        elif this_lyr_one == 1:

                
            print("This object is not a rotten area delineation.")

                
        else:

                
            print("ERROR! Unexpected value of %s found for lyr.1. Exiting." % str(this_lyr_one))
            exit(1)


                
    print("")
    print("Done. Here's the list of rotten delineations:")
    print("")

    print(list_of_rot_area_delineations)


        
    #
    # Then, get the union of the rotten area delineations. The result
    # will be a Polygon or MultiPolygon.
    #
    
    total_rot_area_delineation = unary_union(list_of_rot_area_delineations)


    #
    # Cut out parts of the total rot area delineation that are outside
    # of the full harvest area delineation
    #

    total_rot_area_delineation = total_rot_area_delineation.intersection(full_harvest_area_delineation)
    

    #
    # Print out the area and rot fraction of this delineation
    #

    print("")
    print("---> The area of the total rotten area delineation is %f m**2 or %f ha" % (total_rot_area_delineation.area, total_rot_area_delineation.area / 1e4))
    print("---> The rot fraction of the total rotten area delineation is %f" % (get_rot_fraction_within_given_delineation(total_rot_area_delineation.buffer(0), species_group_id, theoretical_sawlog_volume, rotten_200_with_sawlog, easting_crane_tip, northing_crane_tip)))

    

    #
    # Then, delineate the total healthy area in this harvest area,
    # i.e., the harvest area outside of the total rotten area
    #

    print("")
    print("Now delineating the total healthy area in this harvest area.")


    #
    # Do the delineation. The result will be a Polygon or MultiPolygon.
    #
    
    total_healthy_area_delineation = full_harvest_area_delineation.difference(total_rot_area_delineation)

    
    #
    # Print out the area and rot fraction of this delineation
    #

    print("---> The area of the total healthy area delineation is %f m**2 or %f ha" % (total_healthy_area_delineation.area, total_healthy_area_delineation.area / 1e4))
    print("---> The rot fraction of the total healthy area delineation is %f" % (get_rot_fraction_within_given_delineation(total_healthy_area_delineation.buffer(0), species_group_id, theoretical_sawlog_volume, rotten_200_with_sawlog, easting_crane_tip, northing_crane_tip)))


    #
    # Return what is needed later on for performing the optimization and creating plots
    #
    
    return full_harvest_area_delineation, total_rot_area_delineation, total_healthy_area_delineation



#
# Find the ranking of the different regeneration alternatives,
# separately with respect to BLV and carbon.
#

def perform_optimization(species_group_id, theoretical_sawlog_volume, rotten_200_with_sawlog, easting_crane_tip, northing_crane_tip, birch_blv_vs_tsum, birch_carbon_vs_tsum, pine_blv_vs_tsum, pine_carbon_vs_tsum, spruce_blv_vs_tsum_and_frot, spruce_carbon_vs_tsum_and_frot, spruce_birch_blv_vs_tsum_and_frot, spruce_birch_carbon_vs_tsum_and_frot, temperature_sum_for_stand, full_harvest_area_delineation, total_rot_area_delineation, total_healthy_area_delineation):


    #
    # In the following, collect the metrics (BLV, carbon) for each
    # regeneration plan, i.e., each "alternative".
    #


    list_of_alternatives = ['Spruce for the entire stand', 'Spruce-birch for the entire stand', 'Birch for the entire stand', 'Pine for the entire stand', 'Pine for the rot areas, spruce outside of rot areas', 'Birch for the rot areas, spruce outside of rot areas', 'Pine for the rot areas, spruce-birch outside of rot areas', 'Birch for the rot areas, spruce-birch outside of rot areas', 'Pine for the rot areas, birch outside of rot areas', 'Birch for the rot areas, pine outside of rot areas']

    
    print("")
    print("Found the following alternatives to consider:")
    print("")

    for alternative in list_of_alternatives:

        print(alternative)

    print("")


    blv_eur_for_the_alternatives = []

    carbon_tCO2_for_the_alternatives = []



    #
    # Alternative 1: Spruce for the entire stand
    #
    
    area_of_stand_in_ha = full_harvest_area_delineation.area / 1e4

    rot_fraction_of_stand = get_rot_fraction_within_given_delineation(full_harvest_area_delineation, species_group_id, theoretical_sawlog_volume, rotten_200_with_sawlog, easting_crane_tip, northing_crane_tip)
    
    spruce_blv_eur_per_ha_for_stand = get_metric_for_given_tsum_and_frot(temperature_sum_for_stand / 1000.0, rot_fraction_of_stand, spruce_blv_vs_tsum_and_frot)*1000.0
    total_blv_eur_for_stand = spruce_blv_eur_per_ha_for_stand*area_of_stand_in_ha
    
    spruce_carbon_tCO2_per_ha_for_stand = get_metric_for_given_tsum_and_frot(temperature_sum_for_stand / 1000.0, rot_fraction_of_stand, spruce_carbon_vs_tsum_and_frot)*100.0
    total_carbon_tCO2_for_stand = spruce_carbon_tCO2_per_ha_for_stand*area_of_stand_in_ha
    
    print("")
    print("+++ Alternative 1: spruce for the entire stand +++")
    print("")
    print("Area of entire stand: %f ha" % area_of_stand_in_ha) 
    print("Rot fraction of entire stand: %f" % rot_fraction_of_stand)
    print("BLV in EUR/ha for spruce at this temperature sum and rot fraction: %f" % spruce_blv_eur_per_ha_for_stand)
    print("Carbon content in tCO2-equivalent/ha for spruce at this temperature sum and rot fraction: %f" % spruce_carbon_tCO2_per_ha_for_stand)
    print("")
    print("===> BLV for this alternative: %f EUR" % total_blv_eur_for_stand)
    print("===> Carbon content for this alternative: %f tCO2-equivalent" % total_carbon_tCO2_for_stand)
    print("")
    
    blv_eur_for_the_alternatives.append(float(total_blv_eur_for_stand))
    carbon_tCO2_for_the_alternatives.append(float(total_carbon_tCO2_for_stand))


    #
    # Alternative 2: Spruce-birch for the entire stand
    #
    
    area_of_stand_in_ha = full_harvest_area_delineation.area / 1e4
    
    rot_fraction_of_stand = get_rot_fraction_within_given_delineation(full_harvest_area_delineation, species_group_id, theoretical_sawlog_volume, rotten_200_with_sawlog, easting_crane_tip, northing_crane_tip)
    
    spruce_birch_blv_eur_per_ha_for_stand = get_metric_for_given_tsum_and_frot(temperature_sum_for_stand / 1000.0, rot_fraction_of_stand, spruce_birch_blv_vs_tsum_and_frot)*1000.0
    total_blv_eur_for_stand = spruce_birch_blv_eur_per_ha_for_stand*area_of_stand_in_ha
    
    spruce_birch_carbon_tCO2_per_ha_for_stand = get_metric_for_given_tsum_and_frot(temperature_sum_for_stand / 1000.0, rot_fraction_of_stand, spruce_birch_carbon_vs_tsum_and_frot)*100.0
    total_carbon_tCO2_for_stand = spruce_birch_carbon_tCO2_per_ha_for_stand*area_of_stand_in_ha
    
    print("")
    print("+++ Alternative 2: spruce-birch for the entire stand +++")
    print("")
    print("Area of entire stand: %f ha" % area_of_stand_in_ha) 
    print("Rot fraction of entire stand: %f" % rot_fraction_of_stand)
    print("BLV in EUR/ha for spruce-birch at this temperature sum and rot fraction: %f" % spruce_birch_blv_eur_per_ha_for_stand)
    print("Carbon content in tCO2-equivalent/ha for spruce-birch at this temperature sum and rot fraction: %f" % spruce_birch_carbon_tCO2_per_ha_for_stand)
    print("")
    print("===> BLV for this alternative: %f EUR" % total_blv_eur_for_stand)
    print("===> Carbon content for this alternative: %f tCO2-equivalent" % total_carbon_tCO2_for_stand)
    print("")
    
    blv_eur_for_the_alternatives.append(float(total_blv_eur_for_stand))
    carbon_tCO2_for_the_alternatives.append(float(total_carbon_tCO2_for_stand))



    #
    # Alternative 3: Birch for the entire stand
    #
    
    area_of_stand_in_ha = full_harvest_area_delineation.area / 1e4
    
    birch_blv_eur_per_ha_for_stand = get_metric_for_given_tsum(temperature_sum_for_stand / 1000.0, birch_blv_vs_tsum)*1000.0
    total_blv_eur_for_stand = birch_blv_eur_per_ha_for_stand*area_of_stand_in_ha
    
    birch_carbon_tCO2_per_ha_for_stand = get_metric_for_given_tsum(temperature_sum_for_stand / 1000.0, birch_carbon_vs_tsum)*100.0
    total_carbon_tCO2_for_stand = birch_carbon_tCO2_per_ha_for_stand*area_of_stand_in_ha
    
    print("")
    print("+++ Alternative 3: birch for the entire stand +++")
    print("")
    print("Area of entire stand: %f ha" % area_of_stand_in_ha) 
    print("BLV in EUR/ha for birch at this temperature sum and rot fraction: %f" % birch_blv_eur_per_ha_for_stand)
    print("Carbon content in tCO2-equivalent/ha for birch at this temperature sum and rot fraction: %f" % birch_carbon_tCO2_per_ha_for_stand)
    print("")
    print("===> BLV for this alternative: %f EUR" % total_blv_eur_for_stand)
    print("===> Carbon content for this alternative: %f tCO2-equivalent" % total_carbon_tCO2_for_stand)
    print("")
    
    blv_eur_for_the_alternatives.append(float(total_blv_eur_for_stand))
    carbon_tCO2_for_the_alternatives.append(float(total_carbon_tCO2_for_stand))



    #
    # Alternative 4: Pine for the entire stand
    #
    
    area_of_stand_in_ha = full_harvest_area_delineation.area / 1e4
    
    pine_blv_eur_per_ha_for_stand = get_metric_for_given_tsum(temperature_sum_for_stand / 1000.0, pine_blv_vs_tsum)*1000.0
    total_blv_eur_for_stand = pine_blv_eur_per_ha_for_stand*area_of_stand_in_ha
    
    pine_carbon_tCO2_per_ha_for_stand = get_metric_for_given_tsum(temperature_sum_for_stand / 1000.0, pine_carbon_vs_tsum)*100.0
    total_carbon_tCO2_for_stand = pine_carbon_tCO2_per_ha_for_stand*area_of_stand_in_ha
    
    print("")
    print("+++ Alternative 4: pine for the entire stand +++")
    print("")
    print("Area of entire stand: %f ha" % area_of_stand_in_ha) 
    print("BLV in EUR/ha for pine at this temperature sum and rot fraction: %f" % pine_blv_eur_per_ha_for_stand)
    print("Carbon content in tCO2-equivalent/ha for pine at this temperature sum and rot fraction: %f" % pine_carbon_tCO2_per_ha_for_stand)
    print("")
    print("===> BLV for this alternative: %f EUR" % total_blv_eur_for_stand)
    print("===> Carbon content for this alternative: %f tCO2-equivalent" % total_carbon_tCO2_for_stand)
    print("")
    
    blv_eur_for_the_alternatives.append(float(total_blv_eur_for_stand))
    carbon_tCO2_for_the_alternatives.append(float(total_carbon_tCO2_for_stand))


    
    #
    # Alternative 5: Pine for the rot areas, spruce outside of rot areas.
    #
    
    total_healthy_area_in_ha = total_healthy_area_delineation.area / 1e4
    
    total_rotten_area_in_ha = total_rot_area_delineation.area / 1e4
    
    pine_blv_eur_per_ha_for_rot_areas = get_metric_for_given_tsum(temperature_sum_for_stand / 1000.0, pine_blv_vs_tsum)*1000.0
    pine_total_blv_eur_for_rot_areas = pine_blv_eur_per_ha_for_rot_areas*total_rotten_area_in_ha
    
    pine_carbon_tCO2_per_ha_for_rot_areas = get_metric_for_given_tsum(temperature_sum_for_stand / 1000.0, pine_carbon_vs_tsum)*100.0
    pine_total_carbon_tCO2_for_rot_areas = pine_carbon_tCO2_per_ha_for_rot_areas*total_rotten_area_in_ha
    
    spruce_blv_eur_per_ha_for_healthy_areas = get_metric_for_given_tsum_and_frot(temperature_sum_for_stand / 1000.0, 0.0, spruce_blv_vs_tsum_and_frot)*1000.0
    spruce_total_blv_eur_for_healthy_areas = spruce_blv_eur_per_ha_for_healthy_areas*total_healthy_area_in_ha
    
    spruce_carbon_tCO2_per_ha_for_healthy_areas = get_metric_for_given_tsum_and_frot(temperature_sum_for_stand / 1000.0, 0.0, spruce_carbon_vs_tsum_and_frot)*100.0
    spruce_total_carbon_tCO2_for_healthy_areas = spruce_carbon_tCO2_per_ha_for_healthy_areas*total_healthy_area_in_ha
    
    total_blv_eur_for_stand = pine_total_blv_eur_for_rot_areas + spruce_total_blv_eur_for_healthy_areas
    total_carbon_tCO2_for_stand = pine_total_carbon_tCO2_for_rot_areas + spruce_total_carbon_tCO2_for_healthy_areas

    print("")
    print("+++ Alternative 5: pine for the rot areas, spruce outside of rot areas +++")
    print("")
    print("Total rotten area: %f ha" % total_rotten_area_in_ha)
    print("Total healthy area: %f ha" % total_healthy_area_in_ha) 
    print("BLV in EUR/ha for pine at this temperature sum and rot fraction: %f" % pine_blv_eur_per_ha_for_rot_areas)
    print("BLV in EUR/ha for spruce at this temperature sum and rot fraction: %f" % spruce_blv_eur_per_ha_for_healthy_areas)
    print("Carbon content in tCO2-equivalent/ha for pine at this temperature sum and rot fraction: %f" % pine_carbon_tCO2_per_ha_for_rot_areas)
    print("Carbon content in tCO2-equivalent/ha for spruce at this temperature sum and rot fraction: %f" % spruce_carbon_tCO2_per_ha_for_healthy_areas)
    print("")
    print("===> BLV for this alternative: %f EUR" % total_blv_eur_for_stand)
    print("===> Carbon content for this alternative: %f tCO2-equivalent" % total_carbon_tCO2_for_stand)
    print("")
    
    blv_eur_for_the_alternatives.append(float(total_blv_eur_for_stand))
    carbon_tCO2_for_the_alternatives.append(float(total_carbon_tCO2_for_stand))
    
    
    
    #
    # Alternative 6: Birch for the rot areas, spruce outside of rot areas.
    #
    
    total_healthy_area_in_ha = total_healthy_area_delineation.area / 1e4
    
    total_rotten_area_in_ha = total_rot_area_delineation.area / 1e4

    birch_blv_eur_per_ha_for_rot_areas = get_metric_for_given_tsum(temperature_sum_for_stand / 1000.0, birch_blv_vs_tsum)*1000.0
    birch_total_blv_eur_for_rot_areas = birch_blv_eur_per_ha_for_rot_areas*total_rotten_area_in_ha
    
    birch_carbon_tCO2_per_ha_for_rot_areas = get_metric_for_given_tsum(temperature_sum_for_stand / 1000.0, birch_carbon_vs_tsum)*100.0
    birch_total_carbon_tCO2_for_rot_areas = birch_carbon_tCO2_per_ha_for_rot_areas*total_rotten_area_in_ha
    
    spruce_blv_eur_per_ha_for_healthy_areas = get_metric_for_given_tsum_and_frot(temperature_sum_for_stand / 1000.0, 0.0, spruce_blv_vs_tsum_and_frot)*1000.0
    spruce_total_blv_eur_for_healthy_areas = spruce_blv_eur_per_ha_for_healthy_areas*total_healthy_area_in_ha
    
    spruce_carbon_tCO2_per_ha_for_healthy_areas = get_metric_for_given_tsum_and_frot(temperature_sum_for_stand / 1000.0, 0.0, spruce_carbon_vs_tsum_and_frot)*100.0
    spruce_total_carbon_tCO2_for_healthy_areas = spruce_carbon_tCO2_per_ha_for_healthy_areas*total_healthy_area_in_ha
    
    total_blv_eur_for_stand = birch_total_blv_eur_for_rot_areas + spruce_total_blv_eur_for_healthy_areas
    total_carbon_tCO2_for_stand = birch_total_carbon_tCO2_for_rot_areas + spruce_total_carbon_tCO2_for_healthy_areas
    
    print("")
    print("+++ Alternative 6: birch for the rot areas, spruce outside of rot areas +++")
    print("")
    print("Total rotten area: %f ha" % total_rotten_area_in_ha)
    print("Total healthy area: %f ha" % total_healthy_area_in_ha) 
    print("BLV in EUR/ha for birch at this temperature sum and rot fraction: %f" % birch_blv_eur_per_ha_for_rot_areas)
    print("BLV in EUR/ha for spruce at this temperature sum and rot fraction: %f" % spruce_blv_eur_per_ha_for_healthy_areas)
    print("Carbon content in tCO2-equivalent/ha for birch at this temperature sum and rot fraction: %f" % birch_carbon_tCO2_per_ha_for_rot_areas)
    print("Carbon content in tCO2-equivalent/ha for spruce at this temperature sum and rot fraction: %f" % spruce_carbon_tCO2_per_ha_for_healthy_areas)
    print("")
    print("===> BLV for this alternative: %f EUR" % total_blv_eur_for_stand)
    print("===> Carbon content for this alternative: %f tCO2-equivalent" % total_carbon_tCO2_for_stand)
    print("")
    
    blv_eur_for_the_alternatives.append(float(total_blv_eur_for_stand))
    carbon_tCO2_for_the_alternatives.append(float(total_carbon_tCO2_for_stand))
    


    #
    # Alternative 7: Pine for the rot areas, spruce-birch outside of rot areas.
    #

    total_healthy_area_in_ha = total_healthy_area_delineation.area / 1e4
    
    total_rotten_area_in_ha = total_rot_area_delineation.area / 1e4

    pine_blv_eur_per_ha_for_rot_areas = get_metric_for_given_tsum(temperature_sum_for_stand / 1000.0, pine_blv_vs_tsum)*1000.0
    pine_total_blv_eur_for_rot_areas = pine_blv_eur_per_ha_for_rot_areas*total_rotten_area_in_ha
    
    pine_carbon_tCO2_per_ha_for_rot_areas = get_metric_for_given_tsum(temperature_sum_for_stand / 1000.0, pine_carbon_vs_tsum)*100.0
    pine_total_carbon_tCO2_for_rot_areas = pine_carbon_tCO2_per_ha_for_rot_areas*total_rotten_area_in_ha
    
    spruce_birch_blv_eur_per_ha_for_healthy_areas = get_metric_for_given_tsum_and_frot(temperature_sum_for_stand / 1000.0, 0.0, spruce_birch_blv_vs_tsum_and_frot)*1000.0
    spruce_birch_total_blv_eur_for_healthy_areas = spruce_birch_blv_eur_per_ha_for_healthy_areas*total_healthy_area_in_ha
    
    spruce_birch_carbon_tCO2_per_ha_for_healthy_areas = get_metric_for_given_tsum_and_frot(temperature_sum_for_stand / 1000.0, 0.0, spruce_birch_carbon_vs_tsum_and_frot)*100.0
    spruce_birch_total_carbon_tCO2_for_healthy_areas = spruce_birch_carbon_tCO2_per_ha_for_healthy_areas*total_healthy_area_in_ha
    
    total_blv_eur_for_stand = pine_total_blv_eur_for_rot_areas + spruce_birch_total_blv_eur_for_healthy_areas
    total_carbon_tCO2_for_stand = pine_total_carbon_tCO2_for_rot_areas + spruce_birch_total_carbon_tCO2_for_healthy_areas
    
    print("")
    print("+++ Alternative 7: pine for the rot areas, spruce-birch outside of rot areas +++")
    print("")
    print("Total rotten area: %f ha" % total_rotten_area_in_ha)
    print("Total healthy area: %f ha" % total_healthy_area_in_ha) 
    print("BLV in EUR/ha for pine at this temperature sum and rot fraction: %f" % pine_blv_eur_per_ha_for_rot_areas)
    print("BLV in EUR/ha for spruce-birch at this temperature sum and rot fraction: %f" % spruce_birch_blv_eur_per_ha_for_healthy_areas)
    print("Carbon content in tCO2-equivalent/ha for pine at this temperature sum and rot fraction: %f" % pine_carbon_tCO2_per_ha_for_rot_areas)
    print("Carbon content in tCO2-equivalent/ha for spruce-birch at this temperature sum and rot fraction: %f" % spruce_birch_carbon_tCO2_per_ha_for_healthy_areas)
    print("")
    print("===> BLV for this alternative: %f EUR" % total_blv_eur_for_stand)
    print("===> Carbon content for this alternative: %f tCO2-equivalent" % total_carbon_tCO2_for_stand)
    print("")
    
    blv_eur_for_the_alternatives.append(float(total_blv_eur_for_stand))
    carbon_tCO2_for_the_alternatives.append(float(total_carbon_tCO2_for_stand))


    
    #
    # Alternative 8: Birch for the rot areas, spruce-birch outside of rot areas.
    #
    
    total_healthy_area_in_ha = total_healthy_area_delineation.area / 1e4
    
    total_rotten_area_in_ha = total_rot_area_delineation.area / 1e4
    
    birch_blv_eur_per_ha_for_rot_areas = get_metric_for_given_tsum(temperature_sum_for_stand / 1000.0, birch_blv_vs_tsum)*1000.0
    birch_total_blv_eur_for_rot_areas = birch_blv_eur_per_ha_for_rot_areas*total_rotten_area_in_ha
    
    birch_carbon_tCO2_per_ha_for_rot_areas = get_metric_for_given_tsum(temperature_sum_for_stand / 1000.0, birch_carbon_vs_tsum)*100.0
    birch_total_carbon_tCO2_for_rot_areas = birch_carbon_tCO2_per_ha_for_rot_areas*total_rotten_area_in_ha
    
    spruce_birch_blv_eur_per_ha_for_healthy_areas = get_metric_for_given_tsum_and_frot(temperature_sum_for_stand / 1000.0, 0.0, spruce_birch_blv_vs_tsum_and_frot)*1000.0
    spruce_birch_total_blv_eur_for_healthy_areas = spruce_birch_blv_eur_per_ha_for_healthy_areas*total_healthy_area_in_ha
    
    spruce_birch_carbon_tCO2_per_ha_for_healthy_areas = get_metric_for_given_tsum_and_frot(temperature_sum_for_stand / 1000.0, 0.0, spruce_birch_carbon_vs_tsum_and_frot)*100.0
    spruce_birch_total_carbon_tCO2_for_healthy_areas = spruce_birch_carbon_tCO2_per_ha_for_healthy_areas*total_healthy_area_in_ha
    
    total_blv_eur_for_stand = birch_total_blv_eur_for_rot_areas + spruce_birch_total_blv_eur_for_healthy_areas
    total_carbon_tCO2_for_stand = birch_total_carbon_tCO2_for_rot_areas + spruce_birch_total_carbon_tCO2_for_healthy_areas
    
    print("")
    print("+++ Alternative 8: birch for the rot areas, spruce-birch outside of rot areas +++")
    print("")
    print("Total rotten area: %f ha" % total_rotten_area_in_ha)
    print("Total healthy area: %f ha" % total_healthy_area_in_ha) 
    print("BLV in EUR/ha for birch at this temperature sum and rot fraction: %f" % birch_blv_eur_per_ha_for_rot_areas)
    print("BLV in EUR/ha for spruce-birch at this temperature sum and rot fraction: %f" % spruce_birch_blv_eur_per_ha_for_healthy_areas)
    print("Carbon content in tCO2-equivalent/ha for birch at this temperature sum and rot fraction: %f" % birch_carbon_tCO2_per_ha_for_rot_areas)
    print("Carbon content in tCO2-equivalent/ha for spruce-birch at this temperature sum and rot fraction: %f" % spruce_birch_carbon_tCO2_per_ha_for_healthy_areas)
    print("")
    print("===> BLV for this alternative: %f EUR" % total_blv_eur_for_stand)
    print("===> Carbon content for this alternative: %f tCO2-equivalent" % total_carbon_tCO2_for_stand)
    print("")
    
    blv_eur_for_the_alternatives.append(float(total_blv_eur_for_stand))
    carbon_tCO2_for_the_alternatives.append(float(total_carbon_tCO2_for_stand))
    
    
    
    #
    # Alternative 9: Pine for the rot areas, birch outside of rot areas.
    #
    
    total_healthy_area_in_ha = total_healthy_area_delineation.area / 1e4
    
    total_rotten_area_in_ha = total_rot_area_delineation.area / 1e4

    pine_blv_eur_per_ha_for_rot_areas = get_metric_for_given_tsum(temperature_sum_for_stand / 1000.0, pine_blv_vs_tsum)*1000.0
    pine_total_blv_eur_for_rot_areas = pine_blv_eur_per_ha_for_rot_areas*total_rotten_area_in_ha
    
    pine_carbon_tCO2_per_ha_for_rot_areas = get_metric_for_given_tsum(temperature_sum_for_stand / 1000.0, pine_carbon_vs_tsum)*100.0
    pine_total_carbon_tCO2_for_rot_areas = pine_carbon_tCO2_per_ha_for_rot_areas*total_rotten_area_in_ha
    
    birch_blv_eur_per_ha_for_healthy_areas = get_metric_for_given_tsum(temperature_sum_for_stand / 1000.0, birch_blv_vs_tsum)*1000.0
    birch_total_blv_eur_for_healthy_areas = birch_blv_eur_per_ha_for_healthy_areas*total_healthy_area_in_ha
    
    birch_carbon_tCO2_per_ha_for_healthy_areas = get_metric_for_given_tsum(temperature_sum_for_stand / 1000.0, birch_carbon_vs_tsum)*100.0
    birch_total_carbon_tCO2_for_healthy_areas = birch_carbon_tCO2_per_ha_for_healthy_areas*total_healthy_area_in_ha
    
    total_blv_eur_for_stand = pine_total_blv_eur_for_rot_areas + birch_total_blv_eur_for_healthy_areas
    total_carbon_tCO2_for_stand = pine_total_carbon_tCO2_for_rot_areas + birch_total_carbon_tCO2_for_healthy_areas
    
    print("")
    print("+++ Alternative 9: pine for the rot areas, birch outside of rot areas +++")
    print("")
    print("Total rotten area: %f ha" % total_rotten_area_in_ha)
    print("Total healthy area: %f ha" % total_healthy_area_in_ha) 
    print("BLV in EUR/ha for pine at this temperature sum and rot fraction: %f" % pine_blv_eur_per_ha_for_rot_areas)
    print("BLV in EUR/ha for birch at this temperature sum and rot fraction: %f" % birch_blv_eur_per_ha_for_healthy_areas)
    print("Carbon content in tCO2-equivalent/ha for pine at this temperature sum and rot fraction: %f" % pine_carbon_tCO2_per_ha_for_rot_areas)
    print("Carbon content in tCO2-equivalent/ha for birch at this temperature sum and rot fraction: %f" % birch_carbon_tCO2_per_ha_for_healthy_areas)
    print("")
    print("===> BLV for this alternative: %f EUR" % total_blv_eur_for_stand)
    print("===> Carbon content for this alternative: %f tCO2-equivalent" % total_carbon_tCO2_for_stand)
    print("")
    
    blv_eur_for_the_alternatives.append(float(total_blv_eur_for_stand))
    carbon_tCO2_for_the_alternatives.append(float(total_carbon_tCO2_for_stand))



    #
    # Alternative 10: Birch for the rot areas, pine outside of rot areas.
    #
    
    total_healthy_area_in_ha = total_healthy_area_delineation.area / 1e4
    
    total_rotten_area_in_ha = total_rot_area_delineation.area / 1e4
        
    birch_blv_eur_per_ha_for_rot_areas = get_metric_for_given_tsum(temperature_sum_for_stand / 1000.0, birch_blv_vs_tsum)*1000.0
    birch_total_blv_eur_for_rot_areas = birch_blv_eur_per_ha_for_rot_areas*total_rotten_area_in_ha
    
    birch_carbon_tCO2_per_ha_for_rot_areas = get_metric_for_given_tsum(temperature_sum_for_stand / 1000.0, birch_carbon_vs_tsum)*100.0
    birch_total_carbon_tCO2_for_rot_areas = birch_carbon_tCO2_per_ha_for_rot_areas*total_rotten_area_in_ha
    
    pine_blv_eur_per_ha_for_healthy_areas = get_metric_for_given_tsum(temperature_sum_for_stand / 1000.0, pine_blv_vs_tsum)*1000.0
    pine_total_blv_eur_for_healthy_areas = pine_blv_eur_per_ha_for_healthy_areas*total_healthy_area_in_ha
    
    pine_carbon_tCO2_per_ha_for_healthy_areas = get_metric_for_given_tsum(temperature_sum_for_stand / 1000.0, pine_carbon_vs_tsum)*100.0
    pine_total_carbon_tCO2_for_healthy_areas = pine_carbon_tCO2_per_ha_for_healthy_areas*total_healthy_area_in_ha
    
    total_blv_eur_for_stand = birch_total_blv_eur_for_rot_areas + pine_total_blv_eur_for_healthy_areas
    total_carbon_tCO2_for_stand = birch_total_carbon_tCO2_for_rot_areas + pine_total_carbon_tCO2_for_healthy_areas

    print("")
    print("+++ Alternative 10: birch for the rot areas, pine outside of rot areas +++")
    print("")
    print("Total rotten area: %f ha" % total_rotten_area_in_ha)
    print("Total healthy area: %f ha" % total_healthy_area_in_ha) 
    print("BLV in EUR/ha for birch at this temperature sum and rot fraction: %f" % birch_blv_eur_per_ha_for_rot_areas)
    print("BLV in EUR/ha for pine at this temperature sum and rot fraction: %f" % pine_blv_eur_per_ha_for_healthy_areas)
    print("Carbon content in tCO2-equivalent/ha for birch at this temperature sum and rot fraction: %f" % birch_carbon_tCO2_per_ha_for_rot_areas)
    print("Carbon content in tCO2-equivalent/ha for pine at this temperature sum and rot fraction: %f" % pine_carbon_tCO2_per_ha_for_healthy_areas)
    print("")
    print("===> BLV for this alternative: %f EUR" % total_blv_eur_for_stand)
    print("===> Carbon content for this alternative: %f tCO2-equivalent" % total_carbon_tCO2_for_stand)

    blv_eur_for_the_alternatives.append(float(total_blv_eur_for_stand))
    carbon_tCO2_for_the_alternatives.append(float(total_carbon_tCO2_for_stand))



    #
    # All alternatives have now been processed. Print out a summary, then
    # choose the best one for BLV, and choose the best one for carbon.
    #

    print("")
    print("All alternatives have now been processed.")
    print("")
    print("")
    print("Found the following metrics for the alternatives:")
    print("")
    print("Alternative BLV (EUR) Carbon (tCO2-equivalent)")
    print("")

    for i_alternative in np.arange(0, len(list_of_alternatives)):

        this_alternative = list_of_alternatives[i_alternative]
        this_blv_in_eur = blv_eur_for_the_alternatives[i_alternative]
        this_carbon_in_tCO2 = carbon_tCO2_for_the_alternatives[i_alternative]
    
        print("%s %s %s" % (this_alternative, this_blv_in_eur, this_carbon_in_tCO2))

    print("")



    #
    # List the alternatives by decreasing BLV
    #

    indeces_from_lowest_to_highest_blv = list(np.argsort(blv_eur_for_the_alternatives))
    
    print("")
    print("The alternatives ranked in terms of BLV are as follows:")
    print("")

    for i_alternative in indeces_from_lowest_to_highest_blv[::-1]:
    
        this_alternative = list_of_alternatives[i_alternative]

        this_blv_in_eur = blv_eur_for_the_alternatives[i_alternative]

        print("%s %s" % (this_alternative, this_blv_in_eur))
    

    #
    # Out of those solutions that purify the stand of root rot
    # disease, find the one that maximizes BLV. All alternatives
    # except the first (spruce for the entire stand) and second
    # (spruce-birch for the entire stand) purify the stand of root rot
    # disease.
    #

    indeces_from_lowest_to_highest_blv_for_purifying_solutions = indeces_from_lowest_to_highest_blv.copy()

    indeces_from_lowest_to_highest_blv_for_purifying_solutions.remove(0)

    indeces_from_lowest_to_highest_blv_for_purifying_solutions.remove(1)

    index_for_best_purifying_alternative_for_blv = indeces_from_lowest_to_highest_blv_for_purifying_solutions[-1]
    
    best_purifying_alternative_for_blv = list_of_alternatives[index_for_best_purifying_alternative_for_blv]
    
    print("")
    print("===> To maximize BLV subject to the condition that the stand becomes purified of root rot disease, choose %s" % best_purifying_alternative_for_blv.lower())
    print("")

    
    
    #
    # List the alternatives by decreasing carbon
    #
    
    indeces_from_lowest_to_highest_carbon = list(np.argsort(carbon_tCO2_for_the_alternatives))

    print("")
    print("The alternatives ranked in terms of carbon are as follows:")
    print("")
    
    for i_alternative in indeces_from_lowest_to_highest_carbon[::-1]:
    
        this_alternative = list_of_alternatives[i_alternative]

        this_carbon_in_tCO2 = carbon_tCO2_for_the_alternatives[i_alternative]

        print("%s %s" % (this_alternative, this_carbon_in_tCO2))


    #
    # Out of those solutions that purify the stand of root rot
    # disease, find the one that maximizes carbon. All alternatives
    # except the first (spruce for the entire stand) and second
    # (spruce-birch for the entire stand) purify the stand of root rot
    # disease.
    #

    indeces_from_lowest_to_highest_carbon_for_purifying_solutions = indeces_from_lowest_to_highest_carbon.copy()

    indeces_from_lowest_to_highest_carbon_for_purifying_solutions.remove(0)

    indeces_from_lowest_to_highest_carbon_for_purifying_solutions.remove(1)

    index_for_best_purifying_alternative_for_carbon = indeces_from_lowest_to_highest_carbon_for_purifying_solutions[-1]
    
    best_purifying_alternative_for_carbon = list_of_alternatives[index_for_best_purifying_alternative_for_carbon]
    
    print("")
    print("===> To maximize carbon subject to the condition that the stand becomes purified of root rot disease, choose %s" % best_purifying_alternative_for_carbon.lower())
    print("")



    return list_of_alternatives, indeces_from_lowest_to_highest_blv, blv_eur_for_the_alternatives, indeces_from_lowest_to_highest_carbon, carbon_tCO2_for_the_alternatives



#
# Read in the value this_tsum for temperature sum (in units of 1000
# d.d.)  and a set of tabulated (x, y) data with temperature sum in
# the first column. Return y, i.e., the BLV or carbon reading, for the
# temperature sum closest to this_tsum in the tabulated data.
#

def get_metric_for_given_tsum(this_tsum, xy_tabulated_data):

    i_desired = np.argmin(np.abs(this_tsum - xy_tabulated_data[:, 0]))

    return xy_tabulated_data[i_desired, 1]



#
# Read in the value this_tsum for temperature sum (in units of 1000
# d.d), this_frot for rot fraction, and a set of tabulated (x, y, z)
# data with temperature sum in the first column and rot fraction in
# the second column. Return z, i.e., the BLV or carbon reading for the
# (temperature sum, rot fraction) point closest to (this_tsum,
# this_frot) in the tabulated data.
#

def get_metric_for_given_tsum_and_frot(this_tsum, this_frot, xyz_tabulated_data):

    i_desired = np.argmin(np.linalg.norm(np.array([this_tsum, this_frot]) - xyz_tabulated_data[:, 0:2], axis = 1))

    return xyz_tabulated_data[i_desired, 2]



#
# This function creates, for a given Shapely Polygon, a PathPatch
# object for plotting the object using matplotlib.
#

def create_polygon_pathpatch(the_shape, alpha, facecolor, edgecolor, linewidth = None, zorder = -1):

    #
    # Create the PathPatch object using the given shape and
    # visualization parameters
    #
    
    the_pathpatch_object = PathPatch(Path.make_compound_path(Path(np.asarray(the_shape.exterior.coords)[:, :2]), *[Path(np.asarray(ring.coords)[:, :2]) for ring in the_shape.interiors]), alpha = alpha, facecolor = facecolor, edgecolor = edgecolor, linewidth = linewidth, zorder = zorder)

    return the_pathpatch_object



#
# Takes as input a microstand delineation (a Shapely Polygon or
# MultiPolygon), as well as the species group, theoretical sawlog
# volume, rot indicator, and stem positional data for the full harvest
# area. Computes and returns the fraction of rotten sawlog-caliber
# spruce stems within the given delineation.
#

def get_rot_fraction_within_given_delineation(the_delineation, species_group_id, theoretical_sawlog_volume, rotten_200_with_sawlog, easting_crane_tip, northing_crane_tip):


    #
    # Make sure the given delineation is a shapely Polygon or
    # MultiPolygon. If not, set the rot fraction to zero.
    #
    
    if not (type(the_delineation) is Polygon or type(the_delineation) is MultiPolygon):

        print("")
        print("Warning: Rot fraction queried for a delineation of type", type(the_delineation), ", setting the rot fraction to zero.")
        
        local_rot_fraction = 0.0

    else:
        
        #
        # Get the species group, theoretical sawlog volume, and the
        # rot indicator for each stem within the given
        # delineation. This will be enough to compute the fraction of
        # rotten, sawlog-caliber spruce stems within the delineation.
        #

        #
        # Create an index array of True / False values to indicate
        # whether a stem is within the delineation or not
        #

        stem_indeces_for_this_delineation = []
        
        for i_stem in np.arange(0, easting_crane_tip.shape[0]):

            this_easting = easting_crane_tip[i_stem]

            this_northing = northing_crane_tip[i_stem]
            
            stem_indeces_for_this_delineation.append(Point(this_easting, this_northing).within(the_delineation))

            
        #
        # Then slice the species group, theoretical sawlog volume, and
        # rot indicator data to get these data for just the given
        # delineation
        #
        
        species_group_id_for_this_delineation = species_group_id[stem_indeces_for_this_delineation]
        theoretical_sawlog_volume_for_this_delineation = theoretical_sawlog_volume[stem_indeces_for_this_delineation]
        rotten_200_with_sawlog_for_this_delineation = rotten_200_with_sawlog[stem_indeces_for_this_delineation]


        #
        # Compute the number of rotten sawlog-caliber spruce stems and
        # all sawlog-caliber spruce stems within the delineation, and
        # from these, the rot fraction for the delineation
        #
        
        n_rotten_spruce_sawlog_caliber_stems_in_this_delineation = np.sum(np.logical_and(species_group_id_for_this_delineation == 2, np.logical_and(theoretical_sawlog_volume_for_this_delineation > 0.0, rotten_200_with_sawlog_for_this_delineation == 1)).astype(int))
        n_spruce_sawlog_caliber_stems_in_this_delineation = np.sum(np.logical_and(species_group_id_for_this_delineation == 2, theoretical_sawlog_volume_for_this_delineation > 0.0).astype(int))

        local_rot_fraction = n_rotten_spruce_sawlog_caliber_stems_in_this_delineation / n_spruce_sawlog_caliber_stems_in_this_delineation


    return local_rot_fraction



#
# Get a terrain map in raster form for use as a background for some of
# the plots.
#

def get_terrain_raster_map(stem_positions_min_easting, stem_positions_max_easting, stem_positions_min_northing, stem_positions_max_northing):

    
    #
    # Find which map sheet we are on
    #

    print("")
    print("Now finding which map sheet to use for the background map...")


    #
    # Read in the x and y coordinate of the upper left pixel center of
    # each map sheet of the 1:50000 raster to create a dictionary of the
    # format
    #
    # map sheet world file : (upper left pixel center x-coordinate, upper left pixel center y-coordinate )
    #


    #
    # Get a list of all the available map sheets
    #
    
    map_sheet_worldfiles = glob.glob(parameters.map_sheet_root + '[K-X][0-9]/*/*pgw')


    #
    # Then create the dictionary
    #


    map_sheet_dictionary = {}


    for f in map_sheet_worldfiles:

        this_worldfile_data = np.loadtxt(f)
    
        this_x_coordinate_of_ul_pixel_center = this_worldfile_data[4]

        this_y_coordinate_of_ul_pixel_center = this_worldfile_data[5]

        map_sheet_dictionary[f] = (this_x_coordinate_of_ul_pixel_center, this_y_coordinate_of_ul_pixel_center)



    #
    # Then, find the map sheet whose upper left pixel center coordinates
    # are the first to the left and upwards of the upper left data point
    # of the stand bounding box. Use this map sheet for the plots. We call
    # this map sheet the "base map sheet", and it will be extended by
    # other maps sheets below, if the stand stem positions reach outside
    # of it.
    #


    best_ul_pixel_center_so_far = (-1e15, 1e15)

    best_map_sheet_world_file_so_far = None


    for f, ul_pixel_center_coordinates in map_sheet_dictionary.items():


        this_x_coordinate_of_ul_pixel_center = ul_pixel_center_coordinates[0]

        this_y_coordinate_of_ul_pixel_center = ul_pixel_center_coordinates[1]


        #
        # We only consider sheets whose upper left pixel center is to
        # the left and upwards of the stem position data bounding box
        # upper left point
        #
    
        if this_x_coordinate_of_ul_pixel_center > stem_positions_min_easting or this_y_coordinate_of_ul_pixel_center < stem_positions_max_northing:

            continue


        #
        # See if this map sheet is a better choice than any of the previous ones
        #

        if this_x_coordinate_of_ul_pixel_center >= best_ul_pixel_center_so_far[0] and this_y_coordinate_of_ul_pixel_center <= best_ul_pixel_center_so_far[1]:

            best_ul_pixel_center_so_far = (this_x_coordinate_of_ul_pixel_center, this_y_coordinate_of_ul_pixel_center)
            best_map_sheet_world_file_so_far = f


    #
    # We've now found the "best" map sheet for the given stem position
    # data, i.e., the base map sheet
    #


    worldfile = best_map_sheet_world_file_so_far

    map_rasterfile = worldfile[0:len(worldfile)-4] + '.png'

    print("Done. Using the sheet %s with worldfile %s" % (map_rasterfile, worldfile))



    #
    # Read in the map raster and the associated world file
    #

    Image.MAX_IMAGE_PIXELS = None


    print("")
    print("Now reading in map raster from the file %s..." % map_rasterfile)


    #
    # Load the map image
    #

    map_as_image = Image.open(map_rasterfile)

    print("Done. Here are some stats on the image:")
    print("")


    #
    # Print out some stats on the image
    #

    print("Format:", map_as_image.format)
    print("Size (width, height):", map_as_image.size)
    print("Mode:", map_as_image.mode)
    print("Bands:", map_as_image.getbands())


    #
    # Create a colormap for the upcoming raster map plot to re-create the
    # original PNG map colors. The image mode should be "P", which means
    # that each pixel has a value in the range of 0...255, and each pixel
    # value is mapped to a color using a palette. In these PNGs it
    # appears, however, that the pixel values go from 0...254, i.e., there
    # are a total of 255 different values, not 256. In the palette itself,
    # each element of the color triplet is in the range of 0...255.
    #


    #
    # Get the palette
    #

    the_palette = map_as_image.getpalette()

    n_palette_colors = int(len(the_palette) / 3)

    print("")
    print("Found palette of %d colors" % n_palette_colors)


    #
    # Create a dictionary of the format
    #
    # <pixel value> : <color>
    #
    
    colors_for_pixel_values = {}


    #
    # Get the colors from the original palette
    #
        
    for pixel_value in range(0, n_palette_colors):
    
        colors_for_pixel_values[pixel_value] = list(np.array(the_palette[3*pixel_value : 3*pixel_value+3]) / 255.0)


    raster_cmap = colors.ListedColormap([colors_for_pixel_values[key] for key in range(0, n_palette_colors)])

    cmap_bounds = [b for b in np.arange(0, n_palette_colors + 1) - 0.5]

    raster_cmap_norm = colors.BoundaryNorm(cmap_bounds, n_palette_colors)


    #
    # Convert the raster into a numpy array
    #

    map_as_array = np.array(map_as_image)

    base_map_size_x = map_as_array.shape[1]

    base_map_size_y = map_as_array.shape[0]


    print("")
    print("Converted the raster map into a numpy array of shape", map_as_array.shape)


    #
    # Read in the world file
    #

    print("")
    print("Now reading in the world file for the raster...")

    worldfile_data = np.loadtxt(worldfile)

    pixel_size_in_x_direction = np.abs(worldfile_data[0])
    pixel_size_in_y_direction = np.abs(worldfile_data[3])
    x_coordinate_of_ul_pixel_center = worldfile_data[4]
    y_coordinate_of_ul_pixel_center = worldfile_data[5]

    print("Done. Found the following data:")
    print("")

    print("Pixel size in x direction: %f" % pixel_size_in_x_direction)
    print("Pixel size in y direction: %f" % pixel_size_in_y_direction)
    print("Upper left pixel center x-coordinate: %f" % x_coordinate_of_ul_pixel_center)
    print("Upper left pixel center y-coordinate: %f" % y_coordinate_of_ul_pixel_center)


    #
    # Get the easting-northing limits of the map in terms of pixel centers
    #

    map_pixel_center_min_easting = x_coordinate_of_ul_pixel_center
    map_pixel_center_max_easting = x_coordinate_of_ul_pixel_center + pixel_size_in_x_direction*(base_map_size_x - 1)
    map_pixel_center_max_northing = y_coordinate_of_ul_pixel_center
    map_pixel_center_min_northing = y_coordinate_of_ul_pixel_center - pixel_size_in_y_direction*(base_map_size_y - 1)

    print("")
    print("The map raster pixel centers run from easting %f m to %f m and northing %f m to %f m" % (map_pixel_center_min_easting, map_pixel_center_max_easting, map_pixel_center_min_northing, map_pixel_center_max_northing))


    #
    # Check whether the bounding box of the stem position data is within
    # the range of the base map sheet. If not, extend the base map in the
    # positive easting and negative northing directions with as many map
    # sheets as is necessary.
    #
    
    if stem_positions_max_easting > map_pixel_center_max_easting or stem_positions_min_northing < map_pixel_center_min_northing:

        #
        # Find the number of map sheets that you need in each dimension to cover the full stem position data
        #
    
        map_sheet_size_easting = map_pixel_center_max_easting - map_pixel_center_min_easting + 2.0 * pixel_size_in_x_direction/2.0
        map_sheet_size_northing = map_pixel_center_max_northing - map_pixel_center_min_northing + 2.0 * pixel_size_in_y_direction/2.0
        
        n_tiles_east = (np.floor((stem_positions_max_easting - map_pixel_center_min_easting) / map_sheet_size_easting) + 1).astype(int)
        n_tiles_south = (np.floor(-1.0*(stem_positions_min_northing - map_pixel_center_max_northing) / map_sheet_size_northing) + 1).astype(int)

        print("Bounding box reaches outside of the base raster map geographical range. Now extending the raster map to comprise %d sheets east and %d sheets south..." % (n_tiles_east, n_tiles_south))


        #
        # First, extend the map eastward from the base map sheet.
        #
    
        #
        # Find all the map sheets directly east of the base map sheet,
        # order them in terms of increasing easting of the upper left
        # pixel center, and then add the necessary amount of sheets to the
        # base map sheet, thus creating the "top row" of the new, extended
        # map.
        #

        map_sheets_of_top_row = []


        for this_map_sheet, these_ul_coordinates in map_sheet_dictionary.items():

            if these_ul_coordinates[1] == y_coordinate_of_ul_pixel_center and these_ul_coordinates[0] >= x_coordinate_of_ul_pixel_center:

                map_sheets_of_top_row.append([this_map_sheet, these_ul_coordinates])


        map_sheets_of_top_row.sort(key = lambda ul_coords: ul_coords[1][0])

        map_sheets_of_top_row = map_sheets_of_top_row[:n_tiles_east]


        if len(map_sheets_of_top_row) > 1:

            print("")
            print("Now appending the following map sheets east of the base map sheet:")
            print("")

            for this_map_sheet in map_sheets_of_top_row[1:]:

                this_worldfile = this_map_sheet[0]
                
                print("---> %s" % this_worldfile)
                
                this_map_rasterfile = this_worldfile[0:len(worldfile)-4] + '.png'
                
                this_map_sheet_as_image = Image.open(this_map_rasterfile)
                
                this_map_sheet_as_array = np.array(this_map_sheet_as_image)
                
                map_as_array = np.append(map_as_array, this_map_sheet_as_array, axis = 1)


        #
        # Update the variables holding the map size
        #

        map_pixel_center_max_easting = x_coordinate_of_ul_pixel_center + pixel_size_in_x_direction*(map_as_array.shape[1] - 1)


        #
        # Then, extend the map southward of the top row
        #

        #
        # For each top row map sheet, find all the map sheets directly
        # south of that sheet, order them in terms of decreasing northing
        # of the upper left pixel center, and save them as a list. Each
        # such list makes up a column of map sheets to append to the total
        # map. Once you have this list for each of the top row map sheets,
        # create a big numpy array of them and append them in one go to
        # the total map.
        #

        map_sheets_to_append_southwards_by_column = []
    
    
        for this_top_row_map_sheet in map_sheets_of_top_row:

            map_sheets_of_this_column = []

            x_coordinate_of_this_top_row_map_sheet_ul_pixel_center = this_top_row_map_sheet[1][0]
            y_coordinate_of_this_top_row_map_sheet_ul_pixel_center = this_top_row_map_sheet[1][1]
        
            for this_map_sheet, these_ul_coordinates in map_sheet_dictionary.items():

                if these_ul_coordinates[0] == x_coordinate_of_this_top_row_map_sheet_ul_pixel_center and these_ul_coordinates[1] <= y_coordinate_of_this_top_row_map_sheet_ul_pixel_center:

                    map_sheets_of_this_column.append([this_map_sheet, these_ul_coordinates])


            map_sheets_of_this_column.sort(key = lambda ul_coords: ul_coords[1][1], reverse = True)

            map_sheets_of_this_column = map_sheets_of_this_column[1:n_tiles_south]

            map_sheets_to_append_southwards_by_column.append(map_sheets_of_this_column)

     
        print("")
        print("Preparing to append the following map sheets southwards:")
 
        i_column = 1
    
        for column in map_sheets_to_append_southwards_by_column:

            print("")
            print("Column %d:" % i_column)

            for map_sheet in column:

                print("---> %s" % map_sheet[0])

            i_column = i_column + 1


        #
        # To append the map sheets south of the top row, create one big
        # numpy array of these, and then append
        #

        map_sheets_to_append_southwards_as_array = np.zeros([0, map_as_array.shape[1]])


        #
        # Loop over rows, appending each to the big array as you go
        #

        for i_row in range(0, n_tiles_south-1):

            map_sheets_for_this_row = [e[i_row] for e in  map_sheets_to_append_southwards_by_column]

            map_sheets_for_this_row_as_array = np.zeros([base_map_size_y, 0])

            #
            # Load the map sheet rasters for this row and convert them
            # into a single numpy array
            #

            for this_map_sheet in map_sheets_for_this_row:

                this_worldfile = this_map_sheet[0]
                this_map_rasterfile = this_worldfile[0:len(worldfile)-4] + '.png'

                this_map_sheet_as_image = Image.open(this_map_rasterfile)

                this_map_sheet_as_array = np.array(this_map_sheet_as_image)

                map_sheets_for_this_row_as_array = np.append(map_sheets_for_this_row_as_array, this_map_sheet_as_array, axis = 1)
            

            #
            # Then add this row to the big array
            #

            map_sheets_to_append_southwards_as_array = np.append(map_sheets_to_append_southwards_as_array, map_sheets_for_this_row_as_array, axis = 0)


        #
        # Finally, append all the rows at once to the top row
        #

        map_as_array = np.append(map_as_array, map_sheets_to_append_southwards_as_array, axis = 0)


        #
        # Update variables holding the map size
        #

        map_pixel_center_min_northing = y_coordinate_of_ul_pixel_center - pixel_size_in_y_direction*(map_as_array.shape[0] - 1)


        print("")
        print("Extension operation completed.")
        print("")
        print("The final raster map is a numpy array of shape", map_as_array.shape)
        print("")
        print("The map raster pixel centers now run from easting %f m to %f m and northing %f m to %f m" % (map_pixel_center_min_easting, map_pixel_center_max_easting, map_pixel_center_min_northing, map_pixel_center_max_northing))
        print("")


    return map_as_array, pixel_size_in_x_direction, pixel_size_in_y_direction, map_pixel_center_min_easting, map_pixel_center_max_easting, map_pixel_center_min_northing, map_pixel_center_max_northing, raster_cmap, raster_cmap_norm



#
# Create a set of "map-like" plots of the stand and the
# results. First, create a set of plots common to both delineation
# methods (dbscan, external geojson). Then, create a set of dbscan-specific
# plots.
#

def create_maplike_plots(easting_crane_tip, northing_crane_tip, stem_positions_min_easting, stem_positions_max_easting, stem_positions_min_northing, stem_positions_max_northing, easting_cabin_position, northing_cabin_position, species_group_id, rotten_200_with_sawlog, theoretical_sawlog_volume, fraction_of_rotten_spruce_sawlog_caliber_stems, sliced_fertility_class_as_array, sliced_fertility_class_data_min_easting, sliced_fertility_class_data_max_easting, sliced_fertility_class_data_min_northing, sliced_fertility_class_data_max_northing, fertility_class_pixel_size_x, fertility_class_pixel_size_y, sliced_soil_type_as_array, sliced_soil_type_data_min_easting, sliced_soil_type_data_max_easting, sliced_soil_type_data_min_northing, sliced_soil_type_data_max_northing, soil_type_pixel_size_x, soil_type_pixel_size_y, cluster_delineations, individual_stem_delineations, full_harvest_area_delineation, total_rot_area_delineation, total_healthy_area_delineation, position_and_cluster_id_data_for_each_cluster_including_outliers, rot_cluster_color_map, temperature_sum_for_stand, delineation_method):



    #
    # Set a common set of axis limits for all the plots
    #

    graph_axis_limits = [stem_positions_min_easting - parameters.plot_padding_in_x_and_y, stem_positions_max_easting + parameters.plot_padding_in_x_and_y, stem_positions_min_northing - parameters.plot_padding_in_x_and_y, stem_positions_max_northing + parameters.plot_padding_in_x_and_y]

    
    
    #
    # If the use of external raster data is required, get the terrain
    # map as a raster
    #

    if parameters.use_raster_data:

        
        terrain_map_as_array, terrain_pixel_size_in_x_direction, terrain_pixel_size_in_y_direction, terrain_map_pixel_center_min_easting, terrain_map_pixel_center_max_easting, terrain_map_pixel_center_min_northing, terrain_map_pixel_center_max_northing, terrain_raster_cmap, terrain_raster_cmap_norm = get_terrain_raster_map(stem_positions_min_easting, stem_positions_max_easting, stem_positions_min_northing, stem_positions_max_northing)
    
    
    
    #
    # Then, create the plots common to both delineation methods (dbscan, external geojson)
    #
    
    
    
    #
    # All stems as small black dots, on four different backgrounds: fertility class, soil type, terrain map, white
    #
    

    if parameters.use_raster_data:

        
        #
        # On fertility class
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(sliced_fertility_class_as_array, extent = (sliced_fertility_class_data_min_easting - fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_max_easting + fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_min_northing - fertility_class_pixel_size_y/2.0, sliced_fertility_class_data_max_northing + fertility_class_pixel_size_y/2.0), cmap = parameters.fertility_class_cmap, norm = parameters.fertility_class_cmap_norm, alpha = parameters.fertility_class_visualization_alpha, zorder = -999)

        cbar = fig.colorbar(mappable, ticks = parameters.fertility_class_tick_locations, pad = 0.06, shrink = 0.8)
        cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
        cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
        cbar.ax.set_yticklabels(parameters.fertility_class_tick_labels, size = parameters.fertility_class_tick_label_font_size)

        h_stem_positions = ax.scatter(easting_crane_tip, northing_crane_tip, c = 'k', s = parameters.stem_markersize_small, marker = '.')

        ax.set_aspect('equal', 'box')
        ax.set_title('Stem positions', pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)

        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        
        fig.savefig('stems_as_points_on_fertility_class.png')
        plt.close(fig)



        #
        # On soil type
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(sliced_soil_type_as_array, extent = (sliced_soil_type_data_min_easting - soil_type_pixel_size_x/2.0, sliced_soil_type_data_max_easting + soil_type_pixel_size_x/2.0, sliced_soil_type_data_min_northing - soil_type_pixel_size_y/2.0, sliced_soil_type_data_max_northing + soil_type_pixel_size_y/2.0), cmap = parameters.soil_type_cmap, norm = parameters.soil_type_cmap_norm, alpha = parameters.soil_type_visualization_alpha, zorder = -999)

        cbar = fig.colorbar(mappable, ticks = parameters.soil_type_tick_locations, pad = 0.06, shrink = 0.8)
        cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
        cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
        cbar.ax.set_yticklabels(parameters.soil_type_tick_labels, size = parameters.soil_type_tick_label_font_size)

        h_stem_positions = ax.scatter(easting_crane_tip, northing_crane_tip, c = 'k', s = parameters.stem_markersize_small, marker = '.')

        ax.set_aspect('equal', 'box')
        ax.set_title('Stem positions', pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)

        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))

        fig.savefig('stems_as_points_on_soil_type.png')
        plt.close(fig)



        #
        # On terrain map
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(terrain_map_as_array, cmap = terrain_raster_cmap, norm = terrain_raster_cmap_norm, extent = (terrain_map_pixel_center_min_easting - terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_max_easting + terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_min_northing - terrain_pixel_size_in_y_direction/2.0, terrain_map_pixel_center_max_northing + terrain_pixel_size_in_y_direction/2.0), alpha = parameters.terrain_raster_map_alpha, zorder = -999)

        h_stem_positions = ax.scatter(easting_crane_tip, northing_crane_tip, c = 'k', s = parameters.stem_markersize_small, marker = '.')

        ax.set_aspect('equal', 'box')
        ax.set_title('Stem positions', pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)

        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))

        fig.savefig('stems_as_points_on_terrain_map.png')
        plt.close(fig)



    #
    # On white background
    #

    fig, ax = plt.subplots(figsize = parameters.figure_size)

    h_stem_positions = ax.scatter(easting_crane_tip, northing_crane_tip, c = 'k', s = parameters.stem_markersize_small, marker = '.')
    
    ax.set_aspect('equal', 'box')
    ax.set_title('Stem positions', pad = parameters.titlepad)
    ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
    ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)

    ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
    
    fig.savefig('stems_as_points_on_white_background.png')
    plt.close(fig)



    #
    # Machine cabin position and crane tip position for all harvested stems, on the four different backgrounds
    #

    
    #
    # First, set the cabin position and crane tip position data into two
    # 2D arrays of the following format:
    #
    # [[cabin position x stem 1, cabin position x stem 2, ... cabin position x stem N]
    # [[crane tip position x stem 1, crane tip position x stem 2, ..., crane tip position x stem N]]
    #
    # [[cabin position y stem 1, cabin position y stem 2, ... cabin position y stem N]
    # [[crane tip position y stem 1, crane tip position y stem 2, ..., crane tip position y stem N]]
    #

    cabin_and_crane_tip_position_2D_array_easting = np.vstack((easting_cabin_position, easting_crane_tip))
    cabin_and_crane_tip_position_2D_array_northing = np.vstack((northing_cabin_position, northing_crane_tip))

    
    #
    # Then create the plots
    #

    
    if parameters.use_raster_data:
        
    
        #
        # On fertility class
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(sliced_fertility_class_as_array, extent = (sliced_fertility_class_data_min_easting - fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_max_easting + fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_min_northing - fertility_class_pixel_size_y/2.0, sliced_fertility_class_data_max_northing + fertility_class_pixel_size_y/2.0), cmap = parameters.fertility_class_cmap, norm = parameters.fertility_class_cmap_norm, alpha = parameters.fertility_class_visualization_alpha, zorder = -999)

        cbar = fig.colorbar(mappable, ticks = parameters.fertility_class_tick_locations, pad = 0.06, shrink = 0.8)
        cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
        cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
        cbar.ax.set_yticklabels(parameters.fertility_class_tick_labels, size = parameters.fertility_class_tick_label_font_size)

        h_crane_tip_positions = ax.scatter(easting_crane_tip, northing_crane_tip, c = 'black', s = parameters.stem_markersize_tiny, marker = 'o')
        h_cabin_positions = ax.scatter(easting_cabin_position, northing_cabin_position, edgecolors = 'black', facecolors = 'none', s = parameters.stem_markersize_large, marker = '*')

        ax.plot(cabin_and_crane_tip_position_2D_array_easting, cabin_and_crane_tip_position_2D_array_northing, '-', color = 'r', linewidth = parameters.cabin_to_crane_tip_position_line_width)

        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Machine cabin and stem positions', pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)

        ax.legend([h_crane_tip_positions, h_cabin_positions], ['Crane tip position', 'Cabin position'])

        fig.savefig('machine_cabin_and_stem_positions_on_fertility_class.png')
        plt.close(fig)



        #
        # On soil type
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(sliced_soil_type_as_array, extent = (sliced_soil_type_data_min_easting - soil_type_pixel_size_x/2.0, sliced_soil_type_data_max_easting + soil_type_pixel_size_x/2.0, sliced_soil_type_data_min_northing - soil_type_pixel_size_y/2.0, sliced_soil_type_data_max_northing + soil_type_pixel_size_y/2.0), cmap = parameters.soil_type_cmap, norm = parameters.soil_type_cmap_norm, alpha = parameters.soil_type_visualization_alpha, zorder = -999)

        cbar = fig.colorbar(mappable, ticks = parameters.soil_type_tick_locations, pad = 0.06, shrink = 0.8)
        cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
        cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
        cbar.ax.set_yticklabels(parameters.soil_type_tick_labels, size = parameters.soil_type_tick_label_font_size)

        h_crane_tip_positions = ax.scatter(easting_crane_tip, northing_crane_tip, c = 'black', s = parameters.stem_markersize_tiny, marker = 'o')
        h_cabin_positions = ax.scatter(easting_cabin_position, northing_cabin_position, edgecolors = 'black', facecolors = 'none', s = parameters.stem_markersize_large, marker = '*')

        ax.plot(cabin_and_crane_tip_position_2D_array_easting, cabin_and_crane_tip_position_2D_array_northing, '-', color = 'r', linewidth = parameters.cabin_to_crane_tip_position_line_width)

        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Machine cabin and stem positions', pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)

        ax.legend([h_crane_tip_positions, h_cabin_positions], ['Crane tip position', 'Cabin position'])

        fig.savefig('machine_cabin_and_stem_positions_on_soil_type.png')
        plt.close(fig)



        #
        # On terrain map
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(terrain_map_as_array, cmap = terrain_raster_cmap, norm = terrain_raster_cmap_norm, extent = (terrain_map_pixel_center_min_easting - terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_max_easting + terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_min_northing - terrain_pixel_size_in_y_direction/2.0, terrain_map_pixel_center_max_northing + terrain_pixel_size_in_y_direction/2.0), alpha = parameters.terrain_raster_map_alpha, zorder = -999)

        h_crane_tip_positions = ax.scatter(easting_crane_tip, northing_crane_tip, c = 'black', s = parameters.stem_markersize_tiny, marker = 'o')
        h_cabin_positions = ax.scatter(easting_cabin_position, northing_cabin_position, edgecolors = 'black', facecolors = 'none', s = parameters.stem_markersize_large, marker = '*')

        ax.plot(cabin_and_crane_tip_position_2D_array_easting, cabin_and_crane_tip_position_2D_array_northing, '-', color = 'r', linewidth = parameters.cabin_to_crane_tip_position_line_width)

        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Machine cabin and stem positions', pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)

        ax.legend([h_crane_tip_positions, h_cabin_positions], ['Crane tip position', 'Cabin position'])

        fig.savefig('machine_cabin_and_stem_positions_on_terrain_map.png')
        plt.close(fig)



    #
    # On white background
    #

    fig, ax = plt.subplots(figsize = parameters.figure_size)

    h_crane_tip_positions = ax.scatter(easting_crane_tip, northing_crane_tip, c = 'black', s = parameters.stem_markersize_tiny, marker = 'o')
    h_cabin_positions = ax.scatter(easting_cabin_position, northing_cabin_position, edgecolors = 'black', facecolors = 'none', s = parameters.stem_markersize_large, marker = '*')

    ax.plot(cabin_and_crane_tip_position_2D_array_easting, cabin_and_crane_tip_position_2D_array_northing, '-', color = 'r', linewidth = parameters.cabin_to_crane_tip_position_line_width)
    
    ax.set_aspect('equal', 'box')
    ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
    ax.set_title('Machine cabin and stem positions', pad = parameters.titlepad)
    ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
    ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
    
    ax.legend([h_crane_tip_positions, h_cabin_positions], ['Crane tip position', 'Cabin position'])

    fig.savefig('machine_cabin_and_stem_positions_on_white_background.png')
    plt.close(fig)


    
    #
    # All harvested stems, with color and marker corresponding to
    # species group, on the four different backgrounds
    #


    if parameters.use_raster_data:    

        
        #
        # On fertility class
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(sliced_fertility_class_as_array, extent = (sliced_fertility_class_data_min_easting - fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_max_easting + fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_min_northing - fertility_class_pixel_size_y/2.0, sliced_fertility_class_data_max_northing + fertility_class_pixel_size_y/2.0), cmap = parameters.fertility_class_cmap, norm = parameters.fertility_class_cmap_norm, alpha = parameters.fertility_class_visualization_alpha, zorder = -999)

        cbar = fig.colorbar(mappable, ticks = parameters.fertility_class_tick_locations, pad = 0.06, shrink = 0.8)
        cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
        cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
        cbar.ax.set_yticklabels(parameters.fertility_class_tick_labels, size = parameters.fertility_class_tick_label_font_size)

        h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = "blue", s = parameters.stem_markersize_medium, marker = 'o')
        h_spruce = ax.scatter(easting_crane_tip[species_group_id == 2], northing_crane_tip[species_group_id == 2], c = "magenta", s = parameters.stem_markersize_medium, marker = '*')
        h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = "green", s = parameters.stem_markersize_medium, marker = 'd')
        h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = "black", s = parameters.stem_markersize_medium, marker = '^')

        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Stem positions', pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend([h_pine, h_spruce, h_birch, h_other], ["Pine", "Spruce", "Birch", "Other"])

        fig.savefig('stem_positions_marked_by_species_on_fertility_class.png')
        plt.close(fig)



        #
        # On soil type
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(sliced_soil_type_as_array, extent = (sliced_soil_type_data_min_easting - soil_type_pixel_size_x/2.0, sliced_soil_type_data_max_easting + soil_type_pixel_size_x/2.0, sliced_soil_type_data_min_northing - soil_type_pixel_size_y/2.0, sliced_soil_type_data_max_northing + soil_type_pixel_size_y/2.0), cmap = parameters.soil_type_cmap, norm = parameters.soil_type_cmap_norm, alpha = parameters.soil_type_visualization_alpha, zorder = -999)

        cbar = fig.colorbar(mappable, ticks = parameters.soil_type_tick_locations, pad = 0.06, shrink = 0.8)
        cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
        cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
        cbar.ax.set_yticklabels(parameters.soil_type_tick_labels, size = parameters.soil_type_tick_label_font_size)

        h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = "blue", s = parameters.stem_markersize_medium, marker = 'o')
        h_spruce = ax.scatter(easting_crane_tip[species_group_id == 2], northing_crane_tip[species_group_id == 2], c = "magenta", s = parameters.stem_markersize_medium, marker = '*')
        h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = "green", s = parameters.stem_markersize_medium, marker = 'd')
        h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = "black", s = parameters.stem_markersize_medium, marker = '^')

        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Stem positions', pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend([h_pine, h_spruce, h_birch, h_other], ["Pine", "Spruce", "Birch", "Other"])

        fig.savefig('stem_positions_marked_by_species_on_soil_type.png')
        plt.close(fig)



        #
        # On terrain map
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(terrain_map_as_array, cmap = terrain_raster_cmap, norm = terrain_raster_cmap_norm, extent = (terrain_map_pixel_center_min_easting - terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_max_easting + terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_min_northing - terrain_pixel_size_in_y_direction/2.0, terrain_map_pixel_center_max_northing + terrain_pixel_size_in_y_direction/2.0), alpha = parameters.terrain_raster_map_alpha, zorder = -999)

        h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = "blue", s = parameters.stem_markersize_medium, marker = 'o')
        h_spruce = ax.scatter(easting_crane_tip[species_group_id == 2], northing_crane_tip[species_group_id == 2], c = "magenta", s = parameters.stem_markersize_medium, marker = '*')
        h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = "green", s = parameters.stem_markersize_medium, marker = 'd')
        h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = "black", s = parameters.stem_markersize_medium, marker = '^')

        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Stem positions', pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend([h_pine, h_spruce, h_birch, h_other], ["Pine", "Spruce", "Birch", "Other"])

        fig.savefig('stem_positions_marked_by_species_on_terrain_map.png')
        plt.close(fig)


    
    #
    # On white background
    #
    
    fig, ax = plt.subplots(figsize = parameters.figure_size)

    h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = "blue", s = parameters.stem_markersize_medium, marker = 'o')
    h_spruce = ax.scatter(easting_crane_tip[species_group_id == 2], northing_crane_tip[species_group_id == 2], c = "magenta", s = parameters.stem_markersize_medium, marker = '*')
    h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = "green", s = parameters.stem_markersize_medium, marker = 'd')
    h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = "black", s = parameters.stem_markersize_medium, marker = '^')

    ax.set_aspect('equal', 'box')
    ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
    ax.set_title('Stem positions', pad = parameters.titlepad)
    ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
    ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
    ax.legend([h_pine, h_spruce, h_birch, h_other], ["Pine", "Spruce", "Birch", "Other"])
    
    fig.savefig('stem_positions_marked_by_species_on_white_background.png')
    plt.close(fig)



    #
    # Spruce stems plotted by the value of rotten_200_with_sawlog, on the four different backgrounds
    #


    if parameters.use_raster_data:
    
    
        #
        # On fertility class
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(sliced_fertility_class_as_array, extent = (sliced_fertility_class_data_min_easting - fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_max_easting + fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_min_northing - fertility_class_pixel_size_y/2.0, sliced_fertility_class_data_max_northing + fertility_class_pixel_size_y/2.0), cmap = parameters.fertility_class_cmap, norm = parameters.fertility_class_cmap_norm, alpha = parameters.fertility_class_visualization_alpha, zorder = -999)

        cbar = fig.colorbar(mappable, ticks = parameters.fertility_class_tick_locations, pad = 0.06, shrink = 0.8)
        cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
        cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
        cbar.ax.set_yticklabels(parameters.fertility_class_tick_labels, size = parameters.fertility_class_tick_label_font_size)

        h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)

        h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')

        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Spruce stem positions, rot percent = %3.1f' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems), pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend([h_spruce_rot_status_one, h_spruce_rot_status_zero], ["Rot status 1", "Rot status 0"])

        fig.savefig('spruce_stem_positions_and_rotten_200_with_sawlog_on_fertility_class.png')
        plt.close(fig)



        #
        # On soil type
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(sliced_soil_type_as_array, extent = (sliced_soil_type_data_min_easting - soil_type_pixel_size_x/2.0, sliced_soil_type_data_max_easting + soil_type_pixel_size_x/2.0, sliced_soil_type_data_min_northing - soil_type_pixel_size_y/2.0, sliced_soil_type_data_max_northing + soil_type_pixel_size_y/2.0), cmap = parameters.soil_type_cmap, norm = parameters.soil_type_cmap_norm, alpha = parameters.soil_type_visualization_alpha, zorder = -999)

        cbar = fig.colorbar(mappable, ticks = parameters.soil_type_tick_locations, pad = 0.06, shrink = 0.8)
        cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
        cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
        cbar.ax.set_yticklabels(parameters.soil_type_tick_labels, size = parameters.soil_type_tick_label_font_size)

        h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)

        h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')

        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Spruce stem positions, rot percent = %3.1f' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems), pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend([h_spruce_rot_status_one, h_spruce_rot_status_zero], ["Rot status 1", "Rot status 0"])

        fig.savefig('spruce_stem_positions_and_rotten_200_with_sawlog_on_soil_type.png')
        plt.close(fig)



        #
        # On terrain map
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(terrain_map_as_array, cmap = terrain_raster_cmap, norm = terrain_raster_cmap_norm, extent = (terrain_map_pixel_center_min_easting - terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_max_easting + terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_min_northing - terrain_pixel_size_in_y_direction/2.0, terrain_map_pixel_center_max_northing + terrain_pixel_size_in_y_direction/2.0), alpha = parameters.terrain_raster_map_alpha, zorder = -999)

        h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)

        h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')

        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Spruce stem positions, rot percent = %3.1f' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems), pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend([h_spruce_rot_status_one, h_spruce_rot_status_zero], ["Rot status 1", "Rot status 0"])

        fig.savefig('spruce_stem_positions_and_rotten_200_with_sawlog_on_terrain_map.png')
        plt.close(fig)


    
    #
    # On white background
    #
    
    fig, ax = plt.subplots(figsize = parameters.figure_size)

    h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)
    
    h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')

    ax.set_aspect('equal', 'box')
    ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
    ax.set_title('Spruce stem positions, rot percent = %3.1f' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems), pad = parameters.titlepad)
    ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
    ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
    ax.legend([h_spruce_rot_status_one, h_spruce_rot_status_zero], ["Rot status 1", "Rot status 0"])
    
    fig.savefig('spruce_stem_positions_and_rotten_200_with_sawlog_on_white_background.png')
    plt.close(fig)



    #
    # Spruce stems plotted by caliber and the value of rotten_200_with_sawlog, on the four backgrounds
    #


    if parameters.use_raster_data:
    

        #
        # On fertility class
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(sliced_fertility_class_as_array, extent = (sliced_fertility_class_data_min_easting - fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_max_easting + fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_min_northing - fertility_class_pixel_size_y/2.0, sliced_fertility_class_data_max_northing + fertility_class_pixel_size_y/2.0), cmap = parameters.fertility_class_cmap, norm = parameters.fertility_class_cmap_norm, alpha = parameters.fertility_class_visualization_alpha, zorder = -999)

        cbar = fig.colorbar(mappable, ticks = parameters.fertility_class_tick_locations, pad = 0.06, shrink = 0.8)
        cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
        cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
        cbar.ax.set_yticklabels(parameters.fertility_class_tick_labels, size = parameters.fertility_class_tick_label_font_size)

        h_spruce_rotten_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], c = 'black', s = parameters.stem_markersize_medium, marker = 's', alpha = parameters.soil_type_visualization_alpha)

        h_spruce_healthy_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], edgecolors = 'black', facecolors = 'none', s = parameters.stem_markersize_medium, marker = 's')

        h_spruce_lower_than_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], northing_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], c = 'black', s = parameters.stem_markersize_medium, marker = '*')

        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Spruce stem positions, rot percent = %3.1f' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems), pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend([h_spruce_rotten_sawlog_caliber, h_spruce_healthy_sawlog_caliber, h_spruce_lower_than_sawlog_caliber], ["Rotten, sawlog-caliber", "Healthy, sawlog-caliber", "Lower than sawlog-caliber"])

        fig.savefig('spruce_stem_positions_and_caliber_and_rotten_200_with_sawlog_on_fertility_class.png')
        plt.close(fig)



        #
        # On soil type
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(sliced_soil_type_as_array, extent = (sliced_soil_type_data_min_easting - soil_type_pixel_size_x/2.0, sliced_soil_type_data_max_easting + soil_type_pixel_size_x/2.0, sliced_soil_type_data_min_northing - soil_type_pixel_size_y/2.0, sliced_soil_type_data_max_northing + soil_type_pixel_size_y/2.0), cmap = parameters.soil_type_cmap, norm = parameters.soil_type_cmap_norm, alpha = parameters.soil_type_visualization_alpha, zorder = -999)

        cbar = fig.colorbar(mappable, ticks = parameters.soil_type_tick_locations, pad = 0.06, shrink = 0.8)
        cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
        cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
        cbar.ax.set_yticklabels(parameters.soil_type_tick_labels, size = parameters.soil_type_tick_label_font_size)

        h_spruce_rotten_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], c = 'black', s = parameters.stem_markersize_medium, marker = 's', alpha = parameters.soil_type_visualization_alpha)

        h_spruce_healthy_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], edgecolors = 'black', facecolors = 'none', s = parameters.stem_markersize_medium, marker = 's')

        h_spruce_lower_than_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], northing_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], c = 'black', s = parameters.stem_markersize_medium, marker = '*')

        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Spruce stem positions, rot percent = %3.1f' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems), pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend([h_spruce_rotten_sawlog_caliber, h_spruce_healthy_sawlog_caliber, h_spruce_lower_than_sawlog_caliber], ["Rotten, sawlog-caliber", "Healthy, sawlog-caliber", "Lower than sawlog-caliber"])

        fig.savefig('spruce_stem_positions_and_caliber_and_rotten_200_with_sawlog_on_soil_type.png')
        plt.close(fig)



        #
        # On terrain map
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(terrain_map_as_array, cmap = terrain_raster_cmap, norm = terrain_raster_cmap_norm, extent = (terrain_map_pixel_center_min_easting - terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_max_easting + terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_min_northing - terrain_pixel_size_in_y_direction/2.0, terrain_map_pixel_center_max_northing + terrain_pixel_size_in_y_direction/2.0), alpha = parameters.terrain_raster_map_alpha, zorder = -999)

        h_spruce_rotten_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], c = 'black', s = parameters.stem_markersize_medium, marker = 's', alpha = parameters.soil_type_visualization_alpha)

        h_spruce_healthy_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], edgecolors = 'black', facecolors = 'none', s = parameters.stem_markersize_medium, marker = 's')

        h_spruce_lower_than_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], northing_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], c = 'black', s = parameters.stem_markersize_medium, marker = '*')

        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Spruce stem positions, rot percent = %3.1f' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems), pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend([h_spruce_rotten_sawlog_caliber, h_spruce_healthy_sawlog_caliber, h_spruce_lower_than_sawlog_caliber], ["Rotten, sawlog-caliber", "Healthy, sawlog-caliber", "Lower than sawlog-caliber"])

        fig.savefig('spruce_stem_positions_and_caliber_and_rotten_200_with_sawlog_on_terrain_map.png')
        plt.close(fig)

    

    #
    # On white background
    #
    
    fig, ax = plt.subplots(figsize = parameters.figure_size)

    h_spruce_rotten_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], c = 'black', s = parameters.stem_markersize_medium, marker = 's', alpha = parameters.soil_type_visualization_alpha)

    h_spruce_healthy_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], edgecolors = 'black', facecolors = 'none', s = parameters.stem_markersize_medium, marker = 's')

    h_spruce_lower_than_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], northing_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], c = 'black', s = parameters.stem_markersize_medium, marker = '*')
    
    ax.set_aspect('equal', 'box')
    ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
    ax.set_title('Spruce stem positions, rot percent = %3.1f' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems), pad = parameters.titlepad)
    ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
    ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
    ax.legend([h_spruce_rotten_sawlog_caliber, h_spruce_healthy_sawlog_caliber, h_spruce_lower_than_sawlog_caliber], ["Rotten, sawlog-caliber", "Healthy, sawlog-caliber", "Lower than sawlog-caliber"])
    
    fig.savefig('spruce_stem_positions_and_caliber_and_rotten_200_with_sawlog_on_white_background.png')
    plt.close(fig)


    
    #
    # The total rotten area, with sawlog-caliber rotten spruce stems
    # marked in one way and all other stems marked in another way, on
    # the four different backgrounds
    #


    if parameters.use_raster_data:

        
        #
        # On fertility class
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(sliced_fertility_class_as_array, extent = (sliced_fertility_class_data_min_easting - fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_max_easting + fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_min_northing - fertility_class_pixel_size_y/2.0, sliced_fertility_class_data_max_northing + fertility_class_pixel_size_y/2.0), cmap = parameters.fertility_class_cmap, norm = parameters.fertility_class_cmap_norm, alpha = parameters.fertility_class_visualization_alpha, zorder = -999)

        cbar = fig.colorbar(mappable, ticks = parameters.fertility_class_tick_locations, pad = 0.06, shrink = 0.8)
        cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
        cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
        cbar.ax.set_yticklabels(parameters.fertility_class_tick_labels, size = parameters.fertility_class_tick_label_font_size)

        h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'black', s = parameters.stem_markersize_small, marker = '.')

        h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)


        #
        # Add the buffer(0) call here to get the "holes" to display correctly
        #

        if isinstance(total_rot_area_delineation.buffer(0), MultiPolygon):

            for this_polygon in total_rot_area_delineation.buffer(0).geoms:

                h_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha, parameters.total_rot_area_delineation_color, 'k'))

        else:

            h_delineation = ax.add_patch(create_polygon_pathpatch(total_rot_area_delineation.buffer(0), parameters.delineation_visualization_alpha, parameters.total_rot_area_delineation_color, 'k'))


        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Rotten area', pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend([h_spruce_rot_status_one, h_spruce_rot_status_zero, h_delineation], ["Spruce, rot status one", "All other stems", "Rotten area"])

        fig.savefig('rotten_area_on_fertility_class.png')
        plt.close(fig)



        #
        # On soil type
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(sliced_soil_type_as_array, extent = (sliced_soil_type_data_min_easting - soil_type_pixel_size_x/2.0, sliced_soil_type_data_max_easting + soil_type_pixel_size_x/2.0, sliced_soil_type_data_min_northing - soil_type_pixel_size_y/2.0, sliced_soil_type_data_max_northing + soil_type_pixel_size_y/2.0), cmap = parameters.soil_type_cmap, norm = parameters.soil_type_cmap_norm, alpha = parameters.soil_type_visualization_alpha, zorder = -999)

        cbar = fig.colorbar(mappable, ticks = parameters.soil_type_tick_locations, pad = 0.06, shrink = 0.8)
        cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
        cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
        cbar.ax.set_yticklabels(parameters.soil_type_tick_labels, size = parameters.soil_type_tick_label_font_size)

        h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'black', s = parameters.stem_markersize_small, marker = '.')

        h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)


        #
        # Add the buffer(0) call here to get the "holes" to display correctly
        #

        if isinstance(total_rot_area_delineation.buffer(0), MultiPolygon):

            for this_polygon in total_rot_area_delineation.buffer(0).geoms:

                h_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha, parameters.total_rot_area_delineation_color, 'k'))

        else:

            h_delineation = ax.add_patch(create_polygon_pathpatch(total_rot_area_delineation.buffer(0), parameters.delineation_visualization_alpha, parameters.total_rot_area_delineation_color, 'k'))


        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Rotten area', pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend([h_spruce_rot_status_one, h_spruce_rot_status_zero, h_delineation], ["Spruce, rot status one", "All other stems", "Rotten area"])

        fig.savefig('rotten_area_on_soil_type.png')
        plt.close(fig)



        #
        # On terrain map
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(terrain_map_as_array, cmap = terrain_raster_cmap, norm = terrain_raster_cmap_norm, extent = (terrain_map_pixel_center_min_easting - terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_max_easting + terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_min_northing - terrain_pixel_size_in_y_direction/2.0, terrain_map_pixel_center_max_northing + terrain_pixel_size_in_y_direction/2.0), alpha = parameters.terrain_raster_map_alpha, zorder = -999)

        h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'black', s = parameters.stem_markersize_small, marker = '.')

        h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)


        #
        # Add the buffer(0) call here to get the "holes" to display correctly
        #

        if isinstance(total_rot_area_delineation.buffer(0), MultiPolygon):

            for this_polygon in total_rot_area_delineation.buffer(0).geoms:

                h_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha, parameters.total_rot_area_delineation_color, 'k'))

        else:

            h_delineation = ax.add_patch(create_polygon_pathpatch(total_rot_area_delineation.buffer(0), parameters.delineation_visualization_alpha, parameters.total_rot_area_delineation_color, 'k'))


        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Rotten area', pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend([h_spruce_rot_status_one, h_spruce_rot_status_zero, h_delineation], ["Spruce, rot status one", "All other stems", "Rotten area"])

        fig.savefig('rotten_area_on_terrain_map.png')
        plt.close(fig)



    #
    # On white background
    #

    fig, ax = plt.subplots(figsize = parameters.figure_size)
    
    h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'black', s = parameters.stem_markersize_small, marker = '.')
    h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')
    h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'black', s = parameters.stem_markersize_small, marker = '.')
    h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'black', s = parameters.stem_markersize_small, marker = '.')

    h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)


    #
    # Add the buffer(0) call here to get the "holes" to display correctly
    #

    if isinstance(total_rot_area_delineation.buffer(0), MultiPolygon):

        for this_polygon in total_rot_area_delineation.buffer(0).geoms:

            h_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha, parameters.total_rot_area_delineation_color, 'k'))

    else:

        h_delineation = ax.add_patch(create_polygon_pathpatch(total_rot_area_delineation.buffer(0), parameters.delineation_visualization_alpha, parameters.total_rot_area_delineation_color, 'k'))
        
        
    ax.set_aspect('equal', 'box')
    ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
    ax.set_title('Rotten area', pad = parameters.titlepad)
    ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
    ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
    ax.legend([h_spruce_rot_status_one, h_spruce_rot_status_zero, h_delineation], ["Spruce, rot status one", "All other stems", "Rotten area"])

    fig.savefig('rotten_area_on_white_background.png')
    plt.close(fig)
    
    

    #
    # Harvest area outside of rot areas, with sawlog-caliber rotten
    # spruce stems marked in one way and all other stems marked in
    # another way, on the four different backgrounds
    #


    if parameters.use_raster_data:
        

        #
        # On fertility class
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(sliced_fertility_class_as_array, extent = (sliced_fertility_class_data_min_easting - fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_max_easting + fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_min_northing - fertility_class_pixel_size_y/2.0, sliced_fertility_class_data_max_northing + fertility_class_pixel_size_y/2.0), cmap = parameters.fertility_class_cmap, norm = parameters.fertility_class_cmap_norm, alpha = parameters.fertility_class_visualization_alpha, zorder = -999)

        cbar = fig.colorbar(mappable, ticks = parameters.fertility_class_tick_locations, pad = 0.06, shrink = 0.8)
        cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
        cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
        cbar.ax.set_yticklabels(parameters.fertility_class_tick_labels, size = parameters.fertility_class_tick_label_font_size)

        h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'black', s = parameters.stem_markersize_small, marker = '.')

        h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)


        #
        # Add the buffer(0) call here to get the "holes" to display correctly
        #

        if isinstance(total_healthy_area_delineation.buffer(0), MultiPolygon):

            for this_polygon in total_healthy_area_delineation.buffer(0).geoms:

                h_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_alternative_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))

        else:

            h_delineation = ax.add_patch(create_polygon_pathpatch(total_healthy_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_alternative_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))


        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Harvest area outside of rot areas', pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend([h_spruce_rot_status_one, h_spruce_rot_status_zero, h_delineation], ["Spruce, rot status one", "All other stems", "Harvest area outside of rot areas"])

        fig.savefig('harvest_area_outside_of_rot_areas_on_fertility_class.png')
        plt.close(fig)



        #
        # On soil type
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(sliced_soil_type_as_array, extent = (sliced_soil_type_data_min_easting - soil_type_pixel_size_x/2.0, sliced_soil_type_data_max_easting + soil_type_pixel_size_x/2.0, sliced_soil_type_data_min_northing - soil_type_pixel_size_y/2.0, sliced_soil_type_data_max_northing + soil_type_pixel_size_y/2.0), cmap = parameters.soil_type_cmap, norm = parameters.soil_type_cmap_norm, alpha = parameters.soil_type_visualization_alpha, zorder = -999)

        cbar = fig.colorbar(mappable, ticks = parameters.soil_type_tick_locations, pad = 0.06, shrink = 0.8)
        cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
        cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
        cbar.ax.set_yticklabels(parameters.soil_type_tick_labels, size = parameters.soil_type_tick_label_font_size)

        h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'black', s = parameters.stem_markersize_small, marker = '.')

        h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)


        #
        # Add the buffer(0) call here to get the "holes" to display correctly
        #

        if isinstance(total_healthy_area_delineation.buffer(0), MultiPolygon):

            for this_polygon in total_healthy_area_delineation.buffer(0).geoms:

                h_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_alternative_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))

        else:

            h_delineation = ax.add_patch(create_polygon_pathpatch(total_healthy_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_alternative_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))


        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Harvest area outside of rot areas', pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend([h_spruce_rot_status_one, h_spruce_rot_status_zero, h_delineation], ["Spruce, rot status one", "All other stems", "Harvest area outside of rot areas"])

        fig.savefig('harvest_area_outside_of_rot_areas_on_soil_type.png')
        plt.close(fig)



        #
        # On terrain map
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(terrain_map_as_array, cmap = terrain_raster_cmap, norm = terrain_raster_cmap_norm, extent = (terrain_map_pixel_center_min_easting - terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_max_easting + terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_min_northing - terrain_pixel_size_in_y_direction/2.0, terrain_map_pixel_center_max_northing + terrain_pixel_size_in_y_direction/2.0), alpha = parameters.terrain_raster_map_alpha, zorder = -999)

        h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'black', s = parameters.stem_markersize_small, marker = '.')

        h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)


        #
        # Add the buffer(0) call here to get the "holes" to display correctly
        #

        if isinstance(total_healthy_area_delineation.buffer(0), MultiPolygon):

            for this_polygon in total_healthy_area_delineation.buffer(0).geoms:

                h_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_alternative_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))

        else:

            h_delineation = ax.add_patch(create_polygon_pathpatch(total_healthy_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_alternative_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))


        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Harvest area outside of rot areas', pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend([h_spruce_rot_status_one, h_spruce_rot_status_zero, h_delineation], ["Spruce, rot status one", "All other stems", "Harvest area outside of rot areas"])

        fig.savefig('harvest_area_outside_of_rot_areas_on_terrain_map.png')
        plt.close(fig)



    #
    # On white background
    #

    fig, ax = plt.subplots(figsize = parameters.figure_size)
    
    h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'black', s = parameters.stem_markersize_small, marker = '.')
    h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')
    h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'black', s = parameters.stem_markersize_small, marker = '.')
    h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'black', s = parameters.stem_markersize_small, marker = '.')

    h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)


    #
    # Add the buffer(0) call here to get the "holes" to display correctly
    #

    if isinstance(total_healthy_area_delineation.buffer(0), MultiPolygon):

        for this_polygon in total_healthy_area_delineation.buffer(0).geoms:

            h_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_alternative_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))

    else:

        h_delineation = ax.add_patch(create_polygon_pathpatch(total_healthy_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_alternative_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))
        
        
    ax.set_aspect('equal', 'box')
    ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
    ax.set_title('Harvest area outside of rot areas', pad = parameters.titlepad)
    ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
    ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
    ax.legend([h_spruce_rot_status_one, h_spruce_rot_status_zero, h_delineation], ["Spruce, rot status one", "All other stems", "Harvest area outside of rot areas"])

    fig.savefig('harvest_area_outside_of_rot_areas_on_white_background.png')
    plt.close(fig)
    
    
    
    #
    # Full harvest area delineation, with sawlog-caliber rotten spruce
    # stems marked in one way and all other stems marked in another
    # way, on the four different backgrounds
    #
    

    if parameters.use_raster_data:

        
        #
        # On fertility class
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(sliced_fertility_class_as_array, extent = (sliced_fertility_class_data_min_easting - fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_max_easting + fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_min_northing - fertility_class_pixel_size_y/2.0, sliced_fertility_class_data_max_northing + fertility_class_pixel_size_y/2.0), cmap = parameters.fertility_class_cmap, norm = parameters.fertility_class_cmap_norm, alpha = parameters.fertility_class_visualization_alpha, zorder = -999)

        cbar = fig.colorbar(mappable, ticks = parameters.fertility_class_tick_locations, pad = 0.06, shrink = 0.8)
        cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
        cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
        cbar.ax.set_yticklabels(parameters.fertility_class_tick_labels, size = parameters.fertility_class_tick_label_font_size)

        h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'black', s = parameters.stem_markersize_small, marker = '.')

        h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)


        h_delineation = ax.add_patch(create_polygon_pathpatch(full_harvest_area_delineation, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_alternative_color, 'k'))


        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Full harvest area delineation', pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend([h_spruce_rot_status_one, h_spruce_rot_status_zero, h_delineation], ["Spruce, rot status one", "All other stems", "Full harvest area"])

        fig.savefig('full_harvest_area_delineation_on_fertility_class.png')
        plt.close(fig)



        #
        # On soil type
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(sliced_soil_type_as_array, extent = (sliced_soil_type_data_min_easting - soil_type_pixel_size_x/2.0, sliced_soil_type_data_max_easting + soil_type_pixel_size_x/2.0, sliced_soil_type_data_min_northing - soil_type_pixel_size_y/2.0, sliced_soil_type_data_max_northing + soil_type_pixel_size_y/2.0), cmap = parameters.soil_type_cmap, norm = parameters.soil_type_cmap_norm, alpha = parameters.soil_type_visualization_alpha, zorder = -999)

        cbar = fig.colorbar(mappable, ticks = parameters.soil_type_tick_locations, pad = 0.06, shrink = 0.8)
        cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
        cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
        cbar.ax.set_yticklabels(parameters.soil_type_tick_labels, size = parameters.soil_type_tick_label_font_size)

        h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'black', s = parameters.stem_markersize_small, marker = '.')

        h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)


        h_delineation = ax.add_patch(create_polygon_pathpatch(full_harvest_area_delineation, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_alternative_color, 'k'))


        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Full harvest area delineation', pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend([h_spruce_rot_status_one, h_spruce_rot_status_zero, h_delineation], ["Spruce, rot status one", "All other stems", "Full harvest area"])

        fig.savefig('full_harvest_area_delineation_on_soil_type.png')
        plt.close(fig)



        #
        # On terrain map
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(terrain_map_as_array, cmap = terrain_raster_cmap, norm = terrain_raster_cmap_norm, extent = (terrain_map_pixel_center_min_easting - terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_max_easting + terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_min_northing - terrain_pixel_size_in_y_direction/2.0, terrain_map_pixel_center_max_northing + terrain_pixel_size_in_y_direction/2.0), alpha = parameters.terrain_raster_map_alpha, zorder = -999)

        h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'black', s = parameters.stem_markersize_small, marker = '.')

        h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)


        h_delineation = ax.add_patch(create_polygon_pathpatch(full_harvest_area_delineation, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_alternative_color, 'k'))


        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Full harvest area delineation', pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend([h_spruce_rot_status_one, h_spruce_rot_status_zero, h_delineation], ["Spruce, rot status one", "All other stems", "Full harvest area"])

        fig.savefig('full_harvest_area_delineation_on_terrain_map.png')
        plt.close(fig)

    

    #
    # On white background
    #

    fig, ax = plt.subplots(figsize = parameters.figure_size)
    
    h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'black', s = parameters.stem_markersize_small, marker = '.')
    h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')
    h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'black', s = parameters.stem_markersize_small, marker = '.')
    h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'black', s = parameters.stem_markersize_small, marker = '.')

    h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)

    
    h_delineation = ax.add_patch(create_polygon_pathpatch(full_harvest_area_delineation, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_alternative_color, 'k'))
        
        
    ax.set_aspect('equal', 'box')
    ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
    ax.set_title('Full harvest area delineation', pad = parameters.titlepad)
    ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
    ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
    ax.legend([h_spruce_rot_status_one, h_spruce_rot_status_zero, h_delineation], ["Spruce, rot status one", "All other stems", "Full harvest area"])

    fig.savefig('full_harvest_area_delineation_on_white_background.png')
    plt.close(fig)


    
    #
    # The final segmentation result for the entire stand, with the
    # total healthy and total rotten area shown, with sawlog-caliber
    # rotten spruce stems marked in one way and all other stems marked
    # in another way, on the four different backgrounds
    #


    if parameters.use_raster_data:


        #
        # On fertility class
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(sliced_fertility_class_as_array, extent = (sliced_fertility_class_data_min_easting - fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_max_easting + fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_min_northing - fertility_class_pixel_size_y/2.0, sliced_fertility_class_data_max_northing + fertility_class_pixel_size_y/2.0), cmap = parameters.fertility_class_cmap, norm = parameters.fertility_class_cmap_norm, alpha = parameters.fertility_class_visualization_alpha, zorder = -999)

        cbar = fig.colorbar(mappable, ticks = parameters.fertility_class_tick_locations, pad = 0.06, shrink = 0.8)
        cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
        cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
        cbar.ax.set_yticklabels(parameters.fertility_class_tick_labels, size = parameters.fertility_class_tick_label_font_size)

        h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'black', s = parameters.stem_markersize_small, marker = '.')

        h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)

        legend_handles = [h_spruce_rot_status_one, h_spruce_rot_status_zero]
        legend_strings = ["Spruce, rot status one", "All other stems"]


        #
        # Add the buffer(0) call here to get the "holes" to display correctly
        #

        if isinstance(total_healthy_area_delineation.buffer(0), MultiPolygon):

            for this_polygon in total_healthy_area_delineation.buffer(0).geoms:

                h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))

        else:

            h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(total_healthy_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))

        legend_handles.append(h_total_healthy_area_delineation)

        legend_strings.append("Harvest area outside of rot areas")


        #
        # Add the buffer(0) call here to get the "holes" to display correctly
        #

        if isinstance(total_rot_area_delineation.buffer(0), MultiPolygon):

            for this_polygon in total_rot_area_delineation.buffer(0).geoms:

                h_total_rot_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.total_rot_area_delineation_color, 'k'))

        else:

            h_total_rot_area_delineation = ax.add_patch(create_polygon_pathpatch(total_rot_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.total_rot_area_delineation_color, 'k'))

        legend_handles.append(h_total_rot_area_delineation)

        legend_strings.append("Rotten area")


        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Final segmentation result, $f_{rot}$ = %3.1f%%, $T_{sum}$ = %.1f d.d.' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems, temperature_sum_for_stand), pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend(legend_handles, legend_strings)

        fig.savefig('final_segmentation_result_with_total_rot_area_on_fertility_class.png')
        plt.close(fig)



        #
        # On soil type
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(sliced_soil_type_as_array, extent = (sliced_soil_type_data_min_easting - soil_type_pixel_size_x/2.0, sliced_soil_type_data_max_easting + soil_type_pixel_size_x/2.0, sliced_soil_type_data_min_northing - soil_type_pixel_size_y/2.0, sliced_soil_type_data_max_northing + soil_type_pixel_size_y/2.0), cmap = parameters.soil_type_cmap, norm = parameters.soil_type_cmap_norm, alpha = parameters.soil_type_visualization_alpha, zorder = -999)

        cbar = fig.colorbar(mappable, ticks = parameters.soil_type_tick_locations, pad = 0.06, shrink = 0.8)
        cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
        cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
        cbar.ax.set_yticklabels(parameters.soil_type_tick_labels, size = parameters.soil_type_tick_label_font_size)

        h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'black', s = parameters.stem_markersize_small, marker = '.')

        h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)

        legend_handles = [h_spruce_rot_status_one, h_spruce_rot_status_zero]
        legend_strings = ["Spruce, rot status one", "All other stems"]


        #
        # Add the buffer(0) call here to get the "holes" to display correctly
        #

        if isinstance(total_healthy_area_delineation.buffer(0), MultiPolygon):

            for this_polygon in total_healthy_area_delineation.buffer(0).geoms:

                h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))

        else:

            h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(total_healthy_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))

        legend_handles.append(h_total_healthy_area_delineation)

        legend_strings.append("Harvest area outside of rot areas")


        #
        # Add the buffer(0) call here to get the "holes" to display correctly
        #

        if isinstance(total_rot_area_delineation.buffer(0), MultiPolygon):

            for this_polygon in total_rot_area_delineation.buffer(0).geoms:

                h_total_rot_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.total_rot_area_delineation_color, 'k'))

        else:

            h_total_rot_area_delineation = ax.add_patch(create_polygon_pathpatch(total_rot_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.total_rot_area_delineation_color, 'k'))

        legend_handles.append(h_total_rot_area_delineation)

        legend_strings.append("Rotten area")


        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Final segmentation result, $f_{rot}$ = %3.1f%%, $T_{sum}$ = %.1f d.d.' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems, temperature_sum_for_stand), pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend(legend_handles, legend_strings)

        fig.savefig('final_segmentation_result_with_total_rot_area_on_soil_type.png')
        plt.close(fig)



        #
        # On terrain map
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(terrain_map_as_array, cmap = terrain_raster_cmap, norm = terrain_raster_cmap_norm, extent = (terrain_map_pixel_center_min_easting - terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_max_easting + terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_min_northing - terrain_pixel_size_in_y_direction/2.0, terrain_map_pixel_center_max_northing + terrain_pixel_size_in_y_direction/2.0), alpha = parameters.terrain_raster_map_alpha, zorder = -999)

        h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'black', s = parameters.stem_markersize_small, marker = '.')

        h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)

        legend_handles = [h_spruce_rot_status_one, h_spruce_rot_status_zero]
        legend_strings = ["Spruce, rot status one", "All other stems"]


        #
        # Add the buffer(0) call here to get the "holes" to display correctly
        #

        if isinstance(total_healthy_area_delineation.buffer(0), MultiPolygon):

            for this_polygon in total_healthy_area_delineation.buffer(0).geoms:

                h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))

        else:

            h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(total_healthy_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))

        legend_handles.append(h_total_healthy_area_delineation)

        legend_strings.append("Harvest area outside of rot areas")


        #
        # Add the buffer(0) call here to get the "holes" to display correctly
        #

        if isinstance(total_rot_area_delineation.buffer(0), MultiPolygon):

            for this_polygon in total_rot_area_delineation.buffer(0).geoms:

                h_total_rot_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.total_rot_area_delineation_color, 'k'))

        else:

            h_total_rot_area_delineation = ax.add_patch(create_polygon_pathpatch(total_rot_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.total_rot_area_delineation_color, 'k'))

        legend_handles.append(h_total_rot_area_delineation)

        legend_strings.append("Rotten area")



        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Final segmentation result, $f_{rot}$ = %3.1f%%, $T_{sum}$ = %.1f d.d.' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems, temperature_sum_for_stand), pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend(legend_handles, legend_strings)

        fig.savefig('final_segmentation_result_with_total_rot_area_on_terrain_map.png')
        plt.close(fig)



    #
    # On white background
    #
    
    fig, ax = plt.subplots(figsize = parameters.figure_size)
    
    h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'black', s = parameters.stem_markersize_small, marker = '.')
    h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')
    h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'black', s = parameters.stem_markersize_small, marker = '.')
    h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'black', s = parameters.stem_markersize_small, marker = '.')

    h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)

    legend_handles = [h_spruce_rot_status_one, h_spruce_rot_status_zero]
    legend_strings = ["Spruce, rot status one", "All other stems"]


    #
    # Add the buffer(0) call here to get the "holes" to display correctly
    #

    if isinstance(total_healthy_area_delineation.buffer(0), MultiPolygon):

        for this_polygon in total_healthy_area_delineation.buffer(0).geoms:

            h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))

    else:
        
        h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(total_healthy_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))
    
    legend_handles.append(h_total_healthy_area_delineation)

    legend_strings.append("Harvest area outside of rot areas")


    #
    # Add the buffer(0) call here to get the "holes" to display correctly
    #

    if isinstance(total_rot_area_delineation.buffer(0), MultiPolygon):

        for this_polygon in total_rot_area_delineation.buffer(0).geoms:

            h_total_rot_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.total_rot_area_delineation_color, 'k'))

    else:
        
        h_total_rot_area_delineation = ax.add_patch(create_polygon_pathpatch(total_rot_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.total_rot_area_delineation_color, 'k'))
    
    legend_handles.append(h_total_rot_area_delineation)

    legend_strings.append("Rotten area")


    ax.set_aspect('equal', 'box')
    ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
    ax.set_title('Final segmentation result, $f_{rot}$ = %3.1f%%, $T_{sum}$ = %.1f d.d.' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems, temperature_sum_for_stand), pad = parameters.titlepad)
    ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
    ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
    ax.legend(legend_handles, legend_strings)

    fig.savefig('final_segmentation_result_with_total_rot_area_on_white_background.png')
    plt.close(fig)



    #
    # The final segmentation result for the entire stand, with the
    # following indications for the stems (only spruce here), on the
    # four backgrounds:
    #
    # - Spruce, rotten sawlog-caliber stems -> filled black square
    # - Spruce, healthy sawlog-caliber stems -> open black square
    # - Spruce, lower than sawlog-caliber stems -> filled black star
    #


    if parameters.use_raster_data:

        
        #
        # On fertility class
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(sliced_fertility_class_as_array, extent = (sliced_fertility_class_data_min_easting - fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_max_easting + fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_min_northing - fertility_class_pixel_size_y/2.0, sliced_fertility_class_data_max_northing + fertility_class_pixel_size_y/2.0), cmap = parameters.fertility_class_cmap, norm = parameters.fertility_class_cmap_norm, alpha = parameters.fertility_class_visualization_alpha, zorder = -999)

        cbar = fig.colorbar(mappable, ticks = parameters.fertility_class_tick_locations, pad = 0.06, shrink = 0.8)
        cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
        cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
        cbar.ax.set_yticklabels(parameters.fertility_class_tick_labels, size = parameters.fertility_class_tick_label_font_size)


        h_spruce_rotten_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], c = 'black', s = parameters.stem_markersize_medium, marker = 's', alpha = parameters.soil_type_visualization_alpha)

        h_spruce_healthy_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], edgecolors = 'black', facecolors = 'none', s = parameters.stem_markersize_medium, marker = 's')

        h_spruce_lower_than_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], northing_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], c = 'black', s = parameters.stem_markersize_medium, marker = '*')


        legend_handles = [h_spruce_rotten_sawlog_caliber, h_spruce_healthy_sawlog_caliber, h_spruce_lower_than_sawlog_caliber]

        legend_strings = ["Rotten, sawlog-caliber", "Healthy, sawlog-caliber", "Lower than sawlog-caliber"]


        #
        # Add the buffer(0) call here to get the "holes" to display correctly
        #

        if isinstance(total_healthy_area_delineation.buffer(0), MultiPolygon):

            for this_polygon in total_healthy_area_delineation.buffer(0).geoms:

                h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))

        else:

            h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(total_healthy_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))

        legend_handles.append(h_total_healthy_area_delineation)

        legend_strings.append("Harvest area outside of rot areas")


        #
        # Add the buffer(0) call here to get the "holes" to display correctly
        #

        if isinstance(total_rot_area_delineation.buffer(0), MultiPolygon):

            for this_polygon in total_rot_area_delineation.buffer(0).geoms:

                h_total_rot_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.total_rot_area_delineation_color, 'k'))

        else:

            h_total_rot_area_delineation = ax.add_patch(create_polygon_pathpatch(total_rot_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.total_rot_area_delineation_color, 'k'))

        legend_handles.append(h_total_rot_area_delineation)

        legend_strings.append("Rotten area")


        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Final segmentation result, $f_{rot}$ = %3.1f%%, $T_{sum}$ = %.1f d.d.' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems, temperature_sum_for_stand), pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend(legend_handles, legend_strings)

        fig.savefig('final_segmentation_result_with_total_rot_area_and_spruce_stems_on_fertility_class.png')
        plt.close(fig)



        #
        # On soil type
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(sliced_soil_type_as_array, extent = (sliced_soil_type_data_min_easting - soil_type_pixel_size_x/2.0, sliced_soil_type_data_max_easting + soil_type_pixel_size_x/2.0, sliced_soil_type_data_min_northing - soil_type_pixel_size_y/2.0, sliced_soil_type_data_max_northing + soil_type_pixel_size_y/2.0), cmap = parameters.soil_type_cmap, norm = parameters.soil_type_cmap_norm, alpha = parameters.soil_type_visualization_alpha, zorder = -999)

        cbar = fig.colorbar(mappable, ticks = parameters.soil_type_tick_locations, pad = 0.06, shrink = 0.8)
        cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
        cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
        cbar.ax.set_yticklabels(parameters.soil_type_tick_labels, size = parameters.soil_type_tick_label_font_size)


        h_spruce_rotten_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], c = 'black', s = parameters.stem_markersize_medium, marker = 's', alpha = parameters.soil_type_visualization_alpha)

        h_spruce_healthy_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], edgecolors = 'black', facecolors = 'none', s = parameters.stem_markersize_medium, marker = 's')

        h_spruce_lower_than_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], northing_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], c = 'black', s = parameters.stem_markersize_medium, marker = '*')


        legend_handles = [h_spruce_rotten_sawlog_caliber, h_spruce_healthy_sawlog_caliber, h_spruce_lower_than_sawlog_caliber]

        legend_strings = ["Rotten, sawlog-caliber", "Healthy, sawlog-caliber", "Lower than sawlog-caliber"]


        #
        # Add the buffer(0) call here to get the "holes" to display correctly
        #

        if isinstance(total_healthy_area_delineation.buffer(0), MultiPolygon):

            for this_polygon in total_healthy_area_delineation.buffer(0).geoms:

                h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))

        else:

            h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(total_healthy_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))

        legend_handles.append(h_total_healthy_area_delineation)

        legend_strings.append("Harvest area outside of rot areas")


        #
        # Add the buffer(0) call here to get the "holes" to display correctly
        #

        if isinstance(total_rot_area_delineation.buffer(0), MultiPolygon):

            for this_polygon in total_rot_area_delineation.buffer(0).geoms:

                h_total_rot_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.total_rot_area_delineation_color, 'k'))

        else:

            h_total_rot_area_delineation = ax.add_patch(create_polygon_pathpatch(total_rot_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.total_rot_area_delineation_color, 'k'))

        legend_handles.append(h_total_rot_area_delineation)

        legend_strings.append("Rotten area")


        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Final segmentation result, $f_{rot}$ = %3.1f%%, $T_{sum}$ = %.1f d.d.' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems, temperature_sum_for_stand), pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend(legend_handles, legend_strings)

        fig.savefig('final_segmentation_result_with_total_rot_area_and_spruce_stems_on_soil_type.png')
        plt.close(fig)



        #
        # On terrain map
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        mappable = ax.imshow(terrain_map_as_array, cmap = terrain_raster_cmap, norm = terrain_raster_cmap_norm, extent = (terrain_map_pixel_center_min_easting - terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_max_easting + terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_min_northing - terrain_pixel_size_in_y_direction/2.0, terrain_map_pixel_center_max_northing + terrain_pixel_size_in_y_direction/2.0), alpha = parameters.terrain_raster_map_alpha, zorder = -999)


        h_spruce_rotten_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], c = 'black', s = parameters.stem_markersize_medium, marker = 's', alpha = parameters.soil_type_visualization_alpha)

        h_spruce_healthy_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], edgecolors = 'black', facecolors = 'none', s = parameters.stem_markersize_medium, marker = 's')

        h_spruce_lower_than_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], northing_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], c = 'black', s = parameters.stem_markersize_medium, marker = '*')


        legend_handles = [h_spruce_rotten_sawlog_caliber, h_spruce_healthy_sawlog_caliber, h_spruce_lower_than_sawlog_caliber]

        legend_strings = ["Rotten, sawlog-caliber", "Healthy, sawlog-caliber", "Lower than sawlog-caliber"]


        #
        # Add the buffer(0) call here to get the "holes" to display correctly
        #

        if isinstance(total_healthy_area_delineation.buffer(0), MultiPolygon):

            for this_polygon in total_healthy_area_delineation.buffer(0).geoms:

                h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))

        else:

            h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(total_healthy_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))

        legend_handles.append(h_total_healthy_area_delineation)

        legend_strings.append("Harvest area outside of rot areas")


        #
        # Add the buffer(0) call here to get the "holes" to display correctly
        #

        if isinstance(total_rot_area_delineation.buffer(0), MultiPolygon):

            for this_polygon in total_rot_area_delineation.buffer(0).geoms:

                h_total_rot_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.total_rot_area_delineation_color, 'k'))

        else:

            h_total_rot_area_delineation = ax.add_patch(create_polygon_pathpatch(total_rot_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.total_rot_area_delineation_color, 'k'))

        legend_handles.append(h_total_rot_area_delineation)

        legend_strings.append("Rotten area")


        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Final segmentation result, $f_{rot}$ = %3.1f%%, $T_{sum}$ = %.1f d.d.' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems, temperature_sum_for_stand), pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend(legend_handles, legend_strings)

        fig.savefig('final_segmentation_result_with_total_rot_area_and_spruce_stems_on_terrain_map.png')
        plt.close(fig)



    #
    # On white background
    #
    
    fig, ax = plt.subplots(figsize = parameters.figure_size)


    h_spruce_rotten_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], c = 'black', s = parameters.stem_markersize_medium, marker = 's', alpha = parameters.soil_type_visualization_alpha)

    h_spruce_healthy_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], edgecolors = 'black', facecolors = 'none', s = parameters.stem_markersize_medium, marker = 's')

    h_spruce_lower_than_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], northing_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], c = 'black', s = parameters.stem_markersize_medium, marker = '*')

    
    legend_handles = [h_spruce_rotten_sawlog_caliber, h_spruce_healthy_sawlog_caliber, h_spruce_lower_than_sawlog_caliber]

    legend_strings = ["Rotten, sawlog-caliber", "Healthy, sawlog-caliber", "Lower than sawlog-caliber"]
    
    
    #
    # Add the buffer(0) call here to get the "holes" to display correctly
    #

    if isinstance(total_healthy_area_delineation.buffer(0), MultiPolygon):

        for this_polygon in total_healthy_area_delineation.buffer(0).geoms:

            h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))

    else:
        
        h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(total_healthy_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k', parameters.thicker_line_linewidth_for_delineation_boundary))
    
    legend_handles.append(h_total_healthy_area_delineation)

    legend_strings.append("Harvest area outside of rot areas")


    #
    # Add the buffer(0) call here to get the "holes" to display correctly
    #

    if isinstance(total_rot_area_delineation.buffer(0), MultiPolygon):

        for this_polygon in total_rot_area_delineation.buffer(0).geoms:

            h_total_rot_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.total_rot_area_delineation_color, 'k'))

    else:
        
        h_total_rot_area_delineation = ax.add_patch(create_polygon_pathpatch(total_rot_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.total_rot_area_delineation_color, 'k'))
    
    legend_handles.append(h_total_rot_area_delineation)

    legend_strings.append("Rotten area")


    ax.set_aspect('equal', 'box')
    ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
    ax.set_title('Final segmentation result, $f_{rot}$ = %3.1f%%, $T_{sum}$ = %.1f d.d.' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems, temperature_sum_for_stand), pad = parameters.titlepad)
    ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
    ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
    ax.legend(legend_handles, legend_strings)

    fig.savefig('final_segmentation_result_with_total_rot_area_and_spruce_stems_on_white_background.png')
    plt.close(fig)
    
    

    #
    # Then, create the plots specific to dbscan
    #


    if delineation_method == 'dbscan':
    
    
        #
        # The rotten spruce stems colored by cluster index, on the four different backgrounds
        #


        if parameters.use_raster_data:

        
            #
            # On fertility class
            #

            fig, ax = plt.subplots(figsize = parameters.figure_size)

            mappable = ax.imshow(sliced_fertility_class_as_array, extent = (sliced_fertility_class_data_min_easting - fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_max_easting + fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_min_northing - fertility_class_pixel_size_y/2.0, sliced_fertility_class_data_max_northing + fertility_class_pixel_size_y/2.0), cmap = parameters.fertility_class_cmap, norm = parameters.fertility_class_cmap_norm, alpha = parameters.fertility_class_visualization_alpha, zorder = -999)

            cbar = fig.colorbar(mappable, ticks = parameters.fertility_class_tick_locations, pad = 0.06, shrink = 0.8)
            cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
            cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
            cbar.ax.set_yticklabels(parameters.fertility_class_tick_labels, size = parameters.fertility_class_tick_label_font_size)

            legend_handles = []
            legend_strings = []

            for this_cluster in position_and_cluster_id_data_for_each_cluster_including_outliers:

                if this_cluster[0, 2].astype(int) == -1:

                    this_stem_markersize = parameters.stem_markersize_large*0.3
                    this_stem_marker = 'o'

                else:

                    this_stem_markersize = parameters.stem_markersize_large
                    this_stem_marker = 's'

                this_h = ax.scatter(this_cluster[:, 0], this_cluster[:, 1], c = rot_cluster_color_map[this_cluster[:, 2].astype(int)], s = this_stem_markersize, marker = this_stem_marker)

                legend_handles.append(this_h)

                if this_cluster[0, 2].astype(int) == -1:

                    legend_strings.append("Outliers")

                else:

                    legend_strings.append("Rot cluster number " + str(this_cluster[0, 2].astype(int) + 1))

            ax.set_aspect('equal', 'box')
            ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
            ax.set_title('Clustering of rotten spruce stems via DBSCAN, rot percent = %3.1f' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems), pad = parameters.titlepad)
            ax.legend(legend_handles, legend_strings)
            ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
            ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)

            fig.savefig('clustering_result_for_rotten_spruce_stems_on_fertility_class.png')
            plt.close(fig)



            #
            # On soil type
            #

            fig, ax = plt.subplots(figsize = parameters.figure_size)

            mappable = ax.imshow(sliced_soil_type_as_array, extent = (sliced_soil_type_data_min_easting - soil_type_pixel_size_x/2.0, sliced_soil_type_data_max_easting + soil_type_pixel_size_x/2.0, sliced_soil_type_data_min_northing - soil_type_pixel_size_y/2.0, sliced_soil_type_data_max_northing + soil_type_pixel_size_y/2.0), cmap = parameters.soil_type_cmap, norm = parameters.soil_type_cmap_norm, alpha = parameters.soil_type_visualization_alpha, zorder = -999)

            cbar = fig.colorbar(mappable, ticks = parameters.soil_type_tick_locations, pad = 0.06, shrink = 0.8)
            cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
            cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
            cbar.ax.set_yticklabels(parameters.soil_type_tick_labels, size = parameters.soil_type_tick_label_font_size)

            legend_handles = []
            legend_strings = []

            for this_cluster in position_and_cluster_id_data_for_each_cluster_including_outliers:

                if this_cluster[0, 2].astype(int) == -1:

                    this_stem_markersize = parameters.stem_markersize_large*0.3
                    this_stem_marker = 'o'

                else:

                    this_stem_markersize = parameters.stem_markersize_large
                    this_stem_marker = 's'

                this_h = ax.scatter(this_cluster[:, 0], this_cluster[:, 1], c = rot_cluster_color_map[this_cluster[:, 2].astype(int)], s = this_stem_markersize, marker = this_stem_marker)

                legend_handles.append(this_h)

                if this_cluster[0, 2].astype(int) == -1:

                    legend_strings.append("Outliers")

                else:

                    legend_strings.append("Rot cluster number " + str(this_cluster[0, 2].astype(int) + 1))

            ax.set_aspect('equal', 'box')
            ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
            ax.set_title('Clustering of rotten spruce stems via DBSCAN, rot percent = %3.1f' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems), pad = parameters.titlepad)
            ax.legend(legend_handles, legend_strings)
            ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
            ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)

            fig.savefig('clustering_result_for_rotten_spruce_stems_on_soil_type.png')
            plt.close(fig)



            #
            # On terrain map
            #

            fig, ax = plt.subplots(figsize = parameters.figure_size)

            mappable = ax.imshow(terrain_map_as_array, cmap = terrain_raster_cmap, norm = terrain_raster_cmap_norm, extent = (terrain_map_pixel_center_min_easting - terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_max_easting + terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_min_northing - terrain_pixel_size_in_y_direction/2.0, terrain_map_pixel_center_max_northing + terrain_pixel_size_in_y_direction/2.0), alpha = parameters.terrain_raster_map_alpha, zorder = -999)

            legend_handles = []
            legend_strings = []

            for this_cluster in position_and_cluster_id_data_for_each_cluster_including_outliers:

                if this_cluster[0, 2].astype(int) == -1:

                    this_stem_markersize = parameters.stem_markersize_large*0.3
                    this_stem_marker = 'o'

                else:

                    this_stem_markersize = parameters.stem_markersize_large
                    this_stem_marker = 's'

                this_h = ax.scatter(this_cluster[:, 0], this_cluster[:, 1], c = rot_cluster_color_map[this_cluster[:, 2].astype(int)], s = this_stem_markersize, marker = this_stem_marker)

                legend_handles.append(this_h)

                if this_cluster[0, 2].astype(int) == -1:

                    legend_strings.append("Outliers")

                else:

                    legend_strings.append("Rot cluster number " + str(this_cluster[0, 2].astype(int) + 1))

            ax.set_aspect('equal', 'box')
            ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
            ax.set_title('Clustering of rotten spruce stems via DBSCAN, rot percent = %3.1f' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems), pad = parameters.titlepad)
            ax.legend(legend_handles, legend_strings)
            ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
            ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)

            fig.savefig('clustering_result_for_rotten_spruce_stems_on_terrain_map.png')
            plt.close(fig)



        #
        # On white background
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        legend_handles = []
        legend_strings = []

        for this_cluster in position_and_cluster_id_data_for_each_cluster_including_outliers:

            if this_cluster[0, 2].astype(int) == -1:

                this_stem_markersize = parameters.stem_markersize_large*0.3
                this_stem_marker = 'o'

            else:

                this_stem_markersize = parameters.stem_markersize_large
                this_stem_marker = 's'

            this_h = ax.scatter(this_cluster[:, 0], this_cluster[:, 1], c = rot_cluster_color_map[this_cluster[:, 2].astype(int)], s = this_stem_markersize, marker = this_stem_marker)

            legend_handles.append(this_h)

            if this_cluster[0, 2].astype(int) == -1:

                legend_strings.append("Outliers")

            else:

                legend_strings.append("Rot cluster number " + str(this_cluster[0, 2].astype(int) + 1))

        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Clustering of rotten spruce stems via DBSCAN, rot percent = %3.1f' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems), pad = parameters.titlepad)
        ax.legend(legend_handles, legend_strings)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)

        fig.savefig('clustering_result_for_rotten_spruce_stems_on_white_background.png')
        plt.close(fig)



        #
        # Delineation for each cluster and each individual rotten stem,
        # with all spruce stem positions in the background marked by
        # rotten_200_with_sawlog, on the four different backgrounds
        #


        if parameters.use_raster_data:
            

            #
            # On fertility class
            #

            fig, ax = plt.subplots(figsize = parameters.figure_size)

            mappable = ax.imshow(sliced_fertility_class_as_array, extent = (sliced_fertility_class_data_min_easting - fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_max_easting + fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_min_northing - fertility_class_pixel_size_y/2.0, sliced_fertility_class_data_max_northing + fertility_class_pixel_size_y/2.0), cmap = parameters.fertility_class_cmap, norm = parameters.fertility_class_cmap_norm, alpha = parameters.fertility_class_visualization_alpha, zorder = -999)

            cbar = fig.colorbar(mappable, ticks = parameters.fertility_class_tick_locations, pad = 0.06, shrink = 0.8)
            cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
            cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
            cbar.ax.set_yticklabels(parameters.fertility_class_tick_labels, size = parameters.fertility_class_tick_label_font_size)

            h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)

            h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')

            legend_handles = [h_spruce_rot_status_one, h_spruce_rot_status_zero]
            legend_strings = ["Rot status one", "Rot status zero"]


            for this_delineation in cluster_delineations:

                this_cluster_id = this_delineation[1]

                if this_cluster_id == -1:

                    continue

                this_delineation_shape = this_delineation[0]    

                this_delineation_color = rot_cluster_color_map[this_cluster_id]

                this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))


                legend_handles.append(this_delineation_handle)

                legend_strings.append("Rot cluster number " + str(this_cluster_id + 1))



            if individual_stem_delineations != []:


                for this_delineation in individual_stem_delineations:


                    this_cluster_id = this_delineation[1]

                    this_delineation_shape = this_delineation[0]

                    this_delineation_color = rot_cluster_color_map[this_cluster_id]

                    this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))


                legend_handles.append(this_delineation_handle)

                legend_strings.append("Outliers ")


            ax.set_aspect('equal', 'box')
            ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
            ax.set_title('Spruce stem positions and rot area delineations, rot percent = %3.1f' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems), pad = parameters.titlepad)
            ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
            ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
            ax.legend(legend_handles, legend_strings)

            fig.savefig('spruce_stem_positions_and_rotten_200_with_sawlog_and_rot_area_delineations_on_fertility_class.png')
            plt.close(fig)



            #
            # On soil type
            #

            fig, ax = plt.subplots(figsize = parameters.figure_size)

            mappable = ax.imshow(sliced_soil_type_as_array, extent = (sliced_soil_type_data_min_easting - soil_type_pixel_size_x/2.0, sliced_soil_type_data_max_easting + soil_type_pixel_size_x/2.0, sliced_soil_type_data_min_northing - soil_type_pixel_size_y/2.0, sliced_soil_type_data_max_northing + soil_type_pixel_size_y/2.0), cmap = parameters.soil_type_cmap, norm = parameters.soil_type_cmap_norm, alpha = parameters.soil_type_visualization_alpha, zorder = -999)

            cbar = fig.colorbar(mappable, ticks = parameters.soil_type_tick_locations, pad = 0.06, shrink = 0.8)
            cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
            cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
            cbar.ax.set_yticklabels(parameters.soil_type_tick_labels, size = parameters.soil_type_tick_label_font_size)

            h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)

            h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')

            legend_handles = [h_spruce_rot_status_one, h_spruce_rot_status_zero]
            legend_strings = ["Rot status one", "Rot status zero"]


            for this_delineation in cluster_delineations:

                this_cluster_id = this_delineation[1]

                if this_cluster_id == -1:

                    continue

                this_delineation_shape = this_delineation[0]    

                this_delineation_color = rot_cluster_color_map[this_cluster_id]

                this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))


                legend_handles.append(this_delineation_handle)

                legend_strings.append("Rot cluster number " + str(this_cluster_id + 1))



            if individual_stem_delineations != []:


                for this_delineation in individual_stem_delineations:


                    this_cluster_id = this_delineation[1]

                    this_delineation_shape = this_delineation[0]

                    this_delineation_color = rot_cluster_color_map[this_cluster_id]

                    this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))


                legend_handles.append(this_delineation_handle)

                legend_strings.append("Outliers ")


            ax.set_aspect('equal', 'box')
            ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
            ax.set_title('Spruce stem positions and rot area delineations, rot percent = %3.1f' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems), pad = parameters.titlepad)
            ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
            ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
            ax.legend(legend_handles, legend_strings)

            fig.savefig('spruce_stem_positions_and_rotten_200_with_sawlog_and_rot_area_delineations_on_soil_type.png')
            plt.close(fig)



            #
            # On terrain map
            #

            fig, ax = plt.subplots(figsize = parameters.figure_size)

            mappable = ax.imshow(terrain_map_as_array, cmap = terrain_raster_cmap, norm = terrain_raster_cmap_norm, extent = (terrain_map_pixel_center_min_easting - terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_max_easting + terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_min_northing - terrain_pixel_size_in_y_direction/2.0, terrain_map_pixel_center_max_northing + terrain_pixel_size_in_y_direction/2.0), alpha = parameters.terrain_raster_map_alpha, zorder = -999)

            h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)

            h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')

            legend_handles = [h_spruce_rot_status_one, h_spruce_rot_status_zero]
            legend_strings = ["Rot status one", "Rot status zero"]


            for this_delineation in cluster_delineations:

                this_cluster_id = this_delineation[1]

                if this_cluster_id == -1:

                    continue

                this_delineation_shape = this_delineation[0]    

                this_delineation_color = rot_cluster_color_map[this_cluster_id]

                this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))


                legend_handles.append(this_delineation_handle)

                legend_strings.append("Rot cluster number " + str(this_cluster_id + 1))



            if individual_stem_delineations != []:


                for this_delineation in individual_stem_delineations:


                    this_cluster_id = this_delineation[1]

                    this_delineation_shape = this_delineation[0]

                    this_delineation_color = rot_cluster_color_map[this_cluster_id]

                    this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))


                legend_handles.append(this_delineation_handle)

                legend_strings.append("Outliers ")


            ax.set_aspect('equal', 'box')
            ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
            ax.set_title('Spruce stem positions and rot area delineations, rot percent = %3.1f' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems), pad = parameters.titlepad)
            ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
            ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
            ax.legend(legend_handles, legend_strings)

            fig.savefig('spruce_stem_positions_and_rotten_200_with_sawlog_and_rot_area_delineations_on_terrain_map.png')
            plt.close(fig)



        #
        # On white background
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)

        h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')

        legend_handles = [h_spruce_rot_status_one, h_spruce_rot_status_zero]
        legend_strings = ["Rot status one", "Rot status zero"]


        for this_delineation in cluster_delineations:

            this_cluster_id = this_delineation[1]

            if this_cluster_id == -1:

                continue

            this_delineation_shape = this_delineation[0]    

            this_delineation_color = rot_cluster_color_map[this_cluster_id]

            this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))


            legend_handles.append(this_delineation_handle)

            legend_strings.append("Rot cluster number " + str(this_cluster_id + 1))



        if individual_stem_delineations != []:


            for this_delineation in individual_stem_delineations:


                this_cluster_id = this_delineation[1]

                this_delineation_shape = this_delineation[0]

                this_delineation_color = rot_cluster_color_map[this_cluster_id]

                this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))


            legend_handles.append(this_delineation_handle)

            legend_strings.append("Outliers ")


        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Spruce stem positions and rot area delineations, rot percent = %3.1f' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems), pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend(legend_handles, legend_strings)

        fig.savefig('spruce_stem_positions_and_rotten_200_with_sawlog_and_rot_area_delineations_on_white_background.png')
        plt.close(fig)



        #
        # The rot area delineations, with sawlog-caliber rotten spruce
        # stems marked in one way and all other stems marked in
        # another way, on the four different backgrounds
        #


        if parameters.use_raster_data:
        

            #
            # On fertility class
            #

            fig, ax = plt.subplots(figsize = parameters.figure_size)

            mappable = ax.imshow(sliced_fertility_class_as_array, extent = (sliced_fertility_class_data_min_easting - fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_max_easting + fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_min_northing - fertility_class_pixel_size_y/2.0, sliced_fertility_class_data_max_northing + fertility_class_pixel_size_y/2.0), cmap = parameters.fertility_class_cmap, norm = parameters.fertility_class_cmap_norm, alpha = parameters.fertility_class_visualization_alpha, zorder = -999)

            cbar = fig.colorbar(mappable, ticks = parameters.fertility_class_tick_locations, pad = 0.06, shrink = 0.8)
            cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
            cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
            cbar.ax.set_yticklabels(parameters.fertility_class_tick_labels, size = parameters.fertility_class_tick_label_font_size)

            h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'black', s = parameters.stem_markersize_small, marker = '.')
            h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')
            h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'black', s = parameters.stem_markersize_small, marker = '.')
            h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'black', s = parameters.stem_markersize_small, marker = '.')

            h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)

            legend_handles = [h_spruce_rot_status_one, h_spruce_rot_status_zero]
            legend_strings = ["Spruce, rot status one", "All other stems"]


            #
            # Add the buffer(0) call here to get the "holes" to display correctly
            #

            if isinstance(total_healthy_area_delineation.buffer(0), MultiPolygon):

                for this_polygon in total_healthy_area_delineation.buffer(0).geoms:

                    h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k'))

            else:

                h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(total_healthy_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k'))

            legend_handles.append(h_total_healthy_area_delineation)

            legend_strings.append("Harvest area outside of rot areas")


            for this_delineation in cluster_delineations:

                this_cluster_id = this_delineation[1]

                if this_cluster_id == -1:

                    continue

                this_delineation_shape = this_delineation[0]    

                this_delineation_color = rot_cluster_color_map[this_cluster_id]

                this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))


                legend_handles.append(this_delineation_handle)

                legend_strings.append("Rot area number " + str(this_cluster_id + 1))



            if individual_stem_delineations != []:


                for this_delineation in individual_stem_delineations:


                    this_cluster_id = this_delineation[1]

                    this_delineation_shape = this_delineation[0]

                    this_delineation_color = rot_cluster_color_map[this_cluster_id]

                    this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))

                legend_handles.append(this_delineation_handle)

                legend_strings.append("Rot areas around individual stems ")


            ax.set_aspect('equal', 'box')
            ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
            ax.set_title('Rot area delineations, $f_{rot}$ = %3.1f%%, $T_{sum}$ = %.1f d.d.' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems, temperature_sum_for_stand), pad = parameters.titlepad)
            ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
            ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
            ax.legend(legend_handles, legend_strings)

            fig.savefig('rot_area_delineations_on_fertility_class.png')
            plt.close(fig)



            #
            # On soil type
            #

            fig, ax = plt.subplots(figsize = parameters.figure_size)

            mappable = ax.imshow(sliced_soil_type_as_array, extent = (sliced_soil_type_data_min_easting - soil_type_pixel_size_x/2.0, sliced_soil_type_data_max_easting + soil_type_pixel_size_x/2.0, sliced_soil_type_data_min_northing - soil_type_pixel_size_y/2.0, sliced_soil_type_data_max_northing + soil_type_pixel_size_y/2.0), cmap = parameters.soil_type_cmap, norm = parameters.soil_type_cmap_norm, alpha = parameters.soil_type_visualization_alpha, zorder = -999)

            cbar = fig.colorbar(mappable, ticks = parameters.soil_type_tick_locations, pad = 0.06, shrink = 0.8)
            cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
            cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
            cbar.ax.set_yticklabels(parameters.soil_type_tick_labels, size = parameters.soil_type_tick_label_font_size)

            h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'black', s = parameters.stem_markersize_small, marker = '.')
            h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')
            h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'black', s = parameters.stem_markersize_small, marker = '.')
            h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'black', s = parameters.stem_markersize_small, marker = '.')

            h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)

            legend_handles = [h_spruce_rot_status_one, h_spruce_rot_status_zero]
            legend_strings = ["Spruce, rot status one", "All other stems"]


            #
            # Add the buffer(0) call here to get the "holes" to display correctly
            #

            if isinstance(total_healthy_area_delineation.buffer(0), MultiPolygon):

                for this_polygon in total_healthy_area_delineation.buffer(0).geoms:

                    h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k'))

            else:

                h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(total_healthy_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k'))

            legend_handles.append(h_total_healthy_area_delineation)

            legend_strings.append("Harvest area outside of rot areas")


            for this_delineation in cluster_delineations:

                this_cluster_id = this_delineation[1]

                if this_cluster_id == -1:

                    continue

                this_delineation_shape = this_delineation[0]    

                this_delineation_color = rot_cluster_color_map[this_cluster_id]

                this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))


                legend_handles.append(this_delineation_handle)

                legend_strings.append("Rot area number " + str(this_cluster_id + 1))



            if individual_stem_delineations != []:


                for this_delineation in individual_stem_delineations:


                    this_cluster_id = this_delineation[1]

                    this_delineation_shape = this_delineation[0]

                    this_delineation_color = rot_cluster_color_map[this_cluster_id]

                    this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))

                legend_handles.append(this_delineation_handle)

                legend_strings.append("Rot areas around individual stems ")


            ax.set_aspect('equal', 'box')
            ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
            ax.set_title('Rot area delineations, $f_{rot}$ = %3.1f%%, $T_{sum}$ = %.1f d.d.' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems, temperature_sum_for_stand), pad = parameters.titlepad)
            ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
            ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
            ax.legend(legend_handles, legend_strings)

            fig.savefig('rot_area_delineations_on_soil_type.png')
            plt.close(fig)



            #
            # On terrain map
            #

            fig, ax = plt.subplots(figsize = parameters.figure_size)

            mappable = ax.imshow(terrain_map_as_array, cmap = terrain_raster_cmap, norm = terrain_raster_cmap_norm, extent = (terrain_map_pixel_center_min_easting - terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_max_easting + terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_min_northing - terrain_pixel_size_in_y_direction/2.0, terrain_map_pixel_center_max_northing + terrain_pixel_size_in_y_direction/2.0), alpha = parameters.terrain_raster_map_alpha, zorder = -999)

            h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'black', s = parameters.stem_markersize_small, marker = '.')
            h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')
            h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'black', s = parameters.stem_markersize_small, marker = '.')
            h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'black', s = parameters.stem_markersize_small, marker = '.')

            h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)

            legend_handles = [h_spruce_rot_status_one, h_spruce_rot_status_zero]
            legend_strings = ["Spruce, rot status one", "All other stems"]


            #
            # Add the buffer(0) call here to get the "holes" to display correctly
            #

            if isinstance(total_healthy_area_delineation.buffer(0), MultiPolygon):

                for this_polygon in total_healthy_area_delineation.buffer(0).geoms:

                    h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k'))

            else:

                h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(total_healthy_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k'))

            legend_handles.append(h_total_healthy_area_delineation)

            legend_strings.append("Harvest area outside of rot areas")


            for this_delineation in cluster_delineations:

                this_cluster_id = this_delineation[1]

                if this_cluster_id == -1:

                    continue

                this_delineation_shape = this_delineation[0]    

                this_delineation_color = rot_cluster_color_map[this_cluster_id]

                this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))


                legend_handles.append(this_delineation_handle)

                legend_strings.append("Rot area number " + str(this_cluster_id + 1))



            if individual_stem_delineations != []:


                for this_delineation in individual_stem_delineations:


                    this_cluster_id = this_delineation[1]

                    this_delineation_shape = this_delineation[0]

                    this_delineation_color = rot_cluster_color_map[this_cluster_id]

                    this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))

                legend_handles.append(this_delineation_handle)

                legend_strings.append("Rot areas around individual stems ")


            ax.set_aspect('equal', 'box')
            ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
            ax.set_title('Rot area delineations, $f_{rot}$ = %3.1f%%, $T_{sum}$ = %.1f d.d.' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems, temperature_sum_for_stand), pad = parameters.titlepad)
            ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
            ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
            ax.legend(legend_handles, legend_strings)

            fig.savefig('rot_area_delineations_on_terrain_map.png')
            plt.close(fig)



        #
        # On white background
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_spruce_rot_status_zero = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 0)], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'black', s = parameters.stem_markersize_small, marker = '.')
        h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'black', s = parameters.stem_markersize_small, marker = '.')

        h_spruce_rot_status_one = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], c = 'black', s = parameters.stem_markersize_large, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)

        legend_handles = [h_spruce_rot_status_one, h_spruce_rot_status_zero]
        legend_strings = ["Spruce, rot status one", "All other stems"]


        #
        # Add the buffer(0) call here to get the "holes" to display correctly
        #

        if isinstance(total_healthy_area_delineation.buffer(0), MultiPolygon):

            for this_polygon in total_healthy_area_delineation.buffer(0).geoms:

                h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k'))

        else:

            h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(total_healthy_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k'))

        legend_handles.append(h_total_healthy_area_delineation)

        legend_strings.append("Harvest area outside of rot areas")


        for this_delineation in cluster_delineations:

            this_cluster_id = this_delineation[1]

            if this_cluster_id == -1:

                continue

            this_delineation_shape = this_delineation[0]    

            this_delineation_color = rot_cluster_color_map[this_cluster_id]

            this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))


            legend_handles.append(this_delineation_handle)

            legend_strings.append("Rot area number " + str(this_cluster_id + 1))



        if individual_stem_delineations != []:


            for this_delineation in individual_stem_delineations:


                this_cluster_id = this_delineation[1]

                this_delineation_shape = this_delineation[0]

                this_delineation_color = rot_cluster_color_map[this_cluster_id]

                this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))

            legend_handles.append(this_delineation_handle)

            legend_strings.append("Rot areas around individual stems ")


        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Rot area delineations, $f_{rot}$ = %3.1f%%, $T_{sum}$ = %.1f d.d.' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems, temperature_sum_for_stand), pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend(legend_handles, legend_strings)

        fig.savefig('rot_area_delineations_on_white_background.png')
        plt.close(fig)



        #
        # Rot area delineations, with the following indications for
        # different stems, on the four different backgrounds:
        #
        # - Spruce, rotten sawlog-caliber stems -> filled black square
        # - Spruce, healthy sawlog-caliber stems -> open black square
        # - Spruce, lower than sawlog-caliber stems -> filled black star
        # - Pine -> 'o', blue
        # - Birch -> 'd', green
        # - Other species -> '^', red
        #


        if parameters.use_raster_data:
        
        
            #
            # On fertility class
            #

            fig, ax = plt.subplots(figsize = parameters.figure_size)

            mappable = ax.imshow(sliced_fertility_class_as_array, extent = (sliced_fertility_class_data_min_easting - fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_max_easting + fertility_class_pixel_size_x/2.0, sliced_fertility_class_data_min_northing - fertility_class_pixel_size_y/2.0, sliced_fertility_class_data_max_northing + fertility_class_pixel_size_y/2.0), cmap = parameters.fertility_class_cmap, norm = parameters.fertility_class_cmap_norm, alpha = parameters.fertility_class_visualization_alpha, zorder = -999)

            cbar = fig.colorbar(mappable, ticks = parameters.fertility_class_tick_locations, pad = 0.06, shrink = 0.8)
            cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
            cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
            cbar.ax.set_yticklabels(parameters.fertility_class_tick_labels, size = parameters.fertility_class_tick_label_font_size)

            h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'blue', s = parameters.stem_markersize_medium, marker = 'o')

            h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'green', s = parameters.stem_markersize_medium, marker = 'd')

            h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'red', s = parameters.stem_markersize_medium, marker = '^')

            h_spruce_rotten_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], c = 'black', s = parameters.stem_markersize_medium, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)

            h_spruce_healthy_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], edgecolors = 'black', facecolors = 'none', s = parameters.stem_markersize_medium, marker = 's')

            h_spruce_lower_than_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], northing_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], c = 'black', s = parameters.stem_markersize_medium, marker = '*')

            legend_handles = [h_spruce_rotten_sawlog_caliber, h_spruce_healthy_sawlog_caliber, h_spruce_lower_than_sawlog_caliber, h_pine, h_birch, h_other]

            legend_strings = ['Spruce, sawlog-caliber, rotten', 'Spruce, sawlog-caliber, healthy', 'Spruce, lower than sawlog-caliber', 'Pine', 'Birch', 'Other']


            #
            # Add the buffer(0) call here to get the "holes" to display correctly
            #

            if isinstance(total_healthy_area_delineation.buffer(0), MultiPolygon):

                for this_polygon in total_healthy_area_delineation.buffer(0).geoms:

                    h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k'))

            else:

                h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(total_healthy_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k'))

            legend_handles.append(h_total_healthy_area_delineation)

            legend_strings.append('Harvest area outside of rot areas')


            for this_delineation in cluster_delineations:


                this_cluster_id = this_delineation[1]


                if this_cluster_id == -1:

                    continue


                this_delineation_shape = this_delineation[0]    

                this_delineation_color = rot_cluster_color_map[this_cluster_id]

                this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))


                legend_handles.append(this_delineation_handle)

                legend_strings.append('Rot area number ' + str(this_cluster_id + 1))



            if individual_stem_delineations != []:


                for this_delineation in individual_stem_delineations:


                    this_cluster_id = this_delineation[1]

                    this_delineation_shape = this_delineation[0]

                    this_delineation_color = rot_cluster_color_map[this_cluster_id]

                    this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))


                legend_handles.append(this_delineation_handle)

                legend_strings.append('Rot areas around individual stems ')


            ax.set_aspect('equal', 'box')
            ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
            ax.set_title('Rot area delineations, $f_{rot}$ = %3.1f%%, $T_{sum}$ = %.1f d.d.' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems, temperature_sum_for_stand), pad = parameters.titlepad)
            ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
            ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
            ax.legend(legend_handles, legend_strings)

            fig.savefig('rot_area_delineations_with_details_on_stems_on_fertility_class.png')
            plt.close(fig)



            #
            # On soil type
            #

            fig, ax = plt.subplots(figsize = parameters.figure_size)

            mappable = ax.imshow(sliced_soil_type_as_array, extent = (sliced_soil_type_data_min_easting - soil_type_pixel_size_x/2.0, sliced_soil_type_data_max_easting + soil_type_pixel_size_x/2.0, sliced_soil_type_data_min_northing - soil_type_pixel_size_y/2.0, sliced_soil_type_data_max_northing + soil_type_pixel_size_y/2.0), cmap = parameters.soil_type_cmap, norm = parameters.soil_type_cmap_norm, alpha = parameters.soil_type_visualization_alpha, zorder = -999)

            cbar = fig.colorbar(mappable, ticks = parameters.soil_type_tick_locations, pad = 0.06, shrink = 0.8)
            cbar.ax.tick_params(axis = 'y', which = 'major', pad = 15)
            cbar.ax.tick_params(axis = 'y', which = 'minor', length = 0)
            cbar.ax.set_yticklabels(parameters.soil_type_tick_labels, size = parameters.soil_type_tick_label_font_size)

            h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'blue', s = parameters.stem_markersize_medium, marker = 'o')

            h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'green', s = parameters.stem_markersize_medium, marker = 'd')

            h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'red', s = parameters.stem_markersize_medium, marker = '^')

            h_spruce_rotten_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], c = 'black', s = parameters.stem_markersize_medium, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)

            h_spruce_healthy_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], edgecolors = 'black', facecolors = 'none', s = parameters.stem_markersize_medium, marker = 's')

            h_spruce_lower_than_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], northing_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], c = 'black', s = parameters.stem_markersize_medium, marker = '*')

            legend_handles = [h_spruce_rotten_sawlog_caliber, h_spruce_healthy_sawlog_caliber, h_spruce_lower_than_sawlog_caliber, h_pine, h_birch, h_other]

            legend_strings = ['Spruce, sawlog-caliber, rotten', 'Spruce, sawlog-caliber, healthy', 'Spruce, lower than sawlog-caliber', 'Pine', 'Birch', 'Other']


            #
            # Add the buffer(0) call here to get the "holes" to display correctly
            #

            if isinstance(total_healthy_area_delineation.buffer(0), MultiPolygon):

                for this_polygon in total_healthy_area_delineation.buffer(0).geoms:

                    h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k'))

            else:

                h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(total_healthy_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k'))

            legend_handles.append(h_total_healthy_area_delineation)

            legend_strings.append('Harvest area outside of rot areas')


            for this_delineation in cluster_delineations:


                this_cluster_id = this_delineation[1]


                if this_cluster_id == -1:

                    continue


                this_delineation_shape = this_delineation[0]    

                this_delineation_color = rot_cluster_color_map[this_cluster_id]

                this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))


                legend_handles.append(this_delineation_handle)

                legend_strings.append('Rot area number ' + str(this_cluster_id + 1))



            if individual_stem_delineations != []:


                for this_delineation in individual_stem_delineations:


                    this_cluster_id = this_delineation[1]

                    this_delineation_shape = this_delineation[0]

                    this_delineation_color = rot_cluster_color_map[this_cluster_id]

                    this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))


                legend_handles.append(this_delineation_handle)

                legend_strings.append('Rot areas around individual stems ')


            ax.set_aspect('equal', 'box')
            ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
            ax.set_title('Rot area delineations, $f_{rot}$ = %3.1f%%, $T_{sum}$ = %.1f d.d.' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems, temperature_sum_for_stand), pad = parameters.titlepad)
            ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
            ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
            ax.legend(legend_handles, legend_strings)

            fig.savefig('rot_area_delineations_with_details_on_stems_on_soil_type.png')
            plt.close(fig)



            #
            # On terrain map
            #

            fig, ax = plt.subplots(figsize = parameters.figure_size)

            mappable = ax.imshow(terrain_map_as_array, cmap = terrain_raster_cmap, norm = terrain_raster_cmap_norm, extent = (terrain_map_pixel_center_min_easting - terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_max_easting + terrain_pixel_size_in_x_direction/2.0, terrain_map_pixel_center_min_northing - terrain_pixel_size_in_y_direction/2.0, terrain_map_pixel_center_max_northing + terrain_pixel_size_in_y_direction/2.0), alpha = parameters.terrain_raster_map_alpha, zorder = -999)

            h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'blue', s = parameters.stem_markersize_medium, marker = 'o')

            h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'green', s = parameters.stem_markersize_medium, marker = 'd')

            h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'red', s = parameters.stem_markersize_medium, marker = '^')

            h_spruce_rotten_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], c = 'black', s = parameters.stem_markersize_medium, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)

            h_spruce_healthy_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], edgecolors = 'black', facecolors = 'none', s = parameters.stem_markersize_medium, marker = 's')

            h_spruce_lower_than_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], northing_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], c = 'black', s = parameters.stem_markersize_medium, marker = '*')

            legend_handles = [h_spruce_rotten_sawlog_caliber, h_spruce_healthy_sawlog_caliber, h_spruce_lower_than_sawlog_caliber, h_pine, h_birch, h_other]

            legend_strings = ['Spruce, sawlog-caliber, rotten', 'Spruce, sawlog-caliber, healthy', 'Spruce, lower than sawlog-caliber', 'Pine', 'Birch', 'Other']


            #
            # Add the buffer(0) call here to get the "holes" to display correctly
            #

            if isinstance(total_healthy_area_delineation.buffer(0), MultiPolygon):

                for this_polygon in total_healthy_area_delineation.buffer(0).geoms:

                    h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k'))

            else:

                h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(total_healthy_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k'))

            legend_handles.append(h_total_healthy_area_delineation)

            legend_strings.append('Harvest area outside of rot areas')


            for this_delineation in cluster_delineations:


                this_cluster_id = this_delineation[1]


                if this_cluster_id == -1:

                    continue


                this_delineation_shape = this_delineation[0]    

                this_delineation_color = rot_cluster_color_map[this_cluster_id]

                this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))


                legend_handles.append(this_delineation_handle)

                legend_strings.append('Rot area number ' + str(this_cluster_id + 1))



            if individual_stem_delineations != []:


                for this_delineation in individual_stem_delineations:


                    this_cluster_id = this_delineation[1]

                    this_delineation_shape = this_delineation[0]

                    this_delineation_color = rot_cluster_color_map[this_cluster_id]

                    this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))


                legend_handles.append(this_delineation_handle)

                legend_strings.append('Rot areas around individual stems ')


            ax.set_aspect('equal', 'box')
            ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
            ax.set_title('Rot area delineations, $f_{rot}$ = %3.1f%%, $T_{sum}$ = %.1f d.d.' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems, temperature_sum_for_stand), pad = parameters.titlepad)
            ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
            ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
            ax.legend(legend_handles, legend_strings)

            fig.savefig('rot_area_delineations_with_details_on_stems_on_terrain_map.png')
            plt.close(fig)



        #
        # On white background
        #

        fig, ax = plt.subplots(figsize = parameters.figure_size)

        h_pine = ax.scatter(easting_crane_tip[species_group_id == 1], northing_crane_tip[species_group_id == 1], c = 'blue', s = parameters.stem_markersize_medium, marker = 'o')

        h_birch = ax.scatter(easting_crane_tip[species_group_id == 3], northing_crane_tip[species_group_id == 3], c = 'green', s = parameters.stem_markersize_medium, marker = 'd')

        h_other = ax.scatter(easting_crane_tip[species_group_id == 4], northing_crane_tip[species_group_id == 4], c = 'red', s = parameters.stem_markersize_medium, marker = '^')

        h_spruce_rotten_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 1, theoretical_sawlog_volume > 0.0))], c = 'black', s = parameters.stem_markersize_medium, marker = 's', alpha = parameters.rot_status_one_stem_marker_alpha)

        h_spruce_healthy_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], northing_crane_tip[np.logical_and(species_group_id == 2, np.logical_and(rotten_200_with_sawlog == 0, theoretical_sawlog_volume > 0.0))], edgecolors = 'black', facecolors = 'none', s = parameters.stem_markersize_medium, marker = 's')

        h_spruce_lower_than_sawlog_caliber = ax.scatter(easting_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], northing_crane_tip[np.logical_and(species_group_id == 2, theoretical_sawlog_volume <= 0.0)], c = 'black', s = parameters.stem_markersize_medium, marker = '*')

        legend_handles = [h_spruce_rotten_sawlog_caliber, h_spruce_healthy_sawlog_caliber, h_spruce_lower_than_sawlog_caliber, h_pine, h_birch, h_other]

        legend_strings = ['Spruce, sawlog-caliber, rotten', 'Spruce, sawlog-caliber, healthy', 'Spruce, lower than sawlog-caliber', 'Pine', 'Birch', 'Other']


        #
        # Add the buffer(0) call here to get the "holes" to display correctly
        #

        if isinstance(total_healthy_area_delineation.buffer(0), MultiPolygon):

            for this_polygon in total_healthy_area_delineation.buffer(0).geoms:

                h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(this_polygon, parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k'))

        else:

            h_total_healthy_area_delineation = ax.add_patch(create_polygon_pathpatch(total_healthy_area_delineation.buffer(0), parameters.delineation_visualization_alpha_lighter, parameters.full_harvest_area_delineation_color, 'k'))

        legend_handles.append(h_total_healthy_area_delineation)

        legend_strings.append('Harvest area outside of rot areas')


        for this_delineation in cluster_delineations:


            this_cluster_id = this_delineation[1]


            if this_cluster_id == -1:

                continue


            this_delineation_shape = this_delineation[0]    

            this_delineation_color = rot_cluster_color_map[this_cluster_id]

            this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))


            legend_handles.append(this_delineation_handle)

            legend_strings.append('Rot area number ' + str(this_cluster_id + 1))



        if individual_stem_delineations != []:


            for this_delineation in individual_stem_delineations:


                this_cluster_id = this_delineation[1]

                this_delineation_shape = this_delineation[0]

                this_delineation_color = rot_cluster_color_map[this_cluster_id]

                this_delineation_handle = ax.add_patch(create_polygon_pathpatch(this_delineation_shape, parameters.delineation_visualization_alpha, this_delineation_color, 'k'))


            legend_handles.append(this_delineation_handle)

            legend_strings.append('Rot areas around individual stems ')


        ax.set_aspect('equal', 'box')
        ax.set(xlim = (graph_axis_limits[0], graph_axis_limits[1]), ylim = (graph_axis_limits[2], graph_axis_limits[3]))
        ax.set_title('Rot area delineations, $f_{rot}$ = %3.1f%%, $T_{sum}$ = %.1f d.d.' % (100.0*fraction_of_rotten_spruce_sawlog_caliber_stems, temperature_sum_for_stand), pad = parameters.titlepad)
        ax.set_xlabel('Easting (m)', labelpad = parameters.labelpad)
        ax.set_ylabel('Northing (m)', labelpad = parameters.labelpad)
        ax.legend(legend_handles, legend_strings)

        fig.savefig('rot_area_delineations_with_details_on_stems_on_white_background.png')
        plt.close(fig)
    
        

#
# Plot the results of the optimization as bar diagrams
#

def create_plots_for_optimization_results(list_of_alternatives, indeces_from_lowest_to_highest_blv, blv_eur_for_the_alternatives, indeces_from_lowest_to_highest_carbon, carbon_tCO2_for_the_alternatives, stand_area_in_ha):

    
    #
    # BLV of different alternatives, sorted in descending
    # order, as a bar diagram
    #

    fig, ax = plt.subplots(figsize = parameters.figure_size)

    bar_positions = np.arange(0, len(list_of_alternatives)) + 1

    bar_labels = []

    bar_heights = []

    for i_alternative in indeces_from_lowest_to_highest_blv[::-1]:

        this_alternative_description = list_of_alternatives[i_alternative]
    
        bar_labels.append(this_alternative_description)

        this_alternative_blv = blv_eur_for_the_alternatives[i_alternative]
    
        bar_heights.append(this_alternative_blv)
    
    ax.bar(bar_positions, bar_heights, width = parameters.bar_width_in_bar_diagrams, color = 'red', edgecolor = 'black', linewidth = parameters.linewidth_for_bar_diagram)

    ax.set_xlabel('Alternative', labelpad = parameters.labelpad)

    ax.set_ylabel('BLV (EUR)', labelpad = parameters.labelpad)

    ax.set_xticks(bar_positions, bar_labels, rotation = 90.0)

    fig.subplots_adjust(bottom = parameters.bottom_position_of_bar_diagram)

    ax.set_title('Expected BLV for each regeneration alternative', pad = parameters.titlepad)

    fig.savefig('bar_diagram_of_blvs_for_the_alternatives.png')

    plt.close(fig)



    #
    # The difference in BLV (EUR/ha) of the alternatives 2, 3, ..., 10
    # to the first-best alternative, sorted in descending order, as a
    # bar diagram
    #

    fig, ax = plt.subplots(figsize = parameters.figure_size)

    bar_positions = np.arange(0, len(list_of_alternatives)) + 1

    bar_labels = []

    bar_heights = []

    for i_alternative in indeces_from_lowest_to_highest_blv[::-1]:

        this_alternative_description = list_of_alternatives[i_alternative]
    
        bar_labels.append(this_alternative_description)

        this_alternative_blv_per_ha_difference = (blv_eur_for_the_alternatives[i_alternative] - blv_eur_for_the_alternatives[indeces_from_lowest_to_highest_blv[-1]]) / stand_area_in_ha
    
        bar_heights.append(this_alternative_blv_per_ha_difference)
    
    ax.bar(bar_positions, bar_heights, width = parameters.bar_width_in_bar_diagrams, color = 'red', edgecolor = 'black', linewidth = parameters.linewidth_for_bar_diagram)

    ax.set_xlabel('Alternative', labelpad = parameters.labelpad)

    ax.set_ylabel('Difference in BLV to first-best alternative (EUR/ha)', labelpad = parameters.labelpad)

    ax.set_xticks(bar_positions, bar_labels, rotation = 90.0)

    fig.subplots_adjust(bottom = parameters.bottom_position_of_bar_diagram)

    ax.set_title('Difference in BLV per ha to first-best alternative', pad = parameters.titlepad)

    fig.savefig('bar_diagram_of_blv_per_ha_differences.png')

    plt.close(fig)



    #
    # Carbon of different alternatives, sorted in descending order, as a
    # bar diagram
    #

    fig, ax = plt.subplots(figsize = parameters.figure_size)

    bar_positions = np.arange(0, len(list_of_alternatives)) + 1

    bar_labels = []

    bar_heights = []

    for i_alternative in indeces_from_lowest_to_highest_carbon[::-1]:

        this_alternative_description = list_of_alternatives[i_alternative]
    
        bar_labels.append(this_alternative_description)

        this_alternative_carbon = carbon_tCO2_for_the_alternatives[i_alternative]
    
        bar_heights.append(this_alternative_carbon)
    
    ax.bar(bar_positions, bar_heights, width = parameters.bar_width_in_bar_diagrams, color = 'blue', edgecolor = 'black', linewidth = parameters.linewidth_for_bar_diagram)

    ax.set_xlabel('Alternative', labelpad = parameters.labelpad)
    
    ax.set_ylabel('Carbon (tCO2-equivalent)', labelpad = parameters.labelpad)
    
    ax.set_xticks(bar_positions, bar_labels, rotation = 90.0)
    
    fig.subplots_adjust(bottom = parameters.bottom_position_of_bar_diagram)
    
    ax.set_title('Expected maximum carbon content of stand over one rotation period for each regeneration alternative', pad = parameters.titlepad)
    
    fig.savefig('bar_diagram_of_carbons_for_the_alternatives.png')
    
    plt.close(fig)

    

    #
    # The difference in carbon content (tCO2-equivalent/ha) of the
    # alternatives 2, 3, ..., 10 to the first-best alternative,
    # sorted in descending order, as a bar diagram
    #

    fig, ax = plt.subplots(figsize = parameters.figure_size)

    bar_positions = np.arange(0, len(list_of_alternatives)) + 1

    bar_labels = []

    bar_heights = []

    for i_alternative in indeces_from_lowest_to_highest_carbon[::-1]:

        this_alternative_description = list_of_alternatives[i_alternative]
    
        bar_labels.append(this_alternative_description)

        this_alternative_carbon_per_ha_difference = (carbon_tCO2_for_the_alternatives[i_alternative] - carbon_tCO2_for_the_alternatives[indeces_from_lowest_to_highest_carbon[-1]]) / stand_area_in_ha
    
        bar_heights.append(this_alternative_carbon_per_ha_difference)
    
    ax.bar(bar_positions, bar_heights, width = parameters.bar_width_in_bar_diagrams, color = 'blue', edgecolor = 'black', linewidth = parameters.linewidth_for_bar_diagram)

    ax.set_xlabel('Alternative', labelpad = parameters.labelpad)
    
    ax.set_ylabel('Difference in carbon to first-best alternative (tCO2-equivalent/ha)', labelpad = parameters.labelpad)
    
    ax.set_xticks(bar_positions, bar_labels, rotation = 90.0)
    
    fig.subplots_adjust(bottom = parameters.bottom_position_of_bar_diagram)
    
    ax.set_title('Difference in carbon per ha to first-best alternative', pad = parameters.titlepad)
    
    fig.savefig('bar_diagram_of_carbon_per_ha_differences.png')
    
    plt.close(fig)

