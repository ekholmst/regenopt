#!/usr/bin/python3
#
# parameters.py
#
# Parameter values used by regenopt (regenopt.py)
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

from matplotlib import colors
from matplotlib.pyplot import cm
import numpy as np



#
# DBSCAN parameters. Use DBSCAN to do the microstand delineations,
# unless the external .geojson file is set to something else than None
# below.
#

dbscan_eps = 30.0
dbscan_min_samples = 4



#
# The external .geojson file which defines the microstand
# delineations
#

external_geojson_file_for_delineations = None



#
# Rot area "safety buffer" for DBSCAN, and the buffering for the full
# harvest area delineation for both delineation methods
#

rot_area_buffer_size = 10.0
full_harvest_area_buffer_size = 5.0



#
# Use BLV models with this interest rate
#

blv_interest_rate = 2



#
# Maximum allowed number of iterations when finding the alpha complex
# for a given set of stumps
#

max_iterations_for_alpha_optimization = 10000



#
# Parameters related to the visualizations
#

figure_size = (15, 12)
default_fontsize = 15

stem_markersize_tiny = 5.0
stem_markersize_small = 20.0
stem_markersize_medium = 50.0
stem_markersize_large = 100.0

rot_status_one_stem_marker_alpha = 0.5

fertility_class_tick_label_font_size = 10
fertility_class_visualization_alpha = 0.5

soil_type_tick_label_font_size = 10
soil_type_visualization_alpha = 0.5

terrain_raster_map_alpha = 0.8

delineation_visualization_alpha = 0.5
delineation_visualization_alpha_lighter = 0.2
thicker_line_linewidth_for_delineation_boundary = 5

titlepad = 30
labelpad = 15
plot_padding_in_x_and_y = 10.0

linewidth_for_1D_regression_function_plots = 3
color_carbon_1D_regression_function = 'b'
color_blv_1D_regression_function = 'r'
tickpad_1D_regression_function = 15
labelpad_1D_regression_function = 15
markersize_for_1D_plot_data_points = 10
markeredgewidth_for_1D_plot_data_points = 3

tickpad_2D_regression_function = 15
labelpad_2D_regression_function = 30
markersize_simulation_data_points_in_2D_plot = 100
linewidth_simulation_data_points_in_2D_plot = 3
alpha_3D_surface = 0.5

bar_width_in_bar_diagrams = 0.5
linewidth_for_bar_diagram = 5
bottom_position_of_bar_diagram = 0.50

cabin_to_crane_tip_position_line_width = 1

ymin_buffer = 1.0
ymax_buffer = 1.0

zmin_buffer = 1.0
zmax_buffer = 1.0

rot_area_buffer_resolution = 16
rot_area_buffer_cap_style = 1
rot_area_buffer_join_style = 1

full_harvest_area_buffer_resolution = 16
full_harvest_area_buffer_cap_style = 1
full_harvest_area_buffer_join_style = 1
full_harvest_area_delineation_color = 'grey'
full_harvest_area_delineation_alternative_color = 'green'
total_rot_area_delineation_color = 'red'



#
# Using the fertility class, soil type and terrain map rasters is
# optional
#

use_raster_data = False



#
# Define the stand province in order to read in the correct fertility
# class and soil type rasters when these are used
#

stand_province = 'Etelä-Karjala'



#
# If the rasters are not used, then the dominant fertility class of
# the stand is set to that which is given here
#

hard_coded_dominant_fertility_class = 3



#
# Paths for external input data
#

fertility_class_data_root = ""
soil_type_data_root = ""
map_sheet_root = ""
temperature_sum_data_file = "data_and_models/temperature_sum/easting_northing_tsum.txt"
blv_and_carbon_models_root = "data_and_models/models/"
simulation_results_root = "data_and_models/simulation_data/"



#
# Written labels for the fertility class values 0, 1, ..., 8
#

fertility_class_tick_labels = ['0 (N/A)', '1 (Lehto)', '2 (Lehtomainen kangas, OMT)', '3 (Tuore kangas, MT)', '4 (Kuivahko kangas, VT)', '5 (Kuiva kangas)', '6 (Karukkokangas)', '7 (Kalliomaa ja hietikko)', '8 (Lakimetsä ja tunturi)']

fertility_class_tick_locations = [0, 1, 2, 3, 4, 5, 6, 7, 8]

#
# Create a colormap for the fertility class data, for all the nine
# possible pixel values (0, 1, 2, ..., 8)
#

fertility_class_cmap = colors.ListedColormap(['k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'])

#
# Set the boundaries for the fertility class colors to be from -0.5,
# 0.5, 1.5, 2.5, ..., 8.5
#

fertility_class_cmap_bounds = np.arange(0, 10) - 0.5

fertility_class_cmap_norm = colors.BoundaryNorm(fertility_class_cmap_bounds, ncolors = 9)



#
# Written labels for the soil type values 0, 10, 11, 12, 20, ..., 80
#

soil_type_tick_labels = ['0 (N/A)', '10 (Keskikarkea tai karkea kangasmaa)', '11 (Karkea moreeni)', '12 (Karkea lajittunut maalaji)', '20 (Hienojakoinen kangasmaa)', '21 (Hienoainesmoreeni)', '22 (Hienojakoinen lajittunut maalaji)', '23 (Silttipitoinen maalaji)', '24 (Savimaa)', '30 (Kivinen keskikarkea tai karkea kangasmaa)', '31 (Kivinen karkea moreeni)', '32 (Kivinen karkea lajittunut maalaji)', '40 (Kivinen hienojakoinen kangasmaa)', '50 (Kallio tai kivikko)', '60 (Turvemaa)', '61 (Saraturve)', '62 (Rahkaturve)', '63 (Puuvaltainen turve)', '64 (Eroosioherkkä saraturve)', '65 (Eroosioherkkä rahkaturve)', '66 (Maatumaton saraturve)', '67 (Maatumaton rahkaturve)', '70 (Multamaa)', '80 (Liejumaa)']

soil_type_tick_locations = [0, 10, 11, 12, 20, 21, 22, 23, 24, 30, 31, 32, 40, 50, 60, 61, 62, 63, 64, 65, 66, 67, 70, 80]

#
# Create a colormap for the soil type data, for all the 24 possible
# pixel values (0, 10, 11, 12, 20, ..., 80)
#

soil_type_cmap = colors.ListedColormap([c for c in cm.rainbow(np.linspace(0, 1, 24))])

#
# Set the boundaries for the soil type colors to be from -0.5, 9.5, 10.5, 11.5, 12.5, 20.5, ..., 80.5
#

soil_type_cmap_bounds = [-0.5, 9.5, 10.5, 11.5, 12.5, 20.5, 21.5, 22.5, 23.5, 24.5, 30.5, 31.5, 32.5, 40.5, 50.5, 60.5, 61.5, 62.5, 63.5, 64.5, 65.5, 66.5, 67.5, 70.5, 80.5]

soil_type_cmap_norm = colors.BoundaryNorm(soil_type_cmap_bounds, ncolors = 24)
