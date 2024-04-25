#!/usr/bin/python3
#
# regenopt.py
#
# Optimizes the regeneration of a spruce-dominated stand infected by
# Heterobasidion parviporum root rot disease.
#
# Reads in the stem-level data for a given stand. Then segments the
# stand into regions infected by root rot disease (rotten) and regions
# not infected by root rot disease (healthy). Next, using generalized
# Motti and Motti+Hmodel simulation results, finds the optimal choice
# of tree species to plant in the healthy region and in the rottten
# region, thus producing the recommended microstand setup for
# regenerating the stand. The goal is to maximize either economic
# value or carbon sequestration, subject to the condition that the
# stand becomes purified of root rot disease.
#
# The stem-level data must be of the following format (.csv):
#
# harvester cabin position latitude (deg, WGS84), harvester cabin position longitude (deg, WGS84), crane tip easting (m, ETRS-TM35FIN), crane tip northing (m, ETRS-TM35FIN), DBH (mm), species group ID (1 = pine, 2 = spruce, 3 = birch, 4 = other), commercial volume (m**3), sawlog volume (m**3), theoretical sawlog volume (m**3), rot status (0 = no rot, 1 = rotten)
#
# The first line of the .csv file should be a header, and this line is
# skipped when reading in the data.
#
# Uses functions.py and parameters.py.
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

import sys
import os



#
# Import parameters.py here, before the rest of the modules, so that
# you use the correct parameters.py file here and in functions.py.
#
# Try to import parameters.py from the local directory first. To
# accomplish this, insert the current working directory to the head of
# the system path list.
#

sys.path.insert(0, os.getcwd())

import parameters



#
# Revert the path back to look for other modules elsewhere than in the
# current working directory.
#

sys.path = sys.path[1:]



import functions
import numpy as np
from matplotlib.pyplot import cm
from sklearn.cluster import DBSCAN
from datetime import datetime



#
# Program usage
#

if len(sys.argv) < 2:

    print("Usage: %s [stem-level data set for a single stand (.csv)]" % sys.argv[0])
    exit(1)



print("")
print("Started regenopt.py at %s" % datetime.now())



#
# Assign command-line input parameters
#

input_data_file = str(sys.argv[1])



#
# Print out the values of the command-line parameters
#

print("")
print("Using file %s for the stem-level data" % input_data_file)



#
# If the external .geojson for the microstand delineations was given,
# use that to delineate the rotten areas in the stand. Otherwise, use
# DBSCAN.
#

if parameters.external_geojson_file_for_delineations is not None:

    delineation_method = 'external'

else:

    delineation_method = 'dbscan'


print("")
print("The delineation method is %s" % delineation_method)


if parameters.external_geojson_file_for_delineations is not None:

    print("")
    print("The rotten area delineation geojson file is %s" % parameters.external_geojson_file_for_delineations)



#
# See if we will be using external raster data or not
#

if parameters.use_raster_data:

    print("")
    print("Using external raster data")

else:

    print("")
    print("Not using external raster data")

    

#
# Save out the parameters.py file that will be used in this run
#

parameters_full_path = os.path.abspath(parameters.__file__)

print("")
print("The module parameters.py was loaded from the file %s. Now copying this file to parameters.out..." % parameters_full_path)

os.system("cp " + parameters_full_path + " parameters.out")

print("Done.")



#
# Read in the pure Motti and Motti+Hmodel simulation data for BLV (in
# units of kEUR/ha) and carbon (in units of 100 tCO2/ha) as a function
# of temperature sum (in units of 1000 d.d.) for plotting alongside
# the 1D and 2D regression functions.
#

print("")
print("Now reading in the simulation data for BLV per ha and carbon content per ha as a function of temperature sum and rot fraction of previous tree generation, separately for MT and OMT...")

pure_motti_data_points_birch_blv_mt, pure_motti_data_points_birch_blv_omt, pure_motti_data_points_pine_blv_mt, pure_motti_data_points_pine_blv_omt, pure_motti_data_points_birch_carbon_mt, pure_motti_data_points_birch_carbon_omt, pure_motti_data_points_pine_carbon_mt, pure_motti_data_points_pine_carbon_omt, motti_plus_hmodel_data_points_spruce_blv_mt, motti_plus_hmodel_data_points_spruce_blv_omt, motti_plus_hmodel_data_points_spruce_birch_blv_mt, motti_plus_hmodel_data_points_spruce_birch_blv_omt, motti_plus_hmodel_data_points_spruce_carbon_mt, motti_plus_hmodel_data_points_spruce_carbon_omt, motti_plus_hmodel_data_points_spruce_birch_carbon_mt, motti_plus_hmodel_data_points_spruce_birch_carbon_omt = functions.read_in_simulation_data()

print("Done reading in the simulation data.")



#
# Read in the models for BLV (in units of kEUR/ha) and carbon (in
# units of 100 tCO2/ha) as a function of temperature sum (in units of
# 1000 d.d) and rot fraction.

print("")
print("Now reading in the pre-created regression models for BLV per ha and carbon content per ha as a function of temperature sum and rot fraction of previous tree generation, separately for MT and OMT...")

birch_blv_vs_tsum_mt, birch_blv_vs_tsum_omt, pine_blv_vs_tsum_mt, pine_blv_vs_tsum_omt, birch_carbon_vs_tsum_mt, birch_carbon_vs_tsum_omt, pine_carbon_vs_tsum_mt, pine_carbon_vs_tsum_omt, spruce_blv_vs_tsum_and_frot_mt, spruce_blv_vs_tsum_and_frot_omt, spruce_birch_blv_vs_tsum_and_frot_mt, spruce_birch_blv_vs_tsum_and_frot_omt, spruce_carbon_vs_tsum_and_frot_mt, spruce_carbon_vs_tsum_and_frot_omt, spruce_birch_carbon_vs_tsum_and_frot_mt, spruce_birch_carbon_vs_tsum_and_frot_omt = functions.read_in_models()

print("Done reading in the models.")



#
# Visualize the 1D and 2D models (a total of 16 models)
#

print("")
print("Now plotting the models...")



#
# BLV, MT
#

functions.visualize_1D_regression_function(birch_blv_vs_tsum_mt, pure_motti_data_points_birch_blv_mt, parameters.color_blv_1D_regression_function, 'Temperature sum (1000 d.d.)', 'BLV (kEUR/ha)', 'Fitted model for birch, MT', 'blv_model_for_birch_mt.png')

functions.visualize_1D_regression_function(pine_blv_vs_tsum_mt, pure_motti_data_points_pine_blv_mt, parameters.color_blv_1D_regression_function, 'Temperature sum (1000 d.d.)', 'BLV (kEUR/ha)', 'Fitted model for pine, MT', 'blv_model_for_pine_mt.png')

functions.visualize_2D_regression_function(spruce_blv_vs_tsum_and_frot_mt, motti_plus_hmodel_data_points_spruce_blv_mt, 'Temperature sum (1000 d.d.)', 'Rot fraction', 'BLV (kEUR/ha)', 'Fitted model for spruce, MT', 'blv_model_for_spruce_mt')

functions.visualize_2D_regression_function(spruce_birch_blv_vs_tsum_and_frot_mt, motti_plus_hmodel_data_points_spruce_birch_blv_mt, 'Temperature sum (1000 d.d.)', 'Rot fraction', 'BLV (kEUR/ha)', 'Fitted model for spruce-birch, MT', 'blv_model_for_spruce_birch_mt')



#
# BLV, OMT
#

functions.visualize_1D_regression_function(birch_blv_vs_tsum_omt, pure_motti_data_points_birch_blv_omt, parameters.color_blv_1D_regression_function, 'Temperature sum (1000 d.d.)', 'BLV (kEUR/ha)', 'Fitted model for birch, OMT', 'blv_model_for_birch_omt.png')

functions.visualize_1D_regression_function(pine_blv_vs_tsum_omt, pure_motti_data_points_pine_blv_omt, parameters.color_blv_1D_regression_function, 'Temperature sum (1000 d.d.)', 'BLV (kEUR/ha)', 'Fitted model for pine, OMT', 'blv_model_for_pine_omt.png')

functions.visualize_2D_regression_function(spruce_blv_vs_tsum_and_frot_omt, motti_plus_hmodel_data_points_spruce_blv_omt, 'Temperature sum (1000 d.d.)', 'Rot fraction', 'BLV (kEUR/ha)', 'Fitted model for spruce, OMT', 'blv_model_for_spruce_omt')

functions.visualize_2D_regression_function(spruce_birch_blv_vs_tsum_and_frot_omt, motti_plus_hmodel_data_points_spruce_birch_blv_omt, 'Temperature sum (1000 d.d.)', 'Rot fraction', 'BLV (kEUR/ha)', 'Fitted model for spruce-birch, OMT', 'blv_model_for_spruce_birch_omt')



#
# Carbon, MT
#

functions.visualize_1D_regression_function(birch_carbon_vs_tsum_mt, pure_motti_data_points_birch_carbon_mt, parameters.color_carbon_1D_regression_function, 'Temperature sum (1000 d.d.)', 'Maximum carbon content (100 tCO2eq/ha)', 'Fitted model for birch, MT', 'carbon_model_for_birch_mt.png')

functions.visualize_1D_regression_function(pine_carbon_vs_tsum_mt, pure_motti_data_points_pine_carbon_mt, parameters.color_carbon_1D_regression_function, 'Temperature sum (1000 d.d.)', 'Maximum carbon content (100 tCO2eq/ha)', 'Fitted model for pine, MT', 'carbon_model_for_pine_mt.png')

functions.visualize_2D_regression_function(spruce_carbon_vs_tsum_and_frot_mt, motti_plus_hmodel_data_points_spruce_carbon_mt, 'Temperature sum (1000 d.d.)', 'Rot fraction', 'Maximum carbon content (100 tCO2eq/ha)', 'Fitted model for spruce, MT', 'carbon_model_for_spruce_mt')

functions.visualize_2D_regression_function(spruce_birch_carbon_vs_tsum_and_frot_mt, motti_plus_hmodel_data_points_spruce_birch_carbon_mt, 'Temperature sum (1000 d.d.)', 'Rot fraction', 'Maximum carbon content (100 tCO2eq/ha)', 'Fitted model for spruce-birch, MT', 'carbon_model_for_spruce_birch_mt')



#
# Carbon, OMT
#

functions.visualize_1D_regression_function(birch_carbon_vs_tsum_omt, pure_motti_data_points_birch_carbon_omt, parameters.color_carbon_1D_regression_function, 'Temperature sum (1000 d.d.)', 'Maximum carbon content (100 tCO2eq/ha)', 'Fitted model for birch, OMT', 'carbon_model_for_birch_omt.png')

functions.visualize_1D_regression_function(pine_carbon_vs_tsum_omt, pure_motti_data_points_pine_carbon_omt, parameters.color_carbon_1D_regression_function, 'Temperature sum (1000 d.d.)', 'Maximum carbon content (100 tCO2eq/ha)', 'Fitted model for pine, OMT', 'carbon_model_for_pine_omt.png')

functions.visualize_2D_regression_function(spruce_carbon_vs_tsum_and_frot_omt, motti_plus_hmodel_data_points_spruce_carbon_omt, 'Temperature sum (1000 d.d.)', 'Rot fraction', 'Maximum carbon content (100 tCO2eq/ha)', 'Fitted model for spruce, OMT', 'carbon_model_for_spruce_omt')

functions.visualize_2D_regression_function(spruce_birch_carbon_vs_tsum_and_frot_omt, motti_plus_hmodel_data_points_spruce_birch_carbon_omt, 'Temperature sum (1000 d.d.)', 'Rot fraction', 'Maximum carbon content (100 tCO2eq/ha)', 'Fitted model for spruce-birch, OMT', 'carbon_model_for_spruce_birch_omt')



print("Done plotting the models.")



#
# Read in the stem data for this stand. If a field is NA, set its
# value to -999999.9
#


print("")
print("Now reading in the stem data for this stand...")

stand_data = np.genfromtxt(input_data_file, delimiter = ",", missing_values = "NA", filling_values = -999999.9, skip_header = 1)

print("Done. Read in a total of %d lines with %d columns." % (stand_data.shape[0], stand_data.shape[1]))

    
    
#
# Store the variables of interest into separate arrays
#

cabin_position_latitude = stand_data[:, 0]
cabin_position_longitude = stand_data[:, 1]
easting_crane_tip = stand_data[:, 2]
northing_crane_tip = stand_data[:, 3]
dbh = stand_data[:, 4]
species_group_id = np.round(stand_data[:, 5]).astype(int)
commercial_volume = stand_data[:, 6]
sawlog_volume = stand_data[:, 7]
theoretical_sawlog_volume = stand_data[:, 8]
rotten_200_with_sawlog = np.round(stand_data[:, 9]).astype(int)



#
# Convert the forest machine cabin latitude, longitude readings to
# easting, northing
#

print("")
print("Now converting the forest machine cabin positions from latitude, longitude to easting, northing...")

easting_cabin_position, northing_cabin_position = functions.convert_cabin_positions_from_latlon_to_easting_northing(cabin_position_latitude, cabin_position_longitude)

print("Done converting the coordinates.")



#
# Print out some properties of interest for this stand
#

print("")
print("Here are some properties of interest for this stand:")
print("")



#
# Number of stems, in total and per species
#

n_pine_stems = list(species_group_id == 1).count(True)
n_spruce_stems = list(species_group_id == 2).count(True)
n_birch_stems = list(species_group_id == 3).count(True)
n_other_stems = list(species_group_id == 4).count(True)


print("Number of pine stems was %d" % n_pine_stems)
print("Number of spruce stems was %d" % n_spruce_stems)
print("Number of birch stems was %d" % n_birch_stems)
print("Number of other stems was %d" % n_other_stems)
print("")
print("---> Total number of stems was %d, length of stand data was %d" % (n_pine_stems + n_spruce_stems + n_birch_stems + n_other_stems, stand_data.shape[0]))



#
# Range of stem coordinates
#

stem_positions_min_easting = np.min(easting_crane_tip)
stem_positions_max_easting = np.max(easting_crane_tip)
stem_positions_min_northing = np.min(northing_crane_tip)
stem_positions_max_northing = np.max(northing_crane_tip)


print("")
print("The stem positions run from easting %f m to %f m and northing %f m to %f m" % (stem_positions_min_easting, stem_positions_max_easting, stem_positions_min_northing, stem_positions_max_northing))



#
# Mean DBH for each species
#

print("")

if n_pine_stems > 0:
    
    dbh_pine = np.mean(dbh[species_group_id == 1])
    print("Mean DBH of pine is %f mm" % dbh_pine)


if n_spruce_stems > 0:
    
    dbh_spruce = np.mean(dbh[species_group_id == 2])
    print("Mean DBH of spruce is %f mm" % dbh_spruce)

    
if n_birch_stems > 0:

    dbh_birch = np.mean(dbh[species_group_id == 3])
    print("Mean DBH of birch is %f mm" % dbh_birch)


if n_other_stems > 0:
    
    dbh_other = np.mean(dbh[species_group_id == 4])
    print("Mean DBH of other is %f mm" % dbh_other)

    

#
# Mean commercial volume recovered for each species
#

print("")

if n_pine_stems > 0:
    
    commercial_volume_pine = np.mean(commercial_volume[species_group_id == 1])
    print("Mean commercial volume for pine was %f m**3" % commercial_volume_pine)

if n_spruce_stems > 0:
    
    commercial_volume_spruce = np.mean(commercial_volume[species_group_id == 2])
    print("Mean commercial volume for spruce was %f m**3" % commercial_volume_spruce)

if n_birch_stems > 0:
    
    commercial_volume_birch = np.mean(commercial_volume[species_group_id == 3])
    print("Mean commercial volume for birch was %f m**3" % commercial_volume_birch)

if n_other_stems > 0:
    
    commercial_volume_other = np.mean(commercial_volume[species_group_id == 4])
    print("Mean commercial volume for other was %f m**3" % commercial_volume_other)

print("")



#
# Fraction of rotten stems, as given by Rotten200WithSawLog
#

n_rotten_spruce_sawlog_caliber_stems = np.sum(np.logical_and(species_group_id == 2, np.logical_and(theoretical_sawlog_volume > 0.0, rotten_200_with_sawlog == 1)).astype(int))
n_spruce_sawlog_caliber_stems = np.sum(np.logical_and(species_group_id == 2, theoretical_sawlog_volume > 0.0).astype(int))


    
#
# NB! If the stand has no rotten sawlog-caliber spruce stems, report
# this and exit. That's because these stands are currently not of
# interest to us.
#

if n_rotten_spruce_sawlog_caliber_stems == 0:

    print("ERROR! The stand has zero rotten, sawlog-caliber spruce stems. Exiting.")
    exit(1)


fraction_of_rotten_spruce_sawlog_caliber_stems = n_rotten_spruce_sawlog_caliber_stems / n_spruce_sawlog_caliber_stems


print("Number of sawlog-caliber spruce stems was %d" % n_spruce_sawlog_caliber_stems)
print("Number of rotten sawlog-caliber spruce stems was %d" % n_rotten_spruce_sawlog_caliber_stems)
print("---> Fraction of rotten sawlog-caliber spruce stems was %6.3f" % fraction_of_rotten_spruce_sawlog_caliber_stems)



#
# Total recovered sawlog volume and the sum of the stem-specific
# theoretical maximum recoverable sawlog volume, both for spruce
#

total_theoretical_sawlog_volume_spruce = np.sum(theoretical_sawlog_volume[species_group_id == 2])
total_sawlog_volume_spruce = np.sum(sawlog_volume[species_group_id == 2])

print("")
print("Theoretical maximum recoverable sawlog volume for spruce was %f m**3" % total_theoretical_sawlog_volume_spruce)

print("Total recovered sawlog volume for spruce was %f m**3" % total_sawlog_volume_spruce)

print("")
print("---> Based on these, a total of %f m**3 or %f m**3 per sawlog-caliber spruce stem was lost" % (total_theoretical_sawlog_volume_spruce - total_sawlog_volume_spruce, (total_theoretical_sawlog_volume_spruce - total_sawlog_volume_spruce) / n_spruce_sawlog_caliber_stems))



#
# If the use of external raster data is required, then read in the
# fertility class data and find the dominant fertility class for this
# stand. In addition, read in the soil type data and make sure that
# most of the stems in this stand are on mineral soil.
#

if parameters.use_raster_data:


    #
    # Read in and slice the fertility class data raster
    #

    print("")
    print("Now reading in the fertility class data...")

    sliced_fertility_class_as_array, sliced_fertility_class_data_min_easting, sliced_fertility_class_data_max_easting, sliced_fertility_class_data_min_northing, sliced_fertility_class_data_max_northing, fertility_class_pixel_size_x, fertility_class_pixel_size_y = functions.read_in_ffc_raster(parameters.fertility_class_data_root, "fertilityclass", stem_positions_min_easting, stem_positions_max_easting, stem_positions_min_northing, stem_positions_max_northing)

    print("")
    print("Done reading in the fertility class data.")



    #
    # Read in and slice the soil type data raster
    #

    print("")
    print("Now reading in the soil type data...")

    sliced_soil_type_as_array, sliced_soil_type_data_min_easting, sliced_soil_type_data_max_easting, sliced_soil_type_data_min_northing, sliced_soil_type_data_max_northing, soil_type_pixel_size_x, soil_type_pixel_size_y = functions.read_in_ffc_raster(parameters.soil_type_data_root, "soiltype", stem_positions_min_easting, stem_positions_max_easting, stem_positions_min_northing, stem_positions_max_northing)

    print("")
    print("Done reading in the soil type data.")



    #
    # Find the dominant fertility class under the tree stems
    #

    print("")
    print("Now finding the dominant fertility class under the tree stems of this stand.")



    #
    # Loop over the stems. For each stem, get the fertility class
    # underneath it. Then compute a histogram of these fertility
    # classes. The dominant fertility class is the one which is most
    # frequent in this histogram.
    #

    fertility_classes_under_stems = []
    
    for i in np.arange(0, easting_crane_tip.shape[0]):

        this_x = easting_crane_tip[i]

        this_y = northing_crane_tip[i]
        
        this_i_x_fertility_class_matrix = (np.round((this_x - sliced_fertility_class_data_min_easting) / fertility_class_pixel_size_x)).astype(int)
        
        this_i_y_fertility_class_matrix = (np.round(np.abs(this_y - sliced_fertility_class_data_max_northing) / fertility_class_pixel_size_y)).astype(int)
        
        this_fertility_class = sliced_fertility_class_as_array[this_i_y_fertility_class_matrix, this_i_x_fertility_class_matrix]
        
        fertility_classes_under_stems.append(this_fertility_class)


    fertility_classes_under_stems = np.array(fertility_classes_under_stems)


    print("")
    print("Here is a histogram of the fertility class values under the stems (value, count):")
    print("")


    values, counts = np.unique(fertility_classes_under_stems, return_counts = True)

    for i in np.arange(0, len(values)):

        print("(%d, %d)" % (values[i], counts[i]))


    i_dominant_fertility_class = np.argmax(counts)

    dominant_fertility_class = values[i_dominant_fertility_class]

    print("")
    print("Based on this, the dominant fertility class for this stand is %d" % dominant_fertility_class)



    #
    # Make sure the stems are mostly on mineral soil. If not, then exit.
    #

    print("")
    print("Now checking whether most of the stems are on mineral soil or not.")


    #
    # Loop over the stems. For each stem, get the soil type underneath it
    # and see whether this is mineral soil or peatland. Then compute a
    # histogram of these "True / False" values. If most of the stems are
    # on mineral soil, then continue. Otherwise quit, because the
    # simulations are only valid for mineral soils.
    #


    stem_is_on_mineral_soil = []

    soil_types_under_stems = []

    for i in np.arange(0, easting_crane_tip.shape[0]):

        this_x = easting_crane_tip[i]

        this_y = northing_crane_tip[i]

        this_i_x_soil_type_matrix = (np.round((this_x - sliced_soil_type_data_min_easting) / soil_type_pixel_size_x)).astype(int)

        this_i_y_soil_type_matrix = (np.round(np.abs(this_y - sliced_soil_type_data_max_northing) / soil_type_pixel_size_y)).astype(int)

        this_soil_type = sliced_soil_type_as_array[this_i_y_soil_type_matrix, this_i_x_soil_type_matrix]

        soil_types_under_stems.append(this_soil_type)


        if (this_soil_type >= 10 and this_soil_type <= 50) or this_soil_type == 70 or this_soil_type == 80:

            this_stem_is_on_mineral_soil = True

        else:

            this_stem_is_on_mineral_soil = False


        stem_is_on_mineral_soil.append(this_stem_is_on_mineral_soil)


    soil_types_under_stems = np.array(soil_types_under_stems)


    print("")
    print("Here is a histogram of the soil type values under the stems (value, count):")
    print("")


    values, counts = np.unique(soil_types_under_stems, return_counts = True)

    for i in np.arange(0, len(values)):

        print("(%d, %d)" % (values[i], counts[i]))



    print("")
    print("Here is a histogram of the truth value for whether a stem is on mineral soil or not (value, count):")
    print("")

    true_counts = stem_is_on_mineral_soil.count(True)
    false_counts = stem_is_on_mineral_soil.count(False)

    print("(True, %d)" % true_counts)
    print("(False, %d)" % false_counts)


    if true_counts > false_counts:    

        print("")
        print("Based on this, most of the stems are on mineral soil, and we are good to continue.")

    else:

        print("")
        print("Based on this, most of the stems are not on mineral soil. Exiting.")
        exit(1)


else:


    #
    # Set the dominant fertility class to its given, hard-coded value
    #
    
    dominant_fertility_class = parameters.hard_coded_dominant_fertility_class


    #
    # These are required by functions.create_maplike_plots()
    #

    sliced_fertility_class_as_array = None

    sliced_fertility_class_data_min_easting = None

    sliced_fertility_class_data_max_easting = None

    sliced_fertility_class_data_min_northing = None

    sliced_fertility_class_data_max_northing = None

    fertility_class_pixel_size_x = None

    fertility_class_pixel_size_y = None

    
    sliced_soil_type_as_array = None

    sliced_soil_type_data_min_easting = None

    sliced_soil_type_data_max_easting = None

    sliced_soil_type_data_min_northing = None

    sliced_soil_type_data_max_northing = None

    soil_type_pixel_size_x = None

    soil_type_pixel_size_y = None
    
    

#
# Find the approximate temperature sum for this stand
#

temperature_sum_for_stand = functions.find_temperature_sum_for_stand(easting_crane_tip, northing_crane_tip)

print("")
print("The temperature sum for this stand is %f degree days" % temperature_sum_for_stand)



#
# Then, split the stand into microstands, i.e., delineate the healthy
# and rotten areas in the stand.
#


if delineation_method == 'dbscan':

    
    #
    # To split the stand into microstands, first apply DBSCAN to the
    # rotten stems, and then fit alphashapes around the individual
    # clusters to form the rotten, i.e., unsafe areas.
    #

    print("")
    print("Now clustering the rotten sawlog-caliber spruce stems using DBSCAN...")


    rotten_spruce_stem_positions_array = np.hstack((np.expand_dims(easting_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], 1), np.expand_dims(northing_crane_tip[np.logical_and(species_group_id == 2, rotten_200_with_sawlog == 1)], 1)))

    db_result = DBSCAN(eps = parameters.dbscan_eps, min_samples = parameters.dbscan_min_samples, metric = 'euclidean').fit(rotten_spruce_stem_positions_array, sample_weight = None)

    db_labels = db_result.labels_

    db_set_of_unique_labels = set(db_labels)

    db_number_of_clusters = len(db_set_of_unique_labels)

    
    print("Done.")
    print("")

    print("The number of clusters, possibly including one for the outliers, is %d" % db_number_of_clusters)
    print("The cluster labels are as follows:", db_set_of_unique_labels)
    print("The total number of outliers is %d" % list(db_labels).count(-1))



    #
    # Create a color map for the rot clusters to use throughout the
    # program. The outliers, i.e., individual stem delineations, will
    # always assume the last color in this list.
    #

    rot_cluster_color_map = cm.rainbow(np.linspace(0, 1, db_number_of_clusters))


    #
    # For each cluster, including the possible set of outliers, create a
    # separate array of the format
    #
    # <easting (m) of stem 1> <northing (m) of stem 1> <cluster ID>
    # <easting (m) of stem 2> <northing (m) of stem 2> <cluster ID>
    # ...
    #
    # Store the created arrays into the following list:
    #
    # position_and_cluster_id_data_for_each_cluster_including_outliers
    #


    print("")
    print("Now creating a separate data matrix for each cluster, including the set of outliers...")
    print("")

    
    position_and_cluster_id_data_for_each_cluster_including_outliers = []

    
    for cluster_id in db_set_of_unique_labels:

        print("Processing cluster with label %d" % cluster_id)

        this_cluster_data = rotten_spruce_stem_positions_array[list(db_labels) == cluster_id, :]
        this_cluster_data = np.hstack((this_cluster_data, np.ones([this_cluster_data.shape[0], 1])*cluster_id))
    
        position_and_cluster_id_data_for_each_cluster_including_outliers.append(this_cluster_data)


    print("")
    print("Done. Created a total of %d matrices." % len(position_and_cluster_id_data_for_each_cluster_including_outliers))



    #
    # Delineate the microstands, i.e., segment the stand into regions
    # infested by root rot and, conversely, healthy regions.
    #

    
    print("")
    print("Now starting the microstand delineation process.")

    
    cluster_delineations, individual_stem_delineations, full_harvest_area_delineation, total_rot_area_delineation, total_healthy_area_delineation = functions.delineate_microstands_using_dbscan(position_and_cluster_id_data_for_each_cluster_including_outliers, species_group_id, theoretical_sawlog_volume, rotten_200_with_sawlog, easting_crane_tip, northing_crane_tip)

    
    print("")
    print("Done with the microstand delineation process.")


    
elif delineation_method == 'external':


    #
    # To split the stand into microstands, apply the delineations for
    # healthy and rotten areas, i.e., areas that are safe and unsafe
    # for spruce respectively, as given in the external .geojson file
    #

    
    #
    # These are required by functions.create_maplike_plots()
    #
    
    rot_cluster_color_map = None

    position_and_cluster_id_data_for_each_cluster_including_outliers = []

    cluster_delineations = []

    individual_stem_delineations = []
    
    
    
    #
    # Delineate the microstands
    #
    

    print("")
    print("Now starting the microstand delineation process.")

    
    full_harvest_area_delineation, total_rot_area_delineation, total_healthy_area_delineation = functions.delineate_microstands_using_external_geojson(species_group_id, theoretical_sawlog_volume, rotten_200_with_sawlog, easting_crane_tip, northing_crane_tip)


    print("")
    print("Done with the microstand delineation process.")
    
    
else:

    print("ERROR! Unknown delineation method %s requested. Exiting." % delineation_method)
    exit(1)

    

#
# As a reminder, here is how the delineations created above have been
# stored, in the case of using DBSCAN:
#
# - cluster_delineations, a list of tuples of the format (delineation shape (Polygon), cluster ID (int))
#
# - individual_stem_delineations, a list of tuples of the format (delineation shape (Polygon), cluster ID (int, always -1))
#
# - full_harvest_area_delineation, a delineation shape (Polygon)
#
# - total_rot_area_delineation, a delineation shape (Polygon or MultiPolygon)
#
# - total_healthy_area_delineation, a delineation shape (Polygon or MultiPolygon)
#
# In addition,
#
# position_and_cluster_id_data_for_each_cluster_including_outliers
#
# is a list of arrays, one for each rot cluster, including the set of
# outliers, each array being of the following format:
#
# <easting (m) of stem 1> <northing (m) of stem 1> <cluster ID>
# <easting (m) of stem 2> <northing (m) of stem 2> <cluster ID>
#
# These six objects are used in performing the optimization and
# creating the plots in the following.
#

#
# And here is how the delineations created above have been stored, in
# the case of using the delineations from the external .geojson file:
#
# - cluster_delineations, an empty list
#
# - individual_stem_delineations, an empty list
#
# - full_harvest_area_delineation, a delineation shape (Polygon)
#
# - total_rot_area_delineation, a delineation shape (Polygon or MultiPolygon)
#
# - total_healthy_area_delineation, a delineation shape (Polygon or MultiPolygon)
#
# In addition,
#
# position_and_cluster_id_data_for_each_cluster_including_outliers
#
# is an empty list.
#



#
# Set the BLV and carbon models to those of the dominant fertility
# class of this stand
#


if dominant_fertility_class == 2:


    print("")
    print("Using the OMT regression models for the BLV and carbon calculations.")

    
    birch_blv_vs_tsum = birch_blv_vs_tsum_omt
    
    birch_carbon_vs_tsum = birch_carbon_vs_tsum_omt

    pine_blv_vs_tsum = pine_blv_vs_tsum_omt

    pine_carbon_vs_tsum = pine_carbon_vs_tsum_omt

    spruce_blv_vs_tsum_and_frot = spruce_blv_vs_tsum_and_frot_omt

    spruce_carbon_vs_tsum_and_frot = spruce_carbon_vs_tsum_and_frot_omt

    spruce_birch_blv_vs_tsum_and_frot = spruce_birch_blv_vs_tsum_and_frot_omt

    spruce_birch_carbon_vs_tsum_and_frot = spruce_birch_carbon_vs_tsum_and_frot_omt

    
    
elif dominant_fertility_class == 3:

    
    print("")
    print("Using the MT regression models for the BLV and carbon calculations.")

    
    birch_blv_vs_tsum = birch_blv_vs_tsum_mt

    birch_carbon_vs_tsum = birch_carbon_vs_tsum_mt

    pine_blv_vs_tsum = pine_blv_vs_tsum_mt

    pine_carbon_vs_tsum = pine_carbon_vs_tsum_mt

    spruce_blv_vs_tsum_and_frot = spruce_blv_vs_tsum_and_frot_mt

    spruce_carbon_vs_tsum_and_frot = spruce_carbon_vs_tsum_and_frot_mt

    spruce_birch_blv_vs_tsum_and_frot = spruce_birch_blv_vs_tsum_and_frot_mt

    spruce_birch_carbon_vs_tsum_and_frot = spruce_birch_carbon_vs_tsum_and_frot_mt

    
    
else:

    print("ERROR! No models found for dominant fertility class %d. Exiting." % dominant_fertility_class)
    exit(1)

    

#
# Perform the optimization. Consider a fixed set of alternative
# regeneration plans. Compute the total BLV and carbon storage for
# each. Finally, order the alternatives, separately in terms of BLV
# and carbon, and report these results.
#


print("")
print("Now starting the regeneration optimization process.")


list_of_alternatives, indeces_from_lowest_to_highest_blv, blv_eur_for_the_alternatives, indeces_from_lowest_to_highest_carbon, carbon_tCO2_for_the_alternatives = functions.perform_optimization(species_group_id, theoretical_sawlog_volume, rotten_200_with_sawlog, easting_crane_tip, northing_crane_tip, birch_blv_vs_tsum, birch_carbon_vs_tsum, pine_blv_vs_tsum, pine_carbon_vs_tsum, spruce_blv_vs_tsum_and_frot, spruce_carbon_vs_tsum_and_frot, spruce_birch_blv_vs_tsum_and_frot, spruce_birch_carbon_vs_tsum_and_frot, temperature_sum_for_stand, full_harvest_area_delineation, total_rot_area_delineation, total_healthy_area_delineation)


print("Optimization completed.")



#
# Create plots of the stand and the results of the analysis and the
# optimization
#



print("")
print("Now creating plots of the stand and the results.")



#
# Create a set of "map-like" plots first
#



functions.create_maplike_plots(easting_crane_tip, northing_crane_tip, stem_positions_min_easting, stem_positions_max_easting, stem_positions_min_northing, stem_positions_max_northing, easting_cabin_position, northing_cabin_position, species_group_id, rotten_200_with_sawlog, theoretical_sawlog_volume, fraction_of_rotten_spruce_sawlog_caliber_stems, sliced_fertility_class_as_array, sliced_fertility_class_data_min_easting, sliced_fertility_class_data_max_easting, sliced_fertility_class_data_min_northing, sliced_fertility_class_data_max_northing, fertility_class_pixel_size_x, fertility_class_pixel_size_y, sliced_soil_type_as_array, sliced_soil_type_data_min_easting, sliced_soil_type_data_max_easting, sliced_soil_type_data_min_northing, sliced_soil_type_data_max_northing, soil_type_pixel_size_x, soil_type_pixel_size_y, cluster_delineations, individual_stem_delineations, full_harvest_area_delineation, total_rot_area_delineation, total_healthy_area_delineation, position_and_cluster_id_data_for_each_cluster_including_outliers, rot_cluster_color_map, temperature_sum_for_stand, delineation_method)



#
# Then, plot the results of the optimization as bar diagrams
#



functions.create_plots_for_optimization_results(list_of_alternatives, indeces_from_lowest_to_highest_blv, blv_eur_for_the_alternatives, indeces_from_lowest_to_highest_carbon, carbon_tCO2_for_the_alternatives, full_harvest_area_delineation.area / 1e4)



print("")
print("Done creating plots.")



#
# All done.
#

print("")
print("All done. Exiting.")
print("")

exit(0)
