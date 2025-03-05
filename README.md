# regenopt

A method and code for optimizing the regeneration of a spruce-dominated stand infected by Heterobasidion root rot disease. Please see the publication E. Holmström, J. Honkaniemi, A. Ahtikoski, T. Rajala, J. Hantula, T. Piri, J. Heikkinen, S. Suvanto, T. Räsänen, J.-A. Sorsa, K. Riekki, H. Höglund, A. Lehtonen, M. Peltoniemi, "Optimizing the regeneration of spruce-dominated stands suffering from Heterobasidion root rot in Finland" (https://doi.org/10.1016/j.compag.2025.110134) for details.

To run the code on the command line, give the following command:

`python3 regenopt.py stand.csv`

where `stand.csv` is the path to a file containing the stand data. See the header of regenopt.py as well as parameters.py for details. An example stand is provided under data_and_models/example_stand/.

Uses the following packages: numpy, matplotlib, PIL, json, alphashape, shapely, glob, sys, os, sklearn and datetime.

NB! If you are the owner of the harvester data and use this code, we strongly encourage you to share information on the location of root-rot infested areas within the stand with the forest owner.

Code distributed under the MIT license. Data licensed under CC BY 4.0. To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0/.

Tested on Ubuntu 18.04 / RHEL / Python 3.

This work was supported by the TyviTuho project funded by the Ministry of Agriculture and Forestry of Finland through the Catch the carbon research and innovation program (funding decision VN/5206/2021).

Contact: eero.holmstrom@luke.fi
