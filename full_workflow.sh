# This script runs the simulation with the current config file, with infiltration, creates an animation of eta and clears the output folder

#!/bin/bash

# first need to change malago_road.txt into a csv file by hand. Then run 

python generate_csv_files.py

python Bristol_AI4SWE.py --infiltration

python create_plots.py --animate

python clear_output_folder