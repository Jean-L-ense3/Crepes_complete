#############################
# Jean Littaye - 2024/10/14 #
# Ubuntu                    #
#############################

# This folder should contain 6 main files and 2 main folders:
# spec-file.txt contains all the packages installed thanks to conda with the effective versions
# Dataset_Generator.py To generate all the necessary data sets.
# DA_method.py To apply the DA-based method on a data set.
# NN_method.py To train and validate a NN upon the generated data set.
# Functions.py contains all the functions used to plot/analyse the data.
# Article_plots.py plots the figures that mix both DA and NN results.

# The Data sets used in the article are those in the compressed file "Generated_Datasets.zip".
# The results used in the article are those in the compressed file "Res.zip".

# 1. Install the correct packages with their associated version with the spec-file.txt
#    -> In the command prompt: conda create --name MyEnv  --file spec-file.txt
#    -> Add the Lightning package that cannot be installed with conda: pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/release/stable.zip -U
# 2. Generate the different data sets: run Dataset_Generator.py
# 3. Use freely the different methods (run DA_method.py or NN_method.py)
# /!\ The Article_plots.py script will work only if results have been generated for each 12 scenarii with both DA and NN methods.
