#############################
# Jean Littaye - 2025/10/22 #
# Ubuntu                    #
#############################

# This folder should contain 7 main files and 2 main folders:
# Article_plots.py plots the figures that mix both DA and NN results.
# Dataset_Generator.py To generate all the necessary data sets.
# DA_method.py To apply the DA-based method on a data set.
# DA_method_ensemble.py To apply the DA-based method on a data set considering several members.
# environment.yml contains all the packages installed with their effective version.
# Functions.py contains all the functions used to plot/analyse the data.
# NN_method.py To train and validate a NN upon the generated data set.

# The Data sets used in the article are those in the compressed file "Generated_Datasets.zip".
# The results used in the article are those in the compressed file "Res.zip".

# 1. Install the correct packages with their associated version with the environment.yml file.
# 2. Generate the different data sets: run Dataset_Generator.py
# 3. Use freely the different methods (run DA_method.py, DA_method_ensemble.py or NN_method.py)
# /!\ The Article_plots.py script will work only if results have been generated for each 12 scenarii with both DA and NN methods.
