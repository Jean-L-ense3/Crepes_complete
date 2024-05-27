#############################
# Jean Littaye - 2024/05/27 #
# Ubuntu                    #
#############################

# This folder should contain 5 different files:
# spec-file.txt contains all the packages installed thanks to conda with the effective versions
# Dataset_Generator.py To generate all the necessary data sets.
# DA_method.py To apply the DA-based method on a data set.
# NN_method.py To train and validate a NN upon the generated data set.


# 1. Install the correct packages with their associated version with the spec-file.txt
#    -> In the command prompt: conda create --name MyEnv  --file spec-file.txt
#    -> Add the Lightning package that cannot be installed with conda: pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/release/stable.zip -U
# 2. Generate the different data sets: run Dataset_Generator.py
# 3. Use freely the different methods (run DA_method.py or NN_method.py)
