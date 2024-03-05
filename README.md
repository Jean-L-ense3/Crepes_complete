#############################
# Jean Littaye - 2024/03/03 #
# Under Ubuntu                    #
#############################

# Crepes_complete
Short description:
Repository for the paper "Learning-based calibration of ocean carbon models to tackle physical forcing uncertainties and observation sparsity."

Longer description:
This study is part of the PhD project "Carbon REconstructed Per an Emulator that is Supervised" (Carbone REconstruit Par Emulateur Supervisé).
It contains 5 different files:
- spec-file.txt contains all the packages installed thanks to conda with the effective versions
- Dataset_Generator_DA.py To generate the data sets for the DA-based method.
- Dataset_Generator_NN.py To generate the data sets for the NN-based method.
- DA_method.py To apply the DA-based method on a data set.
- NN_method.py To train and validate a NN upon the generated data set.


For a use without errors:
1. Install the correct packages with their associated version with the spec-file.txt
   -> In the command prompt: conda create --name MyEnv  --file spec-file.txt
   -> Add the Lightning package that cannot be installed with conda: pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/release/stable.zip -U
2. Generate the different data sets: run Dataset_Generator_NN.py and Dataset_Generator_DA.py (start with NN because DA needs mean and std of the NN data set to create its test data set)
3. Use freely the different methods (run DA_method.py or NN_method.py)
