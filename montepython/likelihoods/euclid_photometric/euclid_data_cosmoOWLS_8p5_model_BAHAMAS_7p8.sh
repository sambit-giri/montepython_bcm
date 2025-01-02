#! /bin/bash

### This script shows how to run the euclid_photometric likelihood alone,
### with the provided input file and covariance matrix
### USAGE:  launch this script from the montepython main direactory
### for instance: 'source montepython/likelihoods/euclid_photometric/test.sh'

COVMAT=covmat/euclid_photometric_w0waMN.covmat

# INPUT=input/euclid_photometric_w0waMN.param
# INPUT=input/euclid_bfc_fixed_hydrosim.param 
# INPUT=input/euclid_model_cosmoOWLS_8p5.param
INPUT=input/euclid_model_BAHAMAS_7p8.param

# CHAINS=chains/euclid_photometric
# CHAINS=chains/euclid_photometric_data_cosmoOWLS_8p5_model_cosmoOWLS_8p5
CHAINS=chains/euclid_photometric_data_cosmoOWLS_8p5_model_BAHAMAS_7p8

# echo "delete chains folder"
# rm -rv $CHAINS
# rm -rv ${CHAINS}_test

# echo "running superpessimistic case"
# cp -v montepython/likelihoods/euclid_photometric/euclid_photometric.data.pessimistic montepython/likelihoods/euclid_photometric/euclid_photometric.data

# echo "Remove old fiducial file"
# rm data/euclid_photometric_fiducial.npz

echo "Creating new fiducial file"
python montepython/MontePython.py run -p $INPUT -o $CHAINS -f 0 -N 1

cp data/euclid_photometric_cosmoOWLS_8p5.npz data/euclid_photometric_fiducial.npz

# echo "Testing chi-squared"
# python montepython/MontePython.py run -p $INPUT -o ${CHAINS}_test -f 0 -N 1 --display-each-chi2

echo "Running chains"
python montepython/MontePython.py run -o $CHAINS -f 1.9 -N 100000 --update 100 --superupdate 20 -c $COVMAT

echo "Plotting chains"
python montepython/MontePython.py info $CHAINS
