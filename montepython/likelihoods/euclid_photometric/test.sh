#! /bin/bash

### This script shows how to run the euclid_photometric likelihood alone,
### with the provided input file and covariance matrix
### USAGE:  launch this script from the montepython main dir

INPUT=input/euclid_photometric_w0waMN.param
CHAINS=chains/euclid_photometric
COVMAT=covmat/euclid_photometric_w0waMN.covmat

echo "delete chains folder"
rm -rv $CHAINS
rm -rv ${CHAINS}_test

echo "running superpessimistic case"
cp -v montepython/likelihoods/euclid_photometric/euclid_photometric.data.pessimistic montepython/likelihoods/euclid_photometric/euclid_photometric.data

echo "Remove old fiducial file"
rm data/euclid_photometric_fiducial.npz

echo "Creating new fiducial file"
python montepython/MontePython.py run -p $INPUT -o $CHAINS -f 0 -N 1

echo "Testing chi-squared"
python montepython/MontePython.py run -p $INPUT -o ${CHAINS}_test -f 0 -N 1 --display-each-chi2

echo "Running chains"
python montepython/MontePython.py run -o $CHAINS -f 1.9 -N 100000 --update 100 --superupdate 20 -c $COVMAT
