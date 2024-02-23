#! /bin/bash

### USAGE:  launch this script from the montepython main dir

INPUT=input/euclid_spectropscopic_w0waMN.param
CHAINS=chains/euclid_spectroscopic/
Covmat=covmat/euclid_spectroscopic_w0waMN.covmat

echo "delete chains folder"
rm -rv $CHAINS

echo "running superpessimistic case"
cp -v montepython/likelihoods/euclid_spectroscopic/euclid_spectroscopic.data.superpessimistic montepython/likelihoods/euclid_spectroscopic/euclid_spectroscopic.data

echo "Creating fiducial"
python montepython/MontePython.py run -p $INPUT -o $CHAINS -f 0 -N 1
echo "Testing chi-squared"
python montepython/MontePython.py run -p $INPUT -o $CHAINS -f 0 -N 1 --display-each-chi2

echo "Running chains"
python montepython/MontePython.py run -o $CHAINS -f 1.9 -N 100000 --update 100 --superupdate 20 -c $Covmat

