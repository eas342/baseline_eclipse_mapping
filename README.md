# Baseline Eclipse Mapping Code

This code does a forward model, simulates noise and then fits the simulated lightcurves.

## Forward model

Run the `make_eclipse_w_baseline.ipynb` notebook to generate a forward model with noise.

## Bayesian Fit

The `run_example_fits.py` script does the inference.

## Installation

It was tricky to get starry install on an M1 Mac for me.

I followed some advice from here to install it using Rosetta:
https://stackoverflow.com/questions/65901162/how-can-i-run-pyqt5-on-my-mac-with-m1chip

### Steps

1. Set up a x86 environment (https://github.com/Haydnspass/miniforge#rosetta-on-mac-with-apple-silicon-hardware)

		CONDA_SUBDIR=osx-64 conda create -n BEMRosetta2   # create a new environment
		conda activate BEMRosetta2
		conda env config vars set CONDA_SUBDIR=osx-64

2. Deactivate and reactivate to make sure the changes take hold:

		conda deactivate
		conda activate BEMRosetta2

3. Install Python

		conda install python=3.7

4. Install the required packages:

		pip install -r requirements.txt

The requirements file doesn't have any package version requirements so there's a possibility newer packages create problems. An alternative strategy is to install specific versions with:

		
		pip install -r packages_that_worked.txt
