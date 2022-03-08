# Baseline Eclipse Mapping Code

This code does a forward model, simulates noise and then fits the simulated lightcurves.

## Forward model

Run the `make_eclipse_w_baseline.ipynb` notebook to generate a forward model with noise. Use Starry version 1.0 (my conda environment `BEMStarry1p0`)

## Bayesian Fit

The `run_example_fits.py` script does the inference. Use starry 1.2.0 (my conda environment `BEMRosetta2`)

## Installation (worked for starry 1.2.0)

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

## Installation for Starry 1.0
I used this to make the same forward map as earlier versions of the code. Took a lot of trial and error to get the versions right.

1. Set up a x86 environment (https://github.com/Haydnspass/miniforge#rosetta-on-mac-with-apple-silicon-hardware)

		CONDA_SUBDIR=osx-64 conda env create -n BEMStarry1p0 -f environment_starry1p0.yaml
		conda activate BEMStarry1p0
		conda env config vars set CONDA_SUBDIR=osx-64

2. Deactivate and reactivate to make sure the changes take hold:

		conda deactivate
		conda activate BEMStarry1p0

