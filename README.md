# Baseline Eclipse Mapping Code

This code does a forward model, simulates noise and then fits the simulated lightcurves.

## Forward model

Run the `make_eclipse_w_baseline.ipynb` notebook to generate a forward model with noise.

## Bayesian Fit

The `run_example_fits.py` script does the inference.

## Requirements

This repository hasn't been cleaned up as a proper package should be. The requirements.txt contains a dump of the pip freeze output of versions that work.

