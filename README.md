# FoSTER-AGB

Official implementation of '**FoSTER-AGB: Forest-State Expert Routing for Interpretable GEDI–Landsat Aboveground Biomass Estimation across Unseen Ecoregions**'.

FoSTER-AGB is a forest-state-informed framework for cross-ecoregion aboveground biomass (AGB) estimation using GEDI structural, Landsat spectral, topographic, and observation-quality information.

## Overview

The framework includes:
- multisource feature encoding using an FT-Transformer backbone;
- latent forest-state representation;
- reliability-aware expert routing;
- state-conditioned residual correction;
- forest-state diagnostics and prediction-risk analysis.

## Data

The study uses publicly available GEDI–FIA Fusion data and Landsat observations.

GEDI–FIA Fusion:
https://doi.org/10.3334/ORNLDAAC/2417

## Reproducibility

This repository provides the implementation used for model training, evaluation, ablation experiments, and diagnostic analyses reported in the manuscript.

## License

This project is released under the MIT License.
