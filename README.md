# FoSTER-AGB

**Forest-State Expert Routing for Interpretable GEDI–Landsat Aboveground Biomass Estimation across Unseen Ecoregions**

### Model implementation

- `ft_transformer.py`: FT-Transformer feature encoding and latent representation extraction.
- `main_model.py`: FoSTER-AGB architecture, including reference calibration, expert routing, and residual correction.

## Data Availability

The study uses publicly available GEDI–FIA Fusion data and Landsat observations.

**GEDI–FIA Fusion dataset:**  
https://doi.org/10.3334/ORNLDAAC/2417

**Landsat Collection 2 Level-2:**  
https://developers.google.com/earth-engine/datasets/catalog/landsat

## Implementation

The repository provides the core FoSTER-AGB neural network architecture and supporting data-processing components.

The network is implemented in PyTorch. Its main prediction interface accepts GEDI structural features, spectral features, environmental features, and a five-dimensional uncertainty descriptor.


