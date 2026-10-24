# Gaussian Mixture Model from Scratch

This repository demonstrates the fundamental of Gaussian Mixture Model, a probabilistic unsupervised algorithm that handles samples of instances from several sub-populations. This assumes all data samples are generated from a mixture of several underlying Gaussian distributions. 

## Main Usage
Execute Main Scripts:
```
python main.py --data "double_helix_data_1000.csv" --n_components 48 --init "even_spaced"
```

* `--data`: specify file name from located directory
* `--n_components`: number of Gaussian components
* `--init`: `kmeans` or `even_spaced` parameter initializatioans

The experiment in a notebook form is shown in `GMM_experiment.ipynb` file.

## Study Case: Drone-Based Staircase Monitoring System
A drone-based safety system to monitor visitors trajectory along a double-helix staircase: an 80-feet tower with 12-feet radius and 4-feet wide staircases. A dataset of 1000 of spatial coordinates was collected to represent visitor movement.

<p align ="center">
    <img src="/assets/double_helix.png" />
</p>

### Assumptions of Parameter
To account for the staircase's geometry, the initialization uses altitude to establish an orderinf of the spatial samples, whereas the longitude and latitude will remain redundant due to a circular pattern.
* Means: Initialized using evenly spaced indices across the altitude-ordered dataset to distribute Gaussian components throughout the structure.
* Covariances: Initialized as identity matrices because the initial spatial spread of each component was unknown.
* Mixture weights: Initialized uniformly, assuming no prior knowledge of the relative contribution of each component.

### Results
We compared the result of manual Gaussian Mixture from scratch implementation against the standard `scikit-learn` package. The algorithm converged through iterative training, and subsequently 1000 synthetic samples were generated from learned parameters to examine both approaches against the original samples.

<p align="center">
  <img src="/assets/rotating_3d_plot.gif" />
</p>