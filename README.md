# Gaussian Mixture Model from Scratch

This repository demonstrates the fundamental of Gaussian Mixture Model, a probabilistic unsupervised algorithm that handles samples of instances from several sub-populations. This assumes all data samples are generated from a mixture of several underlying Gaussian distributions. 

## Usage
Execute Main Scripts:
```
python main.py --data "double_helix_data_1000.csv" --n_components 48 --init "even_spaced"
```

* `--data`: specify file name from located directory
* `--n_components`: number of Gaussian components
* `--init`: `kmeans` or `even_spaced` parameter initializatioans

The experiment in a notebook form is shown in `GMM_experiment.ipynb` file.

## Experimental Example
We compared the result of manual Gaussian Mixture from scratch implementation against the standard `scikit-learn` package. The algorithm converged through iterative training, and subsequently 1000 synthetic samples were generated from learned parameters to examine both approaches against the original samples.

![GMM Result](/assets/rotating_3d_plot.gif)

