import os
import argparse
import numpy as pd
import pandas as pd

from pathlib import Path
from model import CustomGMM
from initializer import *


parser = argparse.ArgumentParser(description="Run Gaussian Mixture Model")
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--n_components", type=int, required=True)
args = parser.parse_args()

def main():
    # 1. Load Dataset
    data_path = os.path.join('data', args.data)
    X = pd.read_csv(data_path)

    # 2. Train Gaussian Mixture 
    """
    Initializer Type
        • even_spaced_initializer   -> set mean by putting equal gap respectively
        • kmeans_initializer        -> using standard technique of kmeans clustering 
    """
    model = CustomGMM(n_components=args.n_components)
    gmm = model.fit(X, initializer=even_spaced_initializer, verbose=True)

    # 3. Generate new samples from trained parameters
    samples, components = gmm.generate_samples(1000)

    # 4. Retrieve trained parameters
    gmm_parameters = gmm.get_params()

    # Example of usage
    # gmm_parameters.means      ==> return mean of each sample
    # gmm_parameters.weights    ==> return weights of each sample
    
    # 5. Predict new samples
    predictions = gmm.predict(samples)

    # Example of usage
    # predictions.component         ==> component assigned to each sample
    # predictions.component_scores  ==> probability score


if __name__ == "__main__":
    main()