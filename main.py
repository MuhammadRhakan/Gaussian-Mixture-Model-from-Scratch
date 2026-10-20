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
parser.add_argument("--init", type=str, default='kmeans', choices=['kmeans', 'even_spaced'], help="Select type of parameters initialization, i.e. 'default', 'even_spaced'")
args = parser.parse_args()

def main():
    # 1. Load Dataset
    data_path = os.path.join('data', args.data)
    X = pd.read_csv(data_path)

    # 2. Train Gaussian Mixture 
    initializer_choices = {
        'kmeans': kmeans_initializer,               # Set mean by putting equal gap respectively
        'even_space': even_spaced_initializer       # Using standard technique of kmeans clustering
    }
    initializer = initializer_choices.get(args.init, kmeans_initializer)

    model = CustomGMM(n_components=args.n_components)
    gmm = model.fit(X, initializer=initializer, verbose=True)

    # 3. Generate new samples from trained parameters
    samples, components = gmm.generate_samples(1000)

    # 4. Retrieve trained parameters
    gmm_parameters = gmm.get_params()
    
    # 5. Predict new samples
    predictions = gmm.predict(samples)


if __name__ == "__main__":
    main()