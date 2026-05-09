# This class is used to load the dataset and preprocess it for training and testing.

import pandas as pd

def download_dataset():
    """
    Download the dataset from Hugging Face and return it as a pandas DataFrame.
    """
    data = pd.read_csv("hf://datasets/ibm-research/argument_quality_ranking_30k/")
    return data

def get_data():
    """
    Get the dataset and preprocess it for training and testing.
    """
    data = download_dataset()
    # TODO: Data exploration and preprocessing
    return data