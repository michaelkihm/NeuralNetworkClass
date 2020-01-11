import numpy as np

def permute_data(X, y):
    """
    Permutes data with numpys permute function
    """
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]