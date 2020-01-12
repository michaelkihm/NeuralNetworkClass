import numpy as np

def permute_data(X, y):
    """
    Permutes data with numpys permute function
    """
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]




def to_2d_np(a: np.ndarray, 
          type: str="col") -> np.ndarray:
    '''
    Turns a 1D Tensor into 2D
    '''

    assert a.ndim == 1, \
    "Input tensors must be 1 dimensional"
    
    if type == "col":        
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)

