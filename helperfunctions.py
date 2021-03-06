import numpy as np
from scipy.special import logsumexp

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



def softmax(x, axis=None) ->np.ndarray:
    """
    Computes sofmax of input vector
    """
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


def get_model_accurarcy(y_predicted :np.ndarray, y_true : np.ndarray) ->float:
    return np.sum(np.argmax(y_predicted,axis=1) == np.argmax(y_true,axis=1))/ y_predicted.shape[0]