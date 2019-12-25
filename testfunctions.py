import numpy as np 

def assert_same_shape(array: np.ndarray, array_grad: np.ndarray):
    assert array.shape == array_grad.shape