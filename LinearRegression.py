import numpy as np

def forward_linear_regression(X_batch: np.ndarray,
                              Y_batch: np.ndarray,
                              weights):
    N = np.dot(X_batch,weights['W'])
    P = N + weights['B']

    loss = np.mean(np.power(P-Y_batch),2)

    forward_info  = {}
    forward_info['X'] = X_batch
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['Y'] = Y_batch

    return loss, forward_info


def loss_gradients(forward_info, weights):
    batch_size = forward_info['X'].shape

    dLdP = -2*(forward_info['Y'] - forward_info['P'])

    dPdN = np.ones_like(forward_info['N'])

    dPdB = np.ones_like(forward_info['B'])

    dLdN = dLdP * dPdN

    dNdW = np.transpose(forward_info['X'],(1,0))

    dLdW = dNdW @ dLdN

    dLdB = (dLdP * dPdB).sum(axis=0)

    loss_gradients = {}
    loss_gradients['B'] = dLdB
    loss_gradients['W'] = dLdW

    return  loss_gradients

