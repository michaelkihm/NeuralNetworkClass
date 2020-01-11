from loss import Loss
import numpy as np

class MeanSquaredError(Loss):

    def __init__(self) -> None:
        '''Pass'''
        super().__init__()

    def _output(self) -> float:
        '''
        Computes the per-observation squared error loss
        '''
        return np.sum(np.power(self.prediction - self.target, 2))/self.prediction.shape[0]

    def _input_grad(self) -> np.ndarray:
        '''
        Computes the loss gradient with respect to the input for MSE loss
        '''        
        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]