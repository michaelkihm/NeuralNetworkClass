from paramoperation import ParamOperation
import numpy as np 

class WeightMultiply(ParamOperation):
    """
    Weight multiplication
    """

    def __init__(self, W: np.ndarray):
        super().__init__(W)

    def _output(self) -> np.ndarray:
        """
        Compute output
        """
        return self.input_ @ self.param

    def _input_grad(self, output_grad: np.ndarray) ->np.ndarray:
        """
        Compute input gradient
        """
        return np.dot(output_grad, np.transpose(self.param,(1,0)))

    def _param_grad(self, output_grad: np.ndarray) ->np.ndarray:
        """
        Compute parameter gradient
        """
        return np.transpose(self.input_, (1,0)) @ output_grad


    
