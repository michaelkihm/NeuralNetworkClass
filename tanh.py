from operation import Operation
import numpy as np 

class Tanh(Operation):
    """
    Tanh activation function
    """
    def __init__(self) -> None:
        super().__init__()
    
    def _output(self) ->np.ndarray:
        return np.tanh(self.input_)

    def _input_grad(self, output_grad: np.ndarray) ->np.ndarray:
        return output_grad * (1 - self.output * self.output)
        
