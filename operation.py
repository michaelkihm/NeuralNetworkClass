import numpy as np 


class Operation(object):
    """
    Base class for an operation in a neural network
    """

    def __init__(self):
        pass

    def forward(self, input_: np.ndarray):
        """
        saves input instance variable
        Calls the output function
        """
        self.input_ = input_
        self.output = self._output()

        return self.output

    def backward(self, output_grad: np.ndarray):
        
        assert(self.output.shape == output_grad.shape) #check correct shape

        self.input_grad = self._input_grad(output_grad)

        assert(self.input_.shape == self.input_grad.shape)
        return self.input_grad

    def _output(self) -> np.ndarray:
        """
        Has to be implemented for each Operation
        """
        raise NotImplementedError

    def _input_grad(self, output_grad: np.ndarray) ->np.ndarray:
        """
        This method has to be implemented for each Operation
        """
        raise NotImplementedError
        
       