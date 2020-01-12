from layer import Layer
from sigmoid import Sigmoid
from weightmultiply import WeightMultiply
from operation import Operation
from biasadd import BiasAdd
import numpy as np 


class Dense(Layer):
    '''
    A fully connected layer which inherits from "Layer"
    '''
    def __init__(self,neurons: int, activation: Operation = Sigmoid()):
        '''
        Requires an activation function upon initialization
        '''
        super().__init__(neurons)
        self.activation = activation

    def _setup_layer(self, input_: np.ndarray) -> None:
        '''
        Defines the operations of a fully connected layer.
        '''
        # pylint: disable=no-member
        if self.seed:
           np.random.seed(self.seed)

        self.params = []

        # weights
        self.params.append(np.random.randn(input_.shape[1], self.neurons))

        # bias
        self.params.append(np.random.randn(1, self.neurons))

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]

        return None