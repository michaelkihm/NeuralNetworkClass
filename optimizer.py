class Optimizer(object):
    '''
    Base class for a neural network optimizer.
    '''
    def __init__(self,
                 learning_rate: float = 0.01):
        '''
        Every optimizer must have an initial learning rate.
        '''
        self.learning_rate = learning_rate
        self.first = True
        self.step_number = 0

    def step(self) -> None:
        '''
        Every optimizer must implement the "step" function.
        '''
        raise NotImplementedError