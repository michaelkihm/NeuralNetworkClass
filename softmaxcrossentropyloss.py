import numpy as np 
from loss import Loss
from helperfunctions import softmax

class SoftmaxCrossEntropy(Loss):
    """
    Defines softmax cross entropy loss
    """
    def __init__(self, eps: float=1e-9) -> None:
        super().__init__()
        self.eps = eps
        self.single_class = False

    def _output(self) -> float:

        # applying the softmax function to each row (observation)
        softmax_preds = softmax(self.prediction, axis=1)

        # clipping the softmax output to prevent numeric instability
        self.softmax_preds = np.clip(softmax_preds, self.eps, 1 - self.eps)

        # actual loss computation
        softmax_cross_entropy_loss = ( -1.0 * self.target * np.log(self.softmax_preds) - (1.0 - self.target) * np.log(1 - self.softmax_preds) )

        return np.sum(softmax_cross_entropy_loss) / self.prediction.shape[0]

    def _input_grad(self) -> np.ndarray:
        return (self.softmax_preds - self.target) / self.prediction.shape[0]

        


