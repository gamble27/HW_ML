import numpy as np
# from benchmark import Benchmark
from logit.logit_functional import sigmoid


class Logit:
    """
    performs logistic regression
    with batch gradient descent
    """
    def __init__(self, max_iter: int = 100, lr: float = 0.1, batch_size: int = 32):
        """
        class instance and hyper parameters' initialization
        :param max_iter: maximum iterations for gradient descent, default 100
        :param lr: learning rate, default 0.01
        :param batch_size: mini-batch size
        :return: initialized class instance
        """
        # variables pre-initialisation
        self.__X = None
        self.__y = None

        # hyper parameters initialization
        self.__max_iter = max_iter
        self.__lr = lr
        self.__b_size = batch_size

        # weights pre-initialization
        self.__w = None

    def __cost(self, y_hat: np.ndarray, y_true: np.ndarray) -> float:
        """
        computes cost function
        :param y_true: true labels
        :param y_hat: prediction array with shape (m_samples, 1)
        :return: cost value
        """
        return -np.sum(y_true * np.log(y_hat) +
                       (1-y_true) * np.log(1-y_hat)) / y_true.shape[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        make a prediction
        :param X: variables
        """
        y_hat: np.ndarray = sigmoid(X @ self.__w)
        y_hat[y_hat > 0.5] = 1.
        y_hat[y_hat < 1.] = 0.
        return y_hat

    # @Benchmark
    def fit(self, X: np.ndarray, y:np.ndarray):
        """
        fit model to data
        :param X: variables - ndarray with shape (m_samples, n_dim)
        :param y: labels - ndarray with shape (m_samples, 1)
        :return: self
        """
        # initialize variables and labels
        self.__X = X
        self.__y = y

        # initialize weights
        self.__w = np.random.standard_normal(size=(X.shape[1], 1))
        # self.__w = np.zeros((X.shape[1], 1))

        # batch counter
        n_batch = 0

        for i in range(self.__max_iter):
            # fetch mini-batch
            if n_batch + self.__b_size < self.__X.shape[0]:
                x = self.__X[n_batch : n_batch+self.__b_size, :]
                y = self.__y[n_batch : n_batch+self.__b_size, :]
                n_batch += 1
            else:
                x = self.__X[n_batch:, :]
                y = self.__y[n_batch:, :]
                n_batch = 0

            # compute outputs
            a = x @ self.__w
            y_hat = sigmoid(a)

            # compute cost
            # cost = self.__cost(y_hat, y)
            # print(f"{i + 1} iteration cost: {cost:.4f}")

            # update weights
            hmmm = x.T @ (y_hat-y)
            self.__w = self.__w - self.__lr * (hmmm) / y.shape[0]

        return self
