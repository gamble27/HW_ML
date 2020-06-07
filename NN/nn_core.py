from typing import Callable, Union
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))


def ReLU_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)


def RMSE(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Root Mean Squared Error
    :param y_true: true labels
    :param y_hat: predicted NN output
    :return: RMSE value
    """
    return (((y_true - y_hat)**2).mean())**0.5

def CrossEntropy(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Cross Entropy Loss
    :param y_true: true labels
    :param y_hat: predicted NN output
    :return: RMSE value
    """
    return -(y_true * np.log(y_hat) +
            (1-y_true) * np.log(1 - y_hat)).sum()


class NNBinaryClassifier:
    def __init__(self, architecture: Union, inner_activation: str = "relu", outer_activation: str = "sigmoid"):
        """
        Initialize fully connected neural network
        :param architecture: [input_size, first layer size, ..., n-1 layer size];
                             the output is always one neuron
        :param inner_activation: specify activation function for inner layers:
                                 "relu" (default), "sigmoid", "tanh"
        :param outer_activation: activation function for the last neuron;
                                 same options as for inner_activation; default "sigmoid"
        """

        # initialize activation & its derivative
        self.__acf_out = self.activation_function(outer_activation)
        self.__acf_in  = self.activation_function(inner_activation)
        self.__adf_in  = self.activation_derivative(inner_activation)
        # self.__adf_out = self.activation_derivative(outer_activation)

        # initialize weights
        self.__inner_layers = len(architecture)
        self.__weights = {}
        self.__activated = {}  # coz we will not calculate the same thing multiple times
        # self.__activated["A0"] stands for input data
        # self.__activated["Ai"] stands for i-th layer output
        # self.__activated["argi"] stands for i-th activation argument, i.e. A[l-1]W[l] + b[l]
        for i in range(1, self.__inner_layers):
            self.__weights[f"b{i}"] = np.random.randn(1, architecture[i])  # np.zeros((1, architecture[i]))
            self.__weights[f"W{i}"] = np.random.randn(architecture[i-1], architecture[i])
            # self.__activated[f"A{i}"] = np.zeros((architecture[i-1], architecture[i])) # useless initiation
        self.__weights[f"W{self.__inner_layers}"] = np.random.randn(architecture[-1], 1)
        self.__weights[f"b{self.__inner_layers}"] = np.random.randn(1, 1)  # np.zeros((1,1))  # btw, this is a scalar

        # initialize training parameters (default values)
        self.__lr = 0.1
        self.__epochs = 100

    @classmethod
    def activation_function(cls, f: str) -> Callable:
        """
        Activation wrapper
        :param f: function. Can be "relu", "sigmoid", "tanh"
        :return: activation function.
        """
        if   f == "relu":
            acf = lambda x: np.maximum(x, 0)
        elif f == "sigmoid":
            acf = sigmoid
        elif f == "tanh":
            acf = np.tanh
        else:
            raise ValueError("Unknown activation function")
        return acf

    @classmethod
    def activation_derivative(cls, f: str) -> Callable:
        """
        Activation derivative wrapper
        :param f: function. Can be "relu", "sigmoid", "tanh"
        :return: derivative of the activation function
        """
        if   f == "relu":
            df = ReLU_derivative
        elif f == "sigmoid":
            df = lambda x: sigmoid(x)*sigmoid(-x)
        elif f == "tanh":
            df = lambda x: 4 / (np.exp(x)+np.exp(-x))**2
        else:
            raise ValueError("Unknown activation function")
        return df

    def fit(self, data: np.ndarray, labels: np.ndarray, lr: float = 0.1, epochs: int = 100, print_cost: int = 0):
        """
        fit the model to data using BGD
        :param data: training data, dim = n_samples x input_size
        :param labels: training labels, dim = n_samples x 1
        :param lr: learning rate, default 0.1
        :param epochs: number of epochs to run, default 100
        :param print_cost: print cost each specified number of epochs,
                           default 0 (doesn't print the cost at all)
        :return: None
        """
        self.__lr = lr
        self.__epochs = epochs
        for epoch in range(epochs):
            y_hat = self.__forward(data)
            self.__backward(labels, y_hat)
            if print_cost > 0 and epoch % print_cost == 0:
                print(f"{epoch + 1} epoch entropy: {CrossEntropy(labels, y_hat)}")
                print(f"{epoch + 1} epoch RMSE: {RMSE(labels, y_hat)}\n")

    def __forward(self, x: np.ndarray) -> np.ndarray:
        """
        forward propagation
        :param x: NN input, dim = n_samples x input_size
        :return: NN predicted output, dim = n_samples x 1
        """
        self.__activated["A0"] = x
        for layer in range(1, self.__inner_layers):
            self.__activated[f"arg{layer}"] = (
                    self.__activated[f"A{layer - 1}"] @ self.__weights[f"W{layer}"] +
                    self.__weights[f"b{layer}"]
            )
            self.__activated[f"A{layer}"] = self.__acf_in(
                self.__activated[f"arg{layer}"]
            )
        prediction = self.__acf_out(
            self.__activated[f"A{self.__inner_layers - 1}"] @
            self.__weights[f"W{self.__inner_layers}"] +
            self.__weights[f"b{self.__inner_layers}"]
        )
        return prediction

    def __backward(self, y: np.ndarray, y_hat:np.ndarray):
        """
        backward propagation algorithm
        :param y: true labels, dim = n_samples x input_size
        :param y_hat: NN output, dim = n_samples x input_size
        :return: None
        """
        # initialize delta
        delta: np.ndarray = y_hat - y
        n_samples = y_hat.shape[0]

        # update outer layer
        self.__weights[f"W{self.__inner_layers}"] = (
            self.__weights[f"W{self.__inner_layers}"] - self.__lr *
            ((self.__activated[f"A{self.__inner_layers-1}"]).T @ delta) / n_samples
        )
        self.__weights[f"b{self.__inner_layers}"] = (
            self.__weights[f"b{self.__inner_layers}"] - self.__lr *
            delta.sum(axis=0, keepdims=True) / n_samples
        )

        # update inner layers
        for layer in range(self.__inner_layers-1, 0, -1):
            delta = (
                (delta @ self.__weights[f"W{layer+1}"].T) *
                self.__adf_in(self.__activated[f"arg{layer}"])
            )
            self.__weights[f"W{layer}"] = (
                self.__weights[f"W{layer}"] - self.__lr *
                (self.__activated[f"A{layer-1}"].T @ delta) / n_samples
            ) # A0 is initialized as NN input in self.__forward(...)
            self.__weights[f"b{layer}"] = (
                    self.__weights[f"b{layer}"] - self.__lr *
                    delta.sum(axis=0, keepdims=True) / n_samples
            )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        get predicted labels
        :param x: NN input dim = n_samples x input_size
        :return: NN rounded output
        """
        y_hat = self.__forward(x)
        y_hat[y_hat > 0.5] = 1
        y_hat[y_hat <= 0.5] = 0
        return y_hat

    def save(self, name, path=None):
        """
        Save model architecture and weights
        :param name: file name
        :param path: optional, default None (saves to os.curdir)
        :return: None
        """
        if path is None:
            path = os.curdir

        if not name.endswith('.pkl'):
            name = name + '.pkl'

        with open(path+name, "wb") as f:
            pickle.dump(self.__weights, f)
            f.close()

    def decision_boundary(self, x, y, ax: plt.axis = None):
        """
        Plot decision boundary and labeled data
        :param x: data
        :param y: true labels
        :param ax: matplotlib axes (optional)
        :return: None
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        h = 0.02
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        ax.contourf(xx, yy, z, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        plt.show()


if __name__ == "__main__":
    # check out the model
    model = NNBinaryClassifier((2,7,7,3), inner_activation="tanh")
