import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    sigmoid function numpy implementation
    :param x: array with real values
    :return: 1/(1 + exp(-x))
    """
    return np.power(1 + np.exp(-x),-1)


def predict(X: np.ndarray, w:np.ndarray) -> np.ndarray:
    return sigmoid(X @ w)


def cost(y_true: np.ndarray, y_hat:np.ndarray) -> float:
    return -(y_true*np.log(y_hat) + (1-y_true)*np.log(1-y_hat)).sum() / y_true.shape[0]


def train_logit(X: np.ndarray, y: np.ndarray,
                lr:float = 0.1, max_iter: float = 100,
                print_cost=False) -> np.ndarray:
    n_samples = X.shape[0]
    dim = X.shape[1]
    w = np.random.standard_normal(size=(dim, 1))
    for i in range(max_iter):
        y_hat = predict(X, w)
        w = w - (lr/n_samples) * X.T @ (y_hat - y)
        if print_cost:
            print(f"{i + 1} iteration cost: {cost(y, y_hat):.4f}")
    return w
