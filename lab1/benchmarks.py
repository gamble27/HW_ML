import time
import numpy as np
import pandas as pd
from tqdm import  tqdm
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LinearRegression


class Benchmark:
    """
    a custom benchmark class
    with runtime getter method
    """
    def __init__(self, function):
        self.__f = function
        self.__runtime  = 0

    def __call__(self, *args, **kwargs):
        start = time.time()

        values = self.__f(*args, **kwargs)

        end = time.time()
        self.__runtime = end - start

        return values

    def get_runtime(self):
        return self.__runtime


def generate_regressors(dim: int, betas: np.ndarray) -> pd.DataFrame:
    """
    generate 3 variables X1, X2, X3 with dimensionality dim and
    Y = b1*X1 + b2*X2 + b3*X3 + eps,
    where eps is normal error with std = 0.1

    :param dim: variable dimensionality
    :param betas: 3 coefficients for regression, shape=(3,)
    :return: a data frame containing variables and response
    """

    if betas.shape != (3,):
        raise ValueError(f"Expected betas.shape to be (3,), got {betas.shape}")

    result = pd.DataFrame()
    for i in range(1, 4):
        result[f'X{i}'] = np.random.normal(size=dim)

    result['Y'] = (betas[0]*result['X1'] +
                   + betas[1]*result['X2'] + betas[2]*result['X3'] +
                   np.random.normal(scale=0.1, size=dim))

    return result


@Benchmark
def OLS_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Ordinary Least Squares
    using matrix formula

    :param X: variables
    :param y: response
    :return: regression coefficients
             such that y = Xb

    """
    return np.matmul(
        np.matmul(
            np.linalg.inv(np.matmul(X.T, X)),
            X.T),
        y)


@Benchmark
def OLS_sm(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Ordinary Least Squares
    using statsmodels library

    :param X: variables
    :param y: response
    :return: regression coefficients
             such that y = Xb
    """
    return OLS(y, exog=X).fit().params


@Benchmark
def OLS_skl(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Ordinary Least Squares
    using sklearn library

    :param X: variables
    :param y: response
    :return: regression coefficients
             such that y = Xb
    """
    return LinearRegression().fit(X, y).coef_


@Benchmark
def OLS_GD(X: np.ndarray, y: np.ndarray,
           lr=0.1, max_iter=1000, accuracy=1) -> np.ndarray:
    """
    Ordinary Least Squares
    using Gradient Descent

    :param X: variables
    :param y: response
    :param lr: learning rate, optional, default 0.1
    :param max_iter: maximum iterations, optional, default 1000
    :param accuracy: desired accuracy level, default 1 for this toy dataset
    :return: regression coefficients
             such that y = Xb
    """
    # randomly initialize weights
    b = np.random.random((X.shape[1], 1)) * 42  # because I can do this :]

    # run GD
    running = True
    i = 0
    while running:
        # get prediction
        prediction = np.matmul(X,b)

        # residuals
        error = prediction - y

        # update weights. this has
        # additional complexity,
        # but I haven't figured out
        # how to make it better
        for i in range(b.shape[0]):
            b[i] = b[i] - lr * np.dot(
                error.reshape((X.shape[0],)),
                X[:,i]
            ) / y.shape[0]

        # error
        error = (error*error).sum() / error.shape[0]

        i += 1
        if i > max_iter or error < accuracy:
            running = False

    return b


def experiment(N: int, n_times: int, betas: np.ndarray) -> tuple:
    """
    performs each OLS method n_times
    and returns mean values of runtime

    :param N: variables dimensionality
    :param n_times: how many times to run each function
    :param betas: 3 coefficients for regression, shape=(3,)
    :return: mean values of runtime
    """
    # check betas
    if betas.shape != (3,):
        raise ValueError(f"Expected betas.shape to be (3,), got {betas.shape}")

    # generate and split the data
    df = generate_regressors(N, betas)
    y = df['Y']
    x = df.copy()
    del x['Y']
    x = x.to_numpy()
    y = y.to_numpy().reshape((N, 1))

    # run each function on benchmarks
    functions = [OLS_matrix, OLS_sm,
                 OLS_skl, OLS_GD]
    means = np.zeros((4,))
    for i, f in enumerate(tqdm(functions, desc=f'functions for dim={N}')):
        for _ in tqdm(range(n_times), desc=f'experiments with {f}'):
            b = f(x, y)
            means[i] += f.get_runtime()

    return tuple(means/n_times)


if __name__ == "__main__":
    # settings
    beta = np.random.random((3,)) * 42
    dimensions = [7, 70, 700, 7000,
                  42000, 142000, 1420000,
                  14200000]
    # how many times we run each function
    n_times = 10

    # a tuple of mean runtime values per experiment
    collected_runtime = []

    # run benchmarks using experiment function
    for dim in tqdm(dimensions, desc=f'{dimensions}'):
        collected_runtime. append(experiment(dim, n_times, beta))

    # wrap results into a fancy data frame
    df = pd.DataFrame(collected_runtime)
    df['dim'] = dimensions
    df.columns = ['matrix', 'statsmodels', 'sklearn', 'GD', 'dim']
    print(df)

    # save results
    df.to_csv('benchmarks.csv')
