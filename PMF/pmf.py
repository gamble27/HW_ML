import numpy as np


def fro(t: np.ndarray) -> float:
    """
    Frobenius norm of the matrix
    """
    return np.linalg.norm(t, ord='fro')


def PMF(X: np.ndarray, d: int, l: float, s: float = 1,
        max_iter: int = 100, print_cost: int = 0,
        pretrained_u: np.ndarray = None,
        pretrained_v: np.ndarray = None) -> tuple:
    """
    Probabilistic Matrix Factorization
    algorithm implementation
    :param X: an observed matrix, dim(X) = n x m
    :param l: regularization parameter (lambda)
    :param s: regularization parameter (standard deviation)
    :param d: number of latent features
    :param max_iter: maximum iterations of the algorithm, default 100
    :param print_cost: if > 0 prints cost function every n-th iteration of the algorithm,
                       if = 0 does not print the cost at all
    :param pretrained_u: if you want to run more epochs
                       with pretrained matrices,
                       please, specify both; default None
    :param pretrained_v: see param pretrained_u
    :return: U, V - latent features matrices
             with dim(U) = d x n, dim(V) = d X m
    """

    n, m = X.shape
    ind = np.ones(X.shape)
    ind[X == 0] = 0  # indicator matrix

    # initialize latent features matrices
    if pretrained_u is None and pretrained_v is None:
        v = np.random.normal(0, 1 / l, (d, m))
        u = np.zeros((d, n))
    else:
        v = pretrained_v
        u = pretrained_u

    # iterate through u,v
    for k in range(max_iter):
        # calculate U
        for i in range(n):
            u[:, i] = ((1/(l*s**2 + np.array(
                    [ind[i, j] * fro(v[:, j].reshape(d, 1)) ** 2 for j in range(m)]
            ).sum()) * (v @ (ind * X)[i, :].reshape((1, m)).T)).reshape((d,)))

        # calculate V
        for j in range(m):
            v[:, j] = ((1/(l*s**2 + np.array(
                    [ind[i, j] * fro(u[:, i].reshape(d, 1)) ** 2 for i in range(n)]
            ).sum()) * (u @ (ind * X)[:, j].reshape((1, n)).T)).reshape((d,)))

        # save this for later
        np.save('/home/olga/Projects/HW_ML/lab4/experiment/u.npy', u)
        np.save('/home/olga/Projects/HW_ML/lab4/experiment/v.npy', v)

        # compute cost
        cost = ((s**(-2)) * fro(ind*(X - u.T @ v))**2 + l * fro(u)**2 + l * fro(v)**2) / 2
        if print_cost and (k + 1) % print_cost == 0:
            print(f"{k+1} iteration cost: {np.log(cost):.5f}")

    return u, v


if __name__ == "__main__":
    # test case
    x = np.zeros((10, 20))
    x[4,5] = x[3,2] = x[0,15] = x[7,13] = 1
    x[6,8] = x[9,4] = x[4,17] = x[2,19] = 4
    U,V = PMF(x, d=3, l=.4, s=2, max_iter=10, print_cost=1)
    print((U.T @ V)[6,8])
