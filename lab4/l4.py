import numpy as np
import pandas as pd
from tqdm import trange


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

    # suggestion from the lecturer:
    Omega_u = [list(np.where(X[i, :] > 0)[0]) for i in
               range(n)]  # Omega_u[i] - масив індексів, для яких M_ij - observed
    Omega_v = [list(np.where(X[:, j] > 0)[0]) for j in range(m)]

    # iterate through u,v
    for k in range(max_iter):
        # calculate U
        for i in trange(n):
            u[:, i] = ((1/(l*s**2 + np.array(
                    [ind[i, j] * np.linalg.norm(v[:, j].reshape(d, 1), ord='fro') ** 2 for j in Omega_u[i]]
            ).sum()) * (v @ (ind * X)[i, :].reshape((1, m)).T)).reshape((d,)))

        # calculate V
        for j in trange(m):
            v[:, j] = ((1/(l*s**2 + np.array(
                    [ind[i, j] * np.linalg.norm(u[:, i].reshape(d, 1), ord='fro') ** 2 for i in Omega_v[j]]
            ).sum()) * (u @ (ind * X)[:, j].reshape((1, n)).T)).reshape((d,)))

        # save this for later
        # np.save('/home/olga/Projects/HW_ML/lab4/experiment/u.npy', u)
        # np.save('/home/olga/Projects/HW_ML/lab4/experiment/v.npy', v)

        # compute cost
        cost = ((s**(-2)) * np.linalg.norm(ind*(X - u.T @ v), ord='fro')**2 + l * np.linalg.norm(u, ord='fro')**2 + l * np.linalg.norm(v, ord='fro')**2) / 2
        if print_cost and (k + 1) % print_cost == 0:
            print(f"{k+1} iteration cost: {np.log(cost):.5f}")

    return u, v


# get data
df = pd.read_table("~/Projects/HW_ML/data/data-filtering/user_artists.dat")

# Approach 1. Work with rounded ln(X)
N, M = 1832, 17632
N1 = df['userID'].max()
M1 = df['artistID'].max()

X = np.zeros((N1, M1))
for _, row in df.iterrows():
    X[row['userID']-1, row['artistID']-1] = np.round(np.log(row['weight']))

# X[:20, 40:60] - validation set
Y = []
for i in range(20):
    for j in range(40, 60):
        if X[i,j] > 0:
            Y.append((i,j,X[i,j]))

# Y = np.array(Y)
X[:20, 40:60] = np.zeros(X[:20, 40:60].shape)

# np.save('/home/olga/Projects/HW_ML/lab4/experiment/x.npy', X)
# np.save('/home/olga/Projects/HW_ML/lab4/experiment/y.npy', np.array(Y))

print('beep')
# U = np.load('/home/olga/Projects/HW_ML/lab4/experiment/u.npy')
# V = np.load('/home/olga/Projects/HW_ML/lab4/experiment/v.npy')
U, V = PMF(X, d=100, l=2, s=1, max_iter=5, print_cost=1)#,
           # pretrained_u=U, pretrained_v=V)

np.save('/lab4/u.npy', U)
np.save('/lab4/v.npy', V)
X_hat = U.T @ V

print("pred vs true")
for i, j, x_true in Y:
    print(f"{X_hat[i,j]:.2f}    {x_true}")
