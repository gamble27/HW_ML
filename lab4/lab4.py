import numpy as np
import pandas as pd
from PMF import PMF


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

np.save('/home/olga/Projects/HW_ML/lab4/experiment/x.npy', X)
np.save('/home/olga/Projects/HW_ML/lab4/experiment/y.npy', np.array(Y))

print('beep')
U = np.load('/home/olga/Projects/HW_ML/lab4/experiment/u.npy')
V = np.load('/home/olga/Projects/HW_ML/lab4/experiment/v.npy')
U, V = PMF(X, d=10, l=2, s=1, max_iter=1, print_cost=1,
           pretrained_u=U, pretrained_v=V)

# np.save('/home/olga/Projects/HW_ML/lab4/experiment/u.npy', U)
# np.save('/home/olga/Projects/HW_ML/lab4/experiment/v.npy', V)
X_hat = U.T @ V

print("pred vs true")
for i, j, x_true in Y:
    print(f"{X_hat[i,j]:.2f}    {x_true}")
