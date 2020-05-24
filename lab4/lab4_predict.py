import numpy as np

U = np.load('/home/olga/Projects/HW_ML/lab4/experiment/u.npy')
V = np.load('/home/olga/Projects/HW_ML/lab4/experiment/v.npy')

Y = np.load('/home/olga/Projects/HW_ML/lab4/experiment/y.npy')
Y = list(Y)

X_hat = U.T @ V

print("pred vs true")
for y in Y:
    i, j, x_true = y
    i = int(i)
    j = int(j)

    print(f"{X_hat[i,j]:.2f}    {x_true}")
