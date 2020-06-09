import numpy as np
import pandas as pd
import missingno as msno


def PMF(X: np.ndarray, d: int, l: float, s: float = 1,
		max_iter : int = 100 , print_cost : bool = False) -> tuple :
	n, m = X.shape
	ind = np.ones ( X.shape )
	ind [X == 0] = 0 # indicator matrix

	# u,v initialization
	v = np.random.normal (0 , 1 / l, ( d, m))
	u = np.zeros (( d, n))
	# v = np.load('v.npy')
	# u = np.load('u.npy')

	# suggestion from Moskanova:
	Omega_u = [list(np.where(X[i,:] > 0)[0]) for i in range(n)] # Omega_u[i] - масив індексів, для яких M_ij - observed
	Omega_v = [list(np.where(X[:,j] > 0)[0]) for j in range(m)]

	for k in range ( max_iter ):

		for i in range (n):
			# b = v[:, 1]
			# k = np.linalg.norm(b.reshape(d, 1), ord='fro')
			u[: , i] = ((1 /(l* s **2 + np.array (
							[ ind [ i, j] * np.linalg.norm(v [: , j].reshape(d, 1), ord='fro') ** 2
							for j in Omega_u[i] ]) .sum () ) *
						(v @ (X [i, :] .reshape ((1 , m)) .T ))) .reshape (( d, )))

		for j in range (m):
			v[: , j] = ((1 /(l* s **2 + np.array (
							[ ind [ i, j] * np.linalg.norm(u [: , i].reshape(d, 1), ord='fro') ** 2
							for i in Omega_v[j] ]) .sum () ) *
						(u @ (X [: , j] .reshape ((1 , n)) .T ))) .reshape (( d, )))

		np.save('u.npy', u)
		np.save('v.npy', v)

		cost = (( s **( -2) ) * np.linalg.norm( ind *(X - u.T @ v), ord='fro') **2 +
				l * np.linalg.norm (u, ord='fro') **2 + l * np.linalg.norm (v, ord='fro') **2) / 2
		if print_cost:
			print(f"{k + 1} iteration cost : {np.log(cost):.3f}")

	return u, v

if __name__ == '__main__':
	# df = pd.read_csv('/home/olga/Projects/HW_ML/valentino/movie-ratings.txt',
	#                    names=['userID', 'movieID', 'genreID',
	#                    		  'reviewID', 'rating', 'date'])
	# d = df[['userID', 'movieID', 'rating']]
	# Data = d.iloc[50000:70000]
	# # save cropped dataset
	# Data.to_csv('/home/olga/Projects/HW_ML/valentino/data_new.csv')

	# retrieve dataset from csv
	Data = pd.read_csv('data_new.csv')

	users = list(set(Data['userID'])) # list of unique users IDs
	n = len(users) # number of rows in observed matrix
	movies = list(set(Data['movieID'])) # list of unique films IDs
	m = len(movies) # number of columns in observed matrix

	# form the matrix of observed ratings M
	M = np.zeros((n,m))
	for _, row in Data.iterrows():
		i = users.index(int(row['userID']))
		j = movies.index(int(row['movieID']))
		M[i,j] = int(row['rating'])

	# create validation set
	Y = []
	for i in range(30,60):
		for j in range(0,60):
			if M[i, j] > 0:
				Y.append((i, j, M[i, j]))

	# print(M.shape)
	# print(Y)
	# exit(0)

	M[30:60, :60] = np.zeros((30, 60))

	# visualize M
	msno.matrix(pd.DataFrame(M))

	# train & test
	# U, V = PMF(M, d=100, l=1, s=.5, max_iter=20, print_cost=True)
	# M_hat = U.T @ V
	#
	# print("predict, real")
	# for i, j, x_true in Y:
	# 	print(f"{M_hat[i, j]:.2f}, {x_true}")
