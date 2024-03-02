from __future__ import print_function
import numpy as np
from time import time # for camparing running time

M = 100
d, N = 1000, 10000 # dimension, number of training points
X = np.random.randn(N, d) # N d-dimensional points
z = np.random.randn(d)
Z = np.random.randn(M, d)

# naively compute square distance between two vectors
def dist_pp(z, x):
    d = z - x.reshape(z.shape) # force x and z to have the same dims
    return np.sum(d*d)

# from one point to each point in a set, naive
def dist_ps_naive(z, X):
    N = X.shape[0]
    res = np.zeros((1, N))
    for i in range (N):
        res[0][i] = dist_pp(z, X[i])
    return res

# from one point to each point in a set, fast
def dist_ps_fast(z, X):
    Xsquare = np.sum(X*X, 1) # square of l2 norm of each row of X
    zsquare = np.sum(z*z) # square of l2 norm of z
    return Xsquare + zsquare - 2*X.dot(z) # zsquare can be ignored

# from each point in 1 set to each point in another set, half fast
def dist_ss_0(Z, X):
    M = Z.shape[0]
    N = X.shape[0]
    res = np.zeros((M, N))
    for i in range(M):
        res[i] = dist_ps_fast(Z[i], X)
    return res

# from from each point in one set to each point in another set, fast
def dist_ss_fast(Z, X):
    Xsquare = np.sum(X*X, 1) # square of l2 norm of each ROW of X
    Zsquare = np.sum(Z*Z, 1) # square of l2 norm of each ROW of Z 
    return Zsquare.reshape(-1, 1) + Xsquare.reshape(1, -1) - 2*Z.dot(X.T)


t1 = time()
D1 = dist_ps_naive(z, X)
print('naive point2set, running time:', time() - t1, 's')

t1 = time()
D2 = dist_ps_fast(z, X)
print('fast point2set , running time:', time() - t1, 's')
print('Result difference:', np.linalg.norm(D1 - D2))

t1 = time()
D3 = dist_ss_0(Z, X)
print('half fast set2set running time:', time() - t1, 's')
t1 = time()
D4 = dist_ss_fast(Z, X)
print('fast set2set running time', time() - t1, 's')
print('Result difference:', np.linalg.norm(D3 - D4))
