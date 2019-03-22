import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
from scipy.stats import norm

# Loading data
data = np.load('data2D.npy')

#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)

#ploting data
#v1 = data[:,0]
#v2 = data[:,1]
#
#x = np.array(list(zip(v1,v2)))
#plt.scatter(v1,v2)


# For Validation set
#if is_valid:
#  valid_batch = int(num_pts / 3.0)
#  np.random.seed(45689)
#  rnd_idx = np.arange(num_pts)
#  np.random.shuffle(rnd_idx)
#  val_data = data[rnd_idx[:valid_batch]]
#  data = data[rnd_idx[valid_batch:]]

# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO
    N = X.shape[0]
    K = MU.shape[0]
    D = X.shape[1]

    X = X.reshape(N,1,D)
    pair_dist = np.zeros((N,K))
    
    diff = X - MU
    sq_sum = np.sum(diff ** 2, axis = 2)
    
    pair_dist = np.sqrt(sq_sum)
    
    return pair_dist
    
def Lu(data, k):
    
    #initialize mu based on standard normal distribution
    D = data.shape[1]
    mu = np.zeros((k,D))

    for i in range (k):
        mu[i,:] = [np.random.randn(), np.random.randn()]
    
    print(mu)
    
    
    
if __name__ == '__main__':
    k = 3
    Lu(data, k)
    
    