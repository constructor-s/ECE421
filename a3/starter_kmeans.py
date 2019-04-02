import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
from scipy.stats import norm

# Loading data
data = np.load('data2D.npy')

#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)

##ploting data
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
    
    with tf.variable_scope('distanceFunc'):
        x_sq = tf.reduce_sum(tf.square(X), 1)
        mu_sq = tf.reduce_sum(tf.square(MU), 1)
        
        print(x_sq)
        
        x_sq = tf.reshape(x_sq, [-1,1])
        mu_sq = tf.reshape(mu_sq, [1,-1])
        
        distance = tf.sqrt(tf.maximum(x_sq - 2 * tf.matmul(X, MU, False, True) + mu_sq, 0.0))
        
        print(distance)
    return distance
    
    
#    N = X.shape[0]
#    K = MU.shape[0]
#    D = X.shape[1]
#
#    X = X.reshape(N,1,D)
#    pair_dist = np.zeros((N,K))
#    
#    diff = X - MU
#    sq_sum = np.sum(diff ** 2, axis = 2)
#    
#    pair_dist = np.sqrt(sq_sum)
#    
#    return pair_dist
    
def assign(data, k, learn_rate=0.01):
    
    tf.reset_default_graph()
    tf.set_random_seed(421)
    
    D = data.shape[1]

    
    #initialize mu based on standard normal distribution   
    mu = tf.get_variable(
            dtype=tf.float32,
            initializer=tf.random_normal((k,D), seed=421),
            name='mu'
            )
    
    x = tf.placeholder(
        dtype=tf.float32,
        shape=(D, None),
        name='data'
    )


    distance = distanceFunc(x, mu)
    
    #minimize distance
    op = tf.train.AdamOptimizer(learn_rate, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(distance)
    
    return op
    
def update(data, k, learn_rate=0.01):
    op = assign(data, k, learn_rate)
    #op = tf.to_int32(op)
    op = tf.cast(op, tf.int32)
    
    #cluster based on nearest distance of each data point
    partitions = tf.dynamic_partition(data, op, k)
    
    #???????
    #TypeError: Can't convert Operation 'Adam' to Tensor (target dtype=None, name='x', as_ref=False)
    
    #update center points
    for p in partitions:
        center = tf.concat(0, [tf.expand_dims(tf.reduce_mean(p, 0),0)])
    
    return center
    
    
if __name__ == '__main__':
    k = 3
    update(data, k, 0.01)
    
    