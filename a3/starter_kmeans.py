import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
data = np.load('data2D.npy')

#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)

suffix = '2'
is_valid = suffix == '3'
# For Validation set
if is_valid:
    valid_batch = int(num_pts / 3.0)
    np.random.seed(45689)
    rnd_idx = np.arange(num_pts)
    np.random.shuffle(rnd_idx)
    val_data = data[rnd_idx[:valid_batch]]
    data = data[rnd_idx[valid_batch:]]

# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    
    if isinstance(X, np.ndarray):
        reduce_sum = np.sum
        reshape = np.reshape
    elif isinstance(X, tf.Tensor):
        reduce_sum = tf.reduce_sum
        reshape = tf.reshape
    
    N = X.shape[0]
    K = MU.shape[0]
    D = X.shape[1]

    X = reshape(X, (N,1,D))
    
    diff = X - MU
    sq_sum = reduce_sum(diff ** 2, axis = 2)
    
    assert sq_sum.shape == (N, K)
    
    return sq_sum
    
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
    loss_dict = {}
    pred_dict = {}
    for K in range(1, 6):
        D = data.shape[1]
        N = data.shape[0]
        
        tf.reset_default_graph()        
        X = tf.placeholder(dtype=tf.float32, shape=(N, D), name='X')
        
        MU = tf.get_variable(name='MU', shape=(K, D), dtype=tf.float32, 
                             initializer=tf.initializers.random_normal,
                             trainable=True)
        
        sq_dist = distanceFunc(X, MU)
        
        pred = tf.argmin(sq_dist, axis=1)
        min_sq_dist = tf.reduce_min(sq_dist, axis=1)
        
        loss = tf.reduce_mean(min_sq_dist, axis=0)
        
        assert loss.shape == ()
        
        optim = tf.train.AdamOptimizer(learning_rate=1e-2,
                                        beta1=0.9,
                                        beta2=0.99,
                                        epsilon=1e-5)
        train_op = optim.minimize(loss)
        
        # Add an op to initialize the variables.
        init_op = tf.global_variables_initializer()
        epoches = 500
        loss_curve = []
        pred_list = []
        with tf.Session() as sess:
            # Run the init operation.
            sess.run(init_op)
            
            for i in range(epoches):
                _, loss_val, pred_val = sess.run([train_op, loss, pred], feed_dict={X: data})
                loss_curve.append(loss_val)
                pred_list.append(pred_val)
    #            if len(pred_list) >= 2 and np.all(pred_list[-1] == pred_list[-2]):
    #                print('Converged after %d iterations' % i)
    #                break
        loss_dict[K] = loss_curve
        pred_dict[K] = pred_list
    
#%%    
    fig, ax = plt.subplots(2, 5, figsize=(10, 4), sharey='row')
    for K in range(1, 6):
        loss_curve = loss_dict[K]
        pred_list = pred_dict[K]
        pred_val = pred_list[-1]
        
        ax[0, K-1].plot(loss_curve)
        ax[0, K-1].set_xlabel('Number of updates')
        ax[0, K-1].set_ylabel('Loss')
        ax[0, K-1].grid()
        ax[0, K-1].set_title('K = %d' % K)
        
        for k in range(K):
            ax[1, K-1].plot(data[pred_val == k, 0], data[pred_val == k, 1], '.')
        ax[1, K-1].grid()
        ax[1, K-1].set_xlabel('dim 0')
        ax[1, K-1].set_ylabel('dim 1')
        
    fig.savefig('1_k%d_%s.png' % (K, suffix), dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.1)
    fig.show()
        
        
        
        
        
        
        
        
        
        
        
        
        