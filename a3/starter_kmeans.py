import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

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

    X = reshape(X, (-1, 1, D))
    
    diff = X - MU
    sq_sum = reduce_sum(diff ** 2, axis = 2)
    
#    assert sq_sum.shape == (N, K)
    
    return sq_sum
    
    
if __name__ == '__main__':
    for is_valid in (False, True):
        # Loading data
        data = np.load('data2D.npy')
        
        #data = np.load('data100D.npy')
        [num_pts, dim] = np.shape(data)
        
        # For Validation set
        if is_valid:
            valid_batch = int(num_pts / 3.0)
            np.random.seed(45689)
            rnd_idx = np.arange(num_pts)
            np.random.shuffle(rnd_idx)
            val_data = data[rnd_idx[:valid_batch]]
            data = data[rnd_idx[valid_batch:]]
            
    # %%
        loss_dict = {}
        pred_dict = {}
        valid_loss_dict = {}
        valid_pred_dict = {}
        for K in range(1, 6):
            print('Training K = %d' % K)
            D = data.shape[1]
            N = data.shape[0]
            
            tf.reset_default_graph() 
            tf.set_random_seed(0)
            
            X = tf.placeholder(dtype=tf.float32, shape=(None, D), name='X')
            
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
            valid_loss_curve = []
            valid_pred_list = []
            with tf.Session() as sess:
                # Run the init operation.
                sess.run(init_op)
                
                for i in range(epoches):
                    _, loss_val, pred_val = sess.run([train_op, loss, pred], feed_dict={X: data})
                    loss_curve.append(loss_val)
                    pred_list.append(pred_val)
                    
                    if is_valid:
                        valid_loss_val, valid_pred_val = sess.run([loss, pred], feed_dict={X: val_data})
                    
                        valid_loss_curve.append(valid_loss_val)
                        valid_pred_list.append(valid_pred_val)
                    
            loss_dict[K] = loss_curve
            pred_dict[K] = pred_list
            valid_loss_dict[K] = valid_loss_curve
            valid_pred_dict[K] = valid_pred_list
        
    #%%    
        if not is_valid:
            fig, ax = plt.subplots(2, 5, figsize=(20, 8), sharey='row')
        else:
            fig, ax = plt.subplots(4, 5, figsize=(20, 20), sharey='row')
            
        for K in range(1, 6):
            loss_curve = loss_dict[K]
            pred_list = pred_dict[K]
            pred_val = pred_list[-1]
            
            ax[0, K-1].plot(loss_curve)
            ax[0, K-1].set_xlabel('Number of updates')
            ax[0, K-1].set_ylabel('Training Loss')
            ax[0, K-1].grid()
            ax[0, K-1].set_title('K = %d\nfinal loss = %.2g' % (K, loss_curve[-1]))
            
            for k in range(K):
                ax[1, K-1].plot(data[pred_val == k, 0], data[pred_val == k, 1], '.', alpha=0.3, markeredgewidth=0.0)
                print('Cluster %d' % k, np.count_nonzero(pred_val == k), np.count_nonzero(pred_val == k) * 1.0 / len(pred_val))
            print()
            ax[1, K-1].grid()
            ax[1, K-1].set_xlabel('dim 0')
            ax[1, K-1].set_ylabel('dim 1')
            ax[1, K-1].axis('equal')
            ax[1, K-1].legend([str(k) for k in range(K)])
        
        if not is_valid:
            fig.savefig('1_k%d.png' % K, dpi=150, transparent=True, bbox_inches='tight', pad_inches=0.1)
            fig.show()
        else:
            for K in range(1, 6):
                loss_curve = valid_loss_dict[K]
                pred_list = valid_pred_dict[K]
                pred_val = pred_list[-1]
                
                ax[2, K-1].plot(loss_curve)
                ax[2, K-1].set_xlabel('Number of updates')
                ax[2, K-1].set_ylabel('Validation Loss')
                ax[2, K-1].grid()
                ax[2, K-1].set_title('final loss = %.2g' % loss_curve[-1])
                
                for k in range(K):
                    ax[3, K-1].plot(val_data[pred_val == k, 0], val_data[pred_val == k, 1], '.', alpha=0.3, markeredgewidth=0.0)
                ax[3, K-1].grid()
                ax[3, K-1].set_xlabel('dim 0')
                ax[3, K-1].set_ylabel('dim 1')
                ax[3, K-1].axis('equal')
                ax[3, K-1].legend([str(k) for k in range(K)])
            
            fig.savefig('1_k%d_valid.png' % K, dpi=150, transparent=True, bbox_inches='tight', pad_inches=0.1)
            fig.show()
