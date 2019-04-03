import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp


# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO
    
    pass

def log_GaussPDF(X, MU, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1
    # log_pi: K X 1

    # Outputs:
    # log Gaussian PDF N X K

    if isinstance(X, np.ndarray):
        reduce_sum = np.sum
        reshape = np.reshape
        transpose = np.transpose
        log = np.log
    elif isinstance(X, tf.Tensor):
        reduce_sum = tf.reduce_sum
        reshape = tf.reshape
        transpose = tf.transpose
        log = tf.log
    
    N = X.shape[0]
    K = MU.shape[0]
    D = X.shape[1]
    
    if isinstance(D, tf.Dimension):
        D = D.value
    
#    assert D == 2

    X_ = reshape(X, (-1, 1, D))
    
    diff = X_ - MU
    diff2 = reduce_sum(diff ** 2, -1)
    
#    assert diff2.shape == (N, K)
    
    logN = -0.5 * D * log(2 * np.pi)
    logN = logN + -0.5 * D * log(transpose(sigma))
    logN = logN + -0.5 * (diff2 / transpose(sigma))
    
    return logN    

def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K
    
    if isinstance(log_PDF, np.ndarray):
        from scipy.special import logsumexp
        reduce_logsumexp = lambda a, reduction_indices=1, keep_dims=False: logsumexp(a, axis=reduction_indices, keepdims=keep_dims)
        transpose = np.transpose
    elif isinstance(log_PDF, tf.Tensor):
        reduce_logsumexp = hlp.reduce_logsumexp
        transpose = tf.transpose
    
    N, K = log_PDF.shape
    assert log_pi.shape == (K, 1)
    
    log_post = log_PDF + transpose(log_pi)
    
#    assert log_post.shape == (N, K)
    
    log_sum_exp_log_post = reduce_logsumexp(log_post, reduction_indices=1, keep_dims=True)
    
#    assert log_sum_exp_log_post.shape == (N, 1)
    
    log_post = log_post - log_sum_exp_log_post
    
#    assert log_post.shape == (N, K)
    
    return log_post


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
            tf.set_random_seed(1)
            
            X = tf.placeholder(dtype=tf.float32, shape=(None, D), name='X')
            
            MU = tf.get_variable(name='MU', shape=(K, D), dtype=tf.float32, 
                                 initializer=tf.initializers.random_normal,
                                 trainable=True)
            
            log_sigma = tf.get_variable(name='log_sigma', shape=(K, 1), dtype=tf.float32, 
                                        initializer=tf.initializers.zeros,
                                        trainable=True)
            sigma = tf.exp(log_sigma, name='sigma')
            
            phi = tf.ones((K, 1), dtype=tf.float32)
            log_pi = hlp.logsoftmax(phi)
            
            log_PDF = log_GaussPDF(X, MU, sigma)
            
            log_post = log_posterior(log_PDF, log_pi)
            
            logp = log_PDF + log_post
            logp = hlp.reduce_logsumexp(logp, reduction_indices=1, keep_dims=False)
            logp = tf.reduce_sum(logp, axis=0)
            
            assert logp.shape == ()
            pred = tf.argmax(log_PDF, axis=1)
            
            loss = -logp            
            assert loss.shape == ()
            
            optim = tf.train.AdamOptimizer(learning_rate=1e-2,
                                            beta1=0.9,
                                            beta2=0.99,
                                            epsilon=1e-5)
            train_op = optim.minimize(loss)
            
            # Add an op to initialize the variables.
            init_op = tf.global_variables_initializer()
            epoches = 1000
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
            fig.savefig('2_k%d.png' % K, dpi=150, transparent=True, bbox_inches='tight', pad_inches=0.1)
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
            
            fig.savefig('2_k%d_valid.png' % K, dpi=150, transparent=True, bbox_inches='tight', pad_inches=0.1)
            fig.show()
