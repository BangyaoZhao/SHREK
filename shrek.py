import numpy as np
######### tfp
import tensorflow as tf
# tf.enable_v2_behavior()
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from scipy.special import log_softmax, softmax

class SHREK:
    def __init__(self, shrinkage_factor=0, dtype = tf.float32) -> None:
        self.shrinkage_factor, self.dtype = shrinkage_factor, dtype
    
    def fit(self, X: np.ndarray, y: np.ndarray, num_steps=10000, sample_size=10, learning_rate=0.001):
        self.X_train = tf.constant(X, dtype=self.dtype)
        self.y_train = tf.constant(y, dtype=self.dtype)
        self.n, self.K, self.n_channel, self.n_Tz = self.X_train.shape
        self.n_corr = round(self.n_channel*(self.n_channel-1)/2)
        def jointmodel():
            sigma = yield tfd.Sample(tfd.HalfCauchy(0.0, tf.cast(1, self.dtype)), self.n_channel)
            # covmat
            covmat_chol = yield tfd.LKJ(dimension=self.n_channel, concentration=2, input_output_cholesky=True)
            # weights
            beta = yield tfb.Cumsum()(tfd.Sample(tfd.Normal(0.0, tf.cast(1, self.dtype)), (self.n_channel, self.n_Tz)))
            beta = sigma[:, None] * tf.linalg.matmul(covmat_chol, beta)
            beta = tfp.math.soft_threshold(beta, self.shrinkage_factor)
            logits = tf.linalg.tensordot(self.X_train, beta, axes = [[2, 3], [0, 1]])
            y = yield tfd.Multinomial(logits = logits, total_count = 1)
        self.joint = tfd.JointDistributionCoroutineAutoBatched(jointmodel)

        self.posterior = tfd.JointDistributionSequentialAutoBatched([
            tfd.LogNormal(
                tf.Variable(tf.zeros(self.n_channel, dtype = self.dtype) - 3),
                tfp.util.TransformedVariable(0.1 * tf.ones(self.n_channel, dtype = self.dtype), bijector = tfb.Softplus())),
            tfb.CorrelationCholesky()(tfd.Independent(tfd.Normal(
                tf.Variable(0.1*tf.random.normal((self.n_corr, ), dtype = self.dtype), 
                            dtype = self.dtype),
                tfp.util.TransformedVariable(0.1*tf.ones(self.n_corr, dtype = self.dtype), 
                                             bijector = tfb.Softplus())), 1)),
            tfd.Normal(
                tf.Variable(tf.random.normal((self.n_channel, self.n_Tz), stddev=3*self.shrinkage_factor, dtype = self.dtype), 
                            dtype = self.dtype),
                tfp.util.TransformedVariable(0.01 * tf.ones((self.n_channel, self.n_Tz), dtype = self.dtype), 
                                             bijector = tfb.Softplus())),
        ])

        self.losses = []
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.losses += list(tfp.vi.fit_surrogate_posterior(
            self.loglik, 
            self.posterior,
            optimizer = optimizer,
            num_steps = num_steps, 
            sample_size = sample_size))
        
        self.losses = [float(x) for x in self.losses]
        self.samples = [np.array(x) for x in self.posterior.sample(10000)]
        self.betaMats = self.get_betaMats()
        self.betaMat = np.median(self.betaMats, axis=0)
    
    def loglik(self, *args):
        return self.joint.log_prob(*args, self.y_train)
        
    def get_betaMats(self):
        sigmas, corr_trils, beta_raws = self.samples
        betaMats = sigmas[:, :, None] * tf.linalg.matmul(corr_trils, beta_raws)
        return np.array(tfp.math.soft_threshold(betaMats, self.shrinkage_factor))
    
    # @property
    # def DIC(self):
    #     l = self.loglik(*self.posterior.mean()).numpy()
    #     samples = self.posterior.sample(10000)
    #     ls = self.loglik(*samples).numpy()
    #     return -2 * (2*ls.mean() - l)
    
    @property
    def corr(self):
        corr_trils = self.samples[1]
        corr_mats = np.array([corr_tril@corr_tril.T for corr_tril in corr_trils])
        return np.quantile(corr_mats, [0.05, 0.5, 0.95], axis=0)
    
    def predict_logprob(self, newX: np.ndarray):
        return np.tensordot(newX, self.betaMat, axes=[[2,3], [0,1]])