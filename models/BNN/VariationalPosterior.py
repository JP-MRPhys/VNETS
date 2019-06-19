import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import Normal

class VariationalPosterior():
    def __init__(self, shape, name, prior, isTraining):


        """

        this function create a variational posterior q(w/theta) over a given "weight:w" of the network

        theta is parameterized by mean+standard*noise we apply the reparameterization trick from kingma et al, 2014
        with correct loss function (free energy) we learn mean and standard to estimate of theta, thus can estimate posterior p(w/D)
        by computing KL loss for each variational posterior q(w/theta) with prior(w)

        :param name: is the name of the tensor/variable to create variational posterior  q(w/Q) for true posterior (p(w/D))
        :param shape: is the shape of the weigth variable
        :param prior: is the Prior distribution i.e. Normal(mean,std) or MutlivariateNormal (class see above for an implementation)
        :param istraining: whether in training or inference mode
        :return: samples (i.e. weights), mean of weights, std in-case of the training as there is noise associated with the weights
        """

        # variations
        # theta=mu+sigma i.e. theta = mu+sigma i.e. mu+log(1+exp(rho)), log(1+exp(rho)) is the computed by using tf.math.softplus(rho) to avoid negative sigma
        # need to check for init

        self.shape=shape
        self.prior=prior   #
        self.name = name
        self.mu =  tf.get_variable("{}_mean".format(self.name), shape=shape, dtype=tf.float32);
        self.rho = tf.get_variable("{}_rho".format(self.name), shape=shape, dtype=tf.float32);
        self.sigma = tf.math.softplus(self.rho)

        self.isTraining=isTraining
        self.KL=0
        print(self.mu)
        print(self.rho)
        print("Completed created posterior as above")

    def __call__(self):

        print("Calling posterior variable" + self.name)

        # if training we add noise to variation parameters theta
        if (self.isTraining):
            epsilon = Normal(0, 1.0).sample(self.shape)
            self.samples = self.mu + self.sigma * epsilon
            self.KL=self.compute_KL_univariate_prior(self.prior,self.samples)
            print("Training variational posterior" + self.name + "KL loss" + str(self.Kl)) #debug only
            
        else:
            self.samples = self.mu + self.sigma;
            #return sample, self.mu, self.sigma
            #return sample, self.mu, self.sigma


    def  get_kl(self):
        return self.KL

    def get_samples(self):
        """ :return: get to obtain mean or all weights which have been sample based on training or inference mode
        """

        return


    def compute_KL_univariate_prior(self,univariateprior, samples):
        """
        :param prior:  assuming univatier prior of Normal(m,s); i.e. Normal(s,
        :param posterior: (theta: mean,std) to create posterior q(w/theta) i.e. Normal(mean,std)
        :param sample:
        :return:

        """

        samples = tf.reshape(samples, [-1])  # flatten vector
        (mean2, std2) = univariateprior
        prior = Normal(mean2, std2)
        posterior = Normal(self.mu, self.sigma)
        q_theta = tf.reduce_sum(posterior.log_prob(samples))
        p_d = tf.reduce_sum(prior.log_prob(self.samples))
        KL = tf.subtract(q_theta, p_d)

        return KL


