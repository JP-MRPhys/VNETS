import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple, LSTMCell
from tensorflow_probability.python.distributions import Normal


class BayesianLSTMCell(LSTMCell):

    def __init__(self, num_units, prior, is_training, name, **kwargs):

        super(BayesianLSTMCell, self).__init__(num_units, **kwargs)

        self.w = None
        self.b = None
        self.prior = prior
        self.layer_name = name
        self.isTraining = is_training
        self.num_units = num_units
        self.kl_loss=None

        print("Creating lstm layer:" + name)


    def call(self, inputs, state):

        print("Calling Baysein LSTM" + self.layer_name)

        if self.w is None:

            print("Creating LSTM weights")
            size = inputs.get_shape()[-1].value
            self.w, self.w_mean, self.w_sd = self.variational_posterior((size+self.num_units, 4*self.num_units), self.layer_name+'_weights', self.isTraining)
            self.b, self.b_mean, self.b_sd = self.variational_posterior((4*self.num_units,1), self.layer_name+'_bias', self.isTraining)

        self.theta_w=(self.w_mean, self.w_sd)
        self.theta_b=(self.b_mean, self.b_sd)

        if(self.isTraining):
        #    with tf.variable_scope("KL_loss_" + self.layer_name, reuse=True):
                self.kl_loss = self.compute_KL_univariate_prior(self.prior, self.theta_w, self.w)
                self.kl_loss += self.compute_KL_univariate_prior(self.prior, self.theta_b, self.b)
                #self.kl_loss=kl_loss
                #print("Compute KL loss for LSTM:  " + self.layer_name)
                #print(kl_loss)


        cell, hidden = state
        concat_inputs_hidden = tf.concat([inputs, hidden], 1)
        concat_inputs_hidden = tf.nn.bias_add(tf.matmul(concat_inputs_hidden, self.w), tf.squeeze(self.b))
        # Gates: Input, New, Forget and Output
        i, j, f, o = tf.split(value=concat_inputs_hidden, num_or_size_splits=4, axis=1)
        new_cell = (cell * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * self._activation(j))
        new_hidden = self._activation(new_cell) * tf.sigmoid(o)
        new_state = LSTMStateTuple(new_cell, new_hidden)

        return new_hidden, new_state

    #not sure if need to
    def get_kl(self):
        """
            :returns: the KL loss for the for this lstms weights and bias

        """

        return self.kl_loss

    def compute_KL(self):

        """
            :return:
        """

        kl_loss=self.compute_KL_univariate_prior(self.prior, self.theta_b,self.w)
        kl_loss += self.compute_KL_univariate_prior(self.prior, self.theta_b, self.b)

        return kl_loss


    def variational_posterior(self,shape, name, isTraining):

        """
        this function creates a variational posterior q(w/theta) over a given "weight:w" of the network

        theta is parameterized by mean+standard*noise we apply the re-parameterization trick from kingma et al, 2014
        with correct loss function (free energy) we learn mean and standard to estimate of theta, thus can estimate posterior p(w/D)
        by computing KL loss for each variational posterior q (w/theta) with prior(w)

        :param name: is the name of the tensor/variable to create variational posterior  q(w/Q) for true posterior (p(w/D))
        :param shape: is the shape of the weigth variable
        :param training: whether in training or inference mode
        :return: samples (i.e. weights), mean of weigths, std in-case of the training there is noise assoicated with the weights

        """

        # variations
        # theta=mu+sigma i.e. theta = mu+sigma i.e. mu+log(1+exp(rho)), log(1+exp(rho))
        # is the computed by using tf.math.softplus(rho) to avoid negative sigma
        # need to check for init

        mu = tf.get_variable("{}_mean".format(name), shape=shape, dtype=tf.float32);
        rho = tf.get_variable("{}_rho".format(name), shape=shape, dtype=tf.float32);
        sigma = tf.math.softplus(rho)

        # if training we add noise to variation parameters theta
        if (isTraining):
            epsilon = Normal(0, 1.0).sample(shape)
            sample = mu + sigma * epsilon
        else:
            sample = mu + sigma

        return sample, mu, sigma

    def compute_KL_univariate_prior(self, univariateprior, theta, sample):

        """
        :param prior:  assuming univariate prior of Normal(m,s);
        :param posterior: (theta: mean,std) to create posterior q(w/theta) i.e. Normal(mean,std)
        :param sample:
        :return: KL (analytical)

        """

        sample=tf.reshape(sample, [-1])  #flatten vector
        (mean,std)=theta
        mean =tf.reshape(mean, [-1])
        std=tf.reshape(std, [-1])
        posterior = Normal(mean, std)
        (mean2,std2) = univariateprior
        prior=Normal(mean2, std2)
        q_theta=tf.reduce_sum(posterior.log_prob(sample))
        p_d=tf.reduce_sum(prior.log_prob(sample))
        KL=tf.subtract(q_theta,p_d)

        print("computed KL loss" + self.layer_name)

        return KL



