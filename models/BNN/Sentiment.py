#sentiment analysis using multilayer RNN (LSTM) twitter

import tensorflow as tf
from models.NLP.utilities.StockTwits import get_twets_data, get_training_batch_twets
from models.NLP.utilities.imdbDatareader import *
from models.BNN.bayesianLSTM import BayesianLSTMCell
from models.BNN import VariationalPosterior
from tensorflow_probability.python.distributions import Normal
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SentimentAnalysisMultiLayerLSTM:

    def __init__(self, training):
        self.LSTM_KL=0
        self.embedding_dim = 300  # the number of hidden units in each RNN
        self.keep_prob = 0.5
        self.batch_size = 512
        self.lstm_sizes = [128, 64]  # number hidden layer in each LSTM
        self.num_classes = 2
        self.max_sequence_length = 750
        self.prior=(0,1) #univariator prior
        self.isTraining=training


        with tf.variable_scope('rnn_i/o'):
            # use None for batch size and dynamic sequence length
            self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.embedding_dim])
            self.groundtruths = tf.placeholder(tf.float32, shape=[None, self.num_classes])

        with tf.variable_scope('rnn_cell'):
            self.initial_state, self.lstm_outputs_final, self.final_state, self.cell = self.build_lstm_layers_manual(self.lstm_sizes,
                                                                                                        self.inputs,
                                                                                                        self.keep_prob,
                                                                                                        self.batch_size)

            #self.outputs2 = self.lstm_outputs[:, -1]  # take the last output of the LSTM

            self.outputs2=self.lstm_outputs_final

            print("LAST OUTPUT")

            print(self.outputs2)

            self.softmax_w, self.softmax_w_mean, self.softmax_w_std=  self.variational_posterior((self.lstm_sizes[-1], self.num_classes), "softmax_w", self.isTraining)
            self.softmax_b, self.softmax_b_mean, self.softmax_b_std = self.variational_posterior((self.num_classes), "softmax_b", self.isTraining)

            self.logits=tf.nn.xw_plus_b(self.outputs2,  self.softmax_w,self.softmax_b)

            print("Output's :")
            print(self.outputs2)
            print("Logit's :")
            print(self.logits)


        with tf.variable_scope('rnn_loss', reuse=tf.AUTO_REUSE):
            # use cross_entropy as class loss
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.groundtruths, logits=self.logits)

            if (self.isTraining):

                self.KL  =  self.compute_KL_univariate_prior(self.prior,(self.softmax_w_mean, self.softmax_w_std), self.softmax_w)
                self.KL  += self.compute_KL_univariate_prior(self.prior,(self.softmax_b_mean, self.softmax_b_std), self.softmax_b)
                for i in range(len(self.lstm_sizes)):
                  self.LSTM_KL=self.lstms[i].get_kl()
                  print(self.LSTM_KL)
                  self.KL +=self.LSTM_KL

            self.cost=(self.loss+self.KL)/self.batch_size  #the total cost is the
            self.optimizer = tf.train.AdamOptimizer(0.02).minimize(self.loss)

        #with tf.variable_scope('rnn_accuracy'):
            # self.accuracy = tf.contrib.metrics.accuracy(labels=tf.argmax(self.groundtruths, axis=1), predictions=self.prediction)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())  # don't forget to initial all variables
        self.saver = tf.train.Saver()  # a saver is for saving or restoring your trained weight

        print("Completed creating the graph")

    def train(self, batch_x, batch_y, state):

        fd = {}
        fd[self.inputs] = batch_x
        fd[self.groundtruths] = batch_y
        fd[self.initial_state] = state
        # feed in input and groundtruth to get loss and update the weight via Adam optimizer
        loss, accuracy, final_state, _ = self.sess.run([self.loss, self.accuracy, self.final_state, self.optimizer], fd)

        return loss, accuracy, final_state

    def test(self, batch_x, batch_y, batch_size):
        """
         NEED TO RE-WRITE this function interface by adding the state
        :param batch_x:
        :param batch_y:
        :return

        """
        # restore the model

        # with tf.Session() as sess:
        #    model=model.restore();

        test_state = model.cell.zero_state(batch_size, tf.float32)

        fd = {}
        fd[self.inputs] = batch_x
        fd[self.groundtruths] = batch_y
        fd[self.initial_state] = test_state
        prediction, accuracy = self.sess.run([self.prediction, self.accuracy], fd)

        return prediction, accuracy

    def save(self, e):
        self.saver.save(self.sess, 'model/rnn/rnn_%d.ckpt' % (e + 1))

    def restore(self, e):
        self.saver.restore(self.sess, 'model/rnn/rnn_%d.ckpt' % (e))

    def build_lstm_layers(self, lstm_sizes, embed_input, keep_prob_, batch_size):
        """
        Create the LSTM layers
        inputs: array containing size of hidden layer for each lstm,
                input_embedding, for the shape batch_size, sequence_length, emddeding dimension [None, None, 384], None and None are to handle variable batch size and variable sequence length
                keep_prob for the dropout and batch_size

        outputs: initial state for the RNN (lstm) : tuple of [(batch_size, hidden_layer_1), (batch_size, hidden_layer_2)] .. only two here e.g. [(256,128), (256,64)]
                 outputs of the RNN [Batch_size, sequence_length, last_hidden_layer_dim]
                 RNN cell: tensorflow implementation of the RNN cell
                 final state: tuple of [(batch_size, hidden_layer_1), (batch_size, hidden_layer_2)]

        """
        self.lstms=[]
        for i in range (0,len(lstm_sizes)):
            self.lstms.append(BayesianLSTMCell(lstm_sizes[i], self.prior, self.isTraining, 'lstm'+str(i)))

        # Stack up multiple LSTM layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell(self.lstms)
        # Getting an initial state of all zeros
        initial_state = cell.zero_state(batch_size, tf.float32)
        # perform dynamic unrolling of the network, for variable
        lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, embed_input, initial_state=initial_state)

        print("....................LSTM outputs................")
        print(lstm_outputs)

        return initial_state, lstm_outputs, final_state, cell


    def build_lstm_layers_manual(self, lstm_sizes, inputs, keep_prob_, batch_size):
        """
        Create the LSTM layers
        inputs: array containing size of hidden layer for each lstm,
                input_embedding, for the shape batch_size, sequence_length, emddeding dimension [None, None, 384], None and None are to handle variable batch size and variable sequence length
                keep_prob for the dropout and batch_size

        outputs: initial state for the RNN (lstm) : tuple of [(batch_size, hidden_layer_1), (batch_size, hidden_layer_2)] .. only two here e.g. [(256,128), (256,64)]
                 outputs of the RNN [Batch_size, sequence_length, last_hidden_layer_dim]
                 RNN cell: tensorflow implementation of the RNN cell
                 final state: tuple of [(batch_size, hidden_layer_1), (batch_size, hidden_layer_2)]

        """
        self.lstms=[]
        for i in range (0,len(lstm_sizes)):
            self.lstms.append(BayesianLSTMCell(lstm_sizes[i], self.prior, self.isTraining, 'lstm'+str(i)))

        # Stack up multiple LSTM layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell(self.lstms)
        # Getting an initial state of all zeros
        initial_state = cell.zero_state(batch_size, tf.float32)
        # perform dynamic unrolling of the network, for variable
        #lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, embed_input, initial_state=initial_state)

        # we avoid dynamic RNN, as this produces while loop errors


        if True:
            outputs = []
            state = initial_state
            with tf.variable_scope("RNN"):
                for time_step in range(self.max_sequence_length):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = cell(inputs[:, time_step, :], state)
                    outputs.append(cell_output)

        final_lstm_outputs = cell_output
        final_state = state
        #outputs=tf.reshape(tf.concat(1, outputs), [-1, self.embedding_dim])

        return initial_state, final_lstm_outputs, final_state, cell



    def compute_KL_univariate_prior(self, univariateprior, theta, sample):

        """
        :param prior:  assuming univariate prior of Normal(m,s);
        :param posterior: (theta: mean,std) to create posterior q(w/theta) i.e. Normal(mean,std)
        :param sample:
        :return:

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

        return KL

    def variational_posterior(self,shape, name, isTraining):

        """

        this function create a variational posterior q(w/theta) over a given "weight:w" of the network

        theta is parameterized by mean+standard*noise we apply the reparameterization trick from kingma et al, 2014
        with correct loss function (free energy) we learn mean and standard to estimate of theta, thus can estimate posterior p(w/D)
        by computing KL loss for each variational posterior q(w/theta) with prior(w)

        :param name: is the name of the tensor/variable to create variational posterior  q(w/Q) for true posterior (p(w/D))
        :param shape: is the shape of the weigth variable
        :param training: whether in training or inference mode
        :return: samples (i.e. weights), mean of weigths, std in-case of the training there is noise associated with the weights

        """

        # variations
        # theta=mu+sigma i.e. theta = mu+sigma i.e. mu+log(1+exp(rho)), log(1+exp(rho)) is the computed by using tf.math.softplus(rho) to avoid negative sigma

        # need to check for init

        mu = tf.get_variable("{}_mean".format(name), shape=shape, dtype=tf.float32);
        rho = tf.get_variable("{}_rho".format(name), shape=shape, dtype=tf.float32);
        sigma = tf.math.softplus(rho)

        # if training we sample from noise to variation parameters theta
        if (isTraining):
            epsilon = Normal(0, 1.0).sample(shape)
            sample = mu + sigma * epsilon
        else:
            sample = mu + sigma;

        return sample, mu, sigma   #here sample represent distribution of weights, along with mean (mu) and std (sigma)


if __name__ == '__main__':

    # hyperparameter of our network
    EPOCHS = 20
    tf.reset_default_graph()
    model = SentimentAnalysisMultiLayerLSTM(training=True)


    """

    train_data = get_twets_data()
    n_train = len(train_data)

    BATCH_SIZE = model.batch_size
    print("BATCH SIZE : " + str(BATCH_SIZE))

    rec_loss = []

    for epoch in range(EPOCHS):

        state = model.sess.run([model.initial_state])
        train_data = train_data.sample(frac=1).reset_index(drop=True)
        loss_train = 0
        accuracy_train = 0

        for idx in range(0, n_train, BATCH_SIZE):
            BATCH_X, BATCH_Y = get_training_batch_twets(train_data[idx:(idx + BATCH_SIZE)], BATCH_SIZE,
                                                        model.embedding_dim, num_classes=model.num_classes,
                                                        maxlen=model.max_sequence_length)
            loss_batch, accuracy_batch, state = model.train(BATCH_X, BATCH_Y, state)
            loss_train += loss_batch
            accuracy_train += accuracy_batch
            print("EPOCH: " + str(epoch) + "BATCH_INDEX:" + str(idx) + "Batch Loss:" + str(
                loss_batch) + "Batch Accuracy:" + str(accuracy_train))

        loss_train /= n_train
        accuracy_train /= n_train

        model.save(epoch)  # save your model after each epoch
        rec_loss.append([loss_train, accuracy_train])

    np.save('./model/rnn/rec_loss.npy', rec_loss)
    print("Training completed")
    """