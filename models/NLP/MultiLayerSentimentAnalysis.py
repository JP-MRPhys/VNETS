#sentiment analysis using multilayer RNN (LSTM) twitter


import tensorflow as tf
import numpy as np
from models.NLP.utilities.StockTwits import get_twets_data, get_training_batch_twets
from models.NLP.utilities.imdbDatareader import *


class SentimentAnalysisMultiLayerLSTM:

    def __init__(self):
        self.embedding_dim = 300  # the number of hidden units in each RNN
        self.keep_prob = 0.5
        self.batch_size = 512
        self.lstm_sizes = [128, 64]  # number hidden layer in each LSTM
        self.num_classes = 2
        self.max_sequence_length = 750

        with tf.variable_scope('rnn_i/o'):
            # use None for batch size and dynamic sequence length
            self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.embedding_dim])
            self.groundtruths = tf.placeholder(tf.float32, shape=[None, self.num_classes])

        with tf.variable_scope('rnn_cell'):
            self.initial_state, self.lstm_outputs, self.final_state, self.cell = self.build_lstm_layers(self.lstm_sizes,
                                                                                                        self.inputs,
                                                                                                        self.keep_prob,
                                                                                                        self.batch_size)
            self.outputs2 = self.lstm_outputs[:, -1]  # take the last output of the LSTM
            print(self.outputs2)

            self.logits = tf.contrib.layers.fully_connected(self.outputs2, self.num_classes, activation_fn=None)
            self.prediction = tf.argmax((self.logits), axis=1)
            print(self.prediction)

        with tf.variable_scope('rnn_loss'):
            # use cross_entropy as class loss
            self.loss = tf.losses.softmax_cross_entropy(
                onehot_labels=self.groundtruths, logits=self.logits)
            self.optimizer = tf.train.AdamOptimizer(0.02).minimize(self.loss)

        with tf.variable_scope('rnn_accuracy'):
            self.accuracy = tf.contrib.metrics.accuracy(
                labels=tf.argmax(self.groundtruths, axis=1),
                predictions=self.prediction)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())  # don't forget to initial all variables
        self.saver = tf.train.Saver()  # a saver is for saving or restoring your trained weight

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
        lstms = [tf.contrib.rnn.LSTMCell(size) for size in lstm_sizes]

        # Add dropout to the cell
        drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_) for lstm in lstms]

        # Stack up multiple LSTM layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell(drops)

        # Getting an initial state of all zeros
        initial_state = cell.zero_state(batch_size, tf.float32)

        # perform dynamic unrolling of the network, for variable
        lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, embed_input, initial_state=initial_state)

        return initial_state, lstm_outputs, final_state, cell


if __name__ == '__main__':

    # hyperparameter of our network
    EPOCHS = 20

    tf.reset_default_graph()
    model = SentimentAnalysisMultiLayerLSTM()

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
