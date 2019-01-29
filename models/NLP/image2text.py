"""
Simple Implementation of the show tell model for image to text...
"""


import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

glove_embeddings = np.random.rand(20000, 300)


class image2text():

    # 2. Construct the decoder cell
    def create_cell(self, rnn_size, keep_prob):
        # lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1,0.1,seed=2))

        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, state_is_tuple=True)
        drop = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        return drop

    def __init__(self):
        self.batch_size = 512
        self.max_seq_length = 100
        self.image_embedding_dim = 300
        self.lstm_dim = 128
        self.keep_prob = 0.5
        self.decoder_vocab_size = 20000

        self.image_features = tf.placeholder(tf.float32, shape=[self.batch_size, self.image_embedding_dim])
        self.caption_input = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_seq_length])

        with tf.variable_scope('embeddings'):
            # embedding for the input sentense
            self.decoder_embeddings = tf.get_variable(name="decoder_embedding", shape=np.shape(glove_embeddings),
                                                      initializer=tf.constant_initializer(glove_embeddings),
                                                      trainable=False)

            self.decoder_input_embeddings = tf.nn.embedding_lookup(self.decoder_embeddings, self.caption_input)
            print(self.decoder_input_embeddings)

        with tf.variable_scope('decoder') as decoder_scope:
            self.dec_cell = self.create_cell(self.lstm_dim, self.keep_prob)
            self.zero_state = self.dec_cell.zero_state(self.batch_size, tf.float32)

            _, self.initial_state = self.dec_cell(self.image_features, self.zero_state)

            decoder_scope.reuse_variables()

            self.lstm_outputs, _ = tf.nn.dynamic_rnn(cell=self.dec_cell, inputs=self.decoder_input_embeddings,
                                                     initial_state=self.initial_state, dtype=tf.float32,
                                                     scope=decoder_scope)
            self.lstm_outputs = tf.reshape(self.lstm_outputs, [-1, self.dec_cell.output_size])

        with tf.variable_scope('logits'):
            self.logits = fully_connected(self.lstm_outputs, num_outputs=self.decoder_vocab_size)

        # need to figure this out difference between target captions and input cations ....
        self.target_seq = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_seq_length])
        self.target_seq = tf.reshape(self.target_seq, [-1])

        # need to sort out this see the comment in doc
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.target_seq, logits=self.logits)
        self.optimizer = tf.train.AdamOptimizer(0.02).minimize(self.loss)  # no gradient clipping required

        """
        #to assess why logits are not able to get the proper shap in training variables, remove the training variables which don't work.. 
        print("Optimize")
        self.params = tf.trainable_variables()
        self.gradients = tf.gradients(self.loss, self.params)
        self.clipped_gradients = tf.clip_by_global_norm(self.gradients, clip_norm=5.0)  # how select this value
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_operation = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.params))
        """

if __name__ == '__main__':
    model = image2text()
