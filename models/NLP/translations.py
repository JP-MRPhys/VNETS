import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import *

from models.NLP.utilities.translation_data import load_vec

#seq2seq implementation


# time major: where encoder length comes first before the batch size, this will influence model specific e.g. attention see below for more details


def build_lstm_layers(lstm_sizes, embed_input, keep_prob_, batch_size):
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

    # Stack up multiple LSTM layers
    cell = tf.contrib.rnn.MultiRNNCell(drops)

    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)

    # perform dynamic unrolling of the network,
    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, embed_input, time_major=True, initial_state=initial_state)

    return initial_state, lstm_outputs, final_state, cell


def score():
    return


# 2. Construct the decoder cell
def create_cell(rnn_size, keep_prob):
    # lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1,0.1,seed=2))

    lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    return drop


target_embedding_path = '/home/jehill/python/NLP/datasets/GloVE/MUSE/wiki.multi.de.vec'
source_embedding_path = '/home/jehill/python/NLP/datasets/GloVE/MUSE/wiki.multi.en.vec'

# using existing vocab and embedding
src_embeddings, src_id2word, src_word2id, source_vocab = load_vec(target_embedding_path, nmax=200000)
tgt_embeddings, tgt_id2word, tgt_word2id, target_vocab = load_vec(source_embedding_path, nmax=200000)



class Seq2Seq:

    def __init__(self):

        self.ATTENTION_UNITS = 10  # should be equal to decorder units??
        self.encoder_embedding_dim = 300  # add for  spacy
        self.decoder_embedding_dim = 300  # add from spacy
        self.batch_size = 256
        self.decoder_vocab_size = 200000
        self.encoder_vocab_size = 200000
        self.lstm_sizes = [128, 128]  # number hidden layer in each LSTM
        self.keep_prob = 0.5

        self.rnn_size = 128
        self.num_rnn_layers = 2

        self.encoder_length = 10  # these are length of sentences figure out different for different version
        self.decoder_length = 10  # these are length of sentences figure out different for different version

        self.beam_search = True
        self.beam_width = 10

        self.target_start_token = 10  # '<GO>' need to feed int32 to the network
        self.target_end_token = 10  # '<END>' need to feed int32 to the network

        with tf.variable_scope('rnn_i/o'):
            # use None for batch size and dynamic sequence length
            # embedding to be trained separately, or employ existing embedding (for e.g. from spacy)
            # input's below are assumed gone through embedding layer

            self.encoder_inputs = tf.placeholder(tf.int32, shape=[self.encoder_length, self.batch_size])
            self.decoder_inputs = tf.placeholder(tf.int32, shape=[self.decoder_length, self.batch_size])

            self.source_sequence_length = tf.placeholder(tf.int32, shape=[
                self.batch_size])  # i.e. self.decoder_length use to limit training iter while unrolling .. nneds to be corrected
            self.target_sentence_length = tf.placeholder(tf.int32, shape=[self.batch_size])

        with tf.variable_scope('embeddings'):

            # these are when once wants to train the embedding from scratch ..
            # self.encoder_embeddings=tf.get_variable("encoder_embeddings", [self.decoder_vocab_size,self.encoder_embedding_dim])
            # self.decoder_embeddings=tf.get_variable("decoder_embeddings", [self.encoder_vocab_size,self.decoder_embedding_dim])

            self.encoder_embeddings = tf.get_variable(name="encoder_embedding", shape=np.shape(src_embeddings),
                                                      initializer=tf.constant_initializer(src_embeddings),
                                                      trainable=False)
            self.decoder_embeddings = tf.get_variable(name="decoder_embedding", shape=np.shape(tgt_embeddings),
                                                      initializer=tf.constant_initializer(tgt_embeddings),
                                                      trainable=False)

            self.encoder_input_embeddings = tf.nn.embedding_lookup(self.encoder_embeddings,
                                                                   self.encoder_inputs)  # ouput shape [self.encoder_length, self. batch_siz, emd_dimension (300)]
            self.decoder_input_embeddings = tf.nn.embedding_lookup(self.decoder_embeddings, self.decoder_inputs)

        with tf.variable_scope('encoder'):
            self.encoder_initial_state, self.encoder_outputs, self.encoder_final_state, self.encoder_cell = build_lstm_layers(
                self.lstm_sizes,
                self.encoder_input_embeddings,
                self.keep_prob,
                self.batch_size)

        with tf.variable_scope('decoder'):

            self.dec_cell = tf.contrib.rnn.MultiRNNCell(
                [create_cell(rnn_size, self.keep_prob) for rnn_size in self.lstm_sizes])

            self.projection = tf.layers.Dense(self.decoder_vocab_size, use_bias=False)

            self.train_helper = TrainingHelper(inputs=self.decoder_input_embeddings,
                                               sequence_length=self.target_sentence_length,
                                               time_major=True)  # need to calculate source sequence length what is this what does

            """
            # Getting an initial state of all zeros
            self.dec_initial_state = self.encoder_final_state               #self.dec_cell.zero_state(self.batch_size, tf.float32).clone()

            # perform dynamic unrolling of the network,
            self.dec_outputs, self.dec_final_state = tf.nn.dynamic_rnn(self.dec_cell, self.decoder_input_embeddings, time_major=True,
                                                          initial_state=self.dec_initial_state)

   
            
            self.train_decoder = BasicDecoder(self.dec_cell, self.train_helper,
                                              initial_state=self.dec_initial_state,
                                              output_layer=self.projection)  # with out attention
            """

            self.attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])
            self.attention_mechanism = BahdanauAttention(self.rnn_size,
                                                         self.attention_states)  # pass source_sequence_lenght for improved efficieny ??
            self.attention_cell = AttentionWrapper(self.dec_cell, self.attention_mechanism, attention_layer_size=128)

            self.attention_state = self.attention_cell.zero_state(self.batch_size, dtype=tf.float32)

            self.attention_state.clone(cell_state=self.encoder_final_state)

            self.train_decoder = BasicDecoder(self.attention_cell, self.train_helper,
                                              initial_state=self.attention_state,
                                              output_layer=self.projection)  # using attention

            self.decoded_output_train, self.final_state_attn_decoder, _final_sequence_length = dynamic_decode(
                self.train_decoder, output_time_major=True, impute_finished=False)  # see other options here..

            self.logits = self.decoded_output_train.rnn_output

            print("Shape of logits is .... ")

            print(self.logits)

        if self.beam_search:

            # see issue for more details https://github.com/tensorflow/nmt/issues/93 about this implementation

            # self.beam_search_init_state = tile_batch(self.attention_state, multiplier=self.beam_width)

            self.tiled_encoder_outputs = tile_batch(tf.transpose(self.encoder_outputs, [1, 0, 2]),
                                                    multiplier=self.beam_width)
            self.tiled_encoder_final_state = tile_batch(self.encoder_final_state, multiplier=self.beam_width)

            self.attention_mechanism_beam = BahdanauAttention(self.rnn_size, self.tiled_encoder_outputs)
            self.attention_cell_beam = AttentionWrapper(self.dec_cell, self.attention_mechanism_beam,
                                                        attention_layer_size=128)

            self.attention_state_beam = self.attention_cell_beam.zero_state(self.batch_size * self.beam_width,
                                                                            dtype=tf.float32).clone(
                cell_state=self.tiled_encoder_final_state)  # may need to multiply by beam width

            # self.attention_state_beam.clone(cell_state=self.tiled_encoder_final_state)

            self.inference_decoder = BeamSearchDecoder(cell=self.attention_cell_beam, embedding=self.decoder_embeddings,
                                                       start_tokens=tf.fill([self.batch_size], self.target_start_token),
                                                       end_token=self.target_end_token,
                                                       initial_state=self.attention_state_beam,
                                                       beam_width=self.beam_width,
                                                       output_layer=self.projection,
                                                       length_penalty_weight=0.0)


        else:
            self.inference_helper = GreedyEmbeddingHelper(embedding=self.encoder_outputs,
                                                          start_tokens=tf.fill([self.batch_size],
                                                                               self.target_start_token),
                                                          end_token=self.target_end_token)  # need to change decorder embeddeding

            self.inference_decoder = BasicDecoder(self.decoder_cell, self.inference_helper,
                                                  initial_state=self.decoder_initial_state)  # using attention

        self.decoded_output_inference, _, _ = dynamic_decode(self.inference_decoder, output_time_major=True,
                                                             maximum_iterations=100)  # tf.round(self.target_sentence_length*2))

        self.translations = self.decoded_output_inference.predicted_ids

        with tf.variable_scope('rnn_loss'):
            # use cross_entropy as class loss

            self.target_labels = tf.placeholder(tf.int32, shape=[self.batch_size, self.decoder_length])
            self.loss = tf.losses.sparse_softmax_cross_entropy(
                labels=self.target_labels,
                logits=self.logits)  # may need to sort out this see the comment in doc.

            self.optimizer = tf.train.AdamOptimizer(0.02)  # .minimize(self.loss) if no gradient clipping required

            self.params = tf.trainable_variables()
            self.gradients = tf.gradients(self.loss, self.params)
            self.clipped_gradients = tf.clip_by_global_norm(self.gradients, clip_norm=5.0)  # how select this value
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            self.train_operation = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.params),
                                                                  global_step=self.global_step)

        with tf.variable_scope('rnn_accuracy'):

            self.accuracy = tf.contrib.metrics.accuracy(
                labels=tf.argmax(self.translated_input, axis=1),
                predictions=self.prediction)

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())  # don't forget to initial all variables
            self.saver = tf.train.Saver()  # a saver is for saving or restoring your trained weight

    def inference(self, batch_x, batch_y, batch_size):
        """
         NEED TO RE-WRITE this function interface by adding the state
        :param batch_x:
        :param batch_y:
        :return

        """

        # restore the model

        # with tf.Session() as sess:
        #    model=model.restore();

        # test_state = model.cell.zero_state(batch_size, tf.float32)
        """
        fd = {}
        fd[self.inputs] = batch_x
        fd[self.groundtruths] = batch_y
        fd[self.initial_state] = test_state
        prediction, accuracy = self.sess.run([self.prediction, self.accuracy], fd)

        return prediction, accuracy
        """

    def train(self):
        return

    def save(self, e):
        self.saver.save(self.sess, 'model/rnn/seq2seq_translate_%d.ckpt' % (e + 1))

    def restore(self, e):
        self.saver.restore(self.sess, 'model/rnn/seq2seq_translate_%d.ckpt' % (e))


if __name__ == '__main__':
    model = Seq2Seq()
