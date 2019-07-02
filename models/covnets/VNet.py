import tensorflow as tf
from utils.Layers import convolution_3d, deconvolution_3d, prelu
import os
import numpy as np
from utils.loss_functions import dice
from utils.get_image_data import read_tiff, get_image_filenames, get_batch_data

def convolution_block(layer_input, n_channels, num_convolutions, dropout_keep_prob):
    x = layer_input

    for i in range(num_convolutions - 1):
        with tf.variable_scope('conv_' + str(i+1)):
            x = convolution_3d(x, [5, 5, 5, n_channels, n_channels], [1, 1, 1, 1, 1])
            x = prelu(x)
            x = tf.nn.dropout(x, dropout_keep_prob)

    with tf.variable_scope('conv_' + str(num_convolutions)):
        x = convolution_3d(x, [5, 5, 5, n_channels, n_channels], [1, 1, 1, 1, 1])
        x = x + layer_input
        x = prelu(x)
        x = tf.nn.dropout(x, dropout_keep_prob)

    return x


def convolution_block_2(layer_input, fine_grained_features, n_channels, num_convolutions, dropout_keep_prob):

    x = tf.concat((layer_input, fine_grained_features), axis=-1)

    if num_convolutions == 1:
        with tf.variable_scope('conv_' + str(1)):
            x = convolution_3d(x, [5, 5, 5, n_channels * 2, n_channels], [1, 1, 1, 1, 1])
            x = x + layer_input
            x = prelu(x)
            x = tf.nn.dropout(x, dropout_keep_prob)
        return x

    with tf.variable_scope('conv_' + str(1)):
        x = convolution_3d(x, [5, 5, 5, n_channels * 2, n_channels], [1, 1, 1, 1, 1])
        x = prelu(x)
        x = tf.nn.dropout(x, dropout_keep_prob)

    for i in range(1, num_convolutions - 1):
        with tf.variable_scope('conv_' + str(i+1)):
            x = convolution_3d(x, [5, 5, 5, n_channels, n_channels], [1, 1, 1, 1, 1])
            x = prelu(x)
            x = tf.nn.dropout(x, dropout_keep_prob)

    with tf.variable_scope('conv_' + str(num_convolutions)):
        x = convolution_3d(x, [5, 5, 5, n_channels, n_channels], [1, 1, 1, 1, 1])
        x = x + layer_input
        x = prelu(x)
        x = tf.nn.dropout(x, dropout_keep_prob)

    return x


def down_convolution(layer_input, in_channels, dropout_keep_prob):
    with tf.variable_scope('down_convolution'):
        x = convolution_3d(layer_input, [2, 2, 2, in_channels, in_channels * 2], [1, 2, 2, 2, 1])
        x = prelu(x)
        x = tf.nn.dropout(x, dropout_keep_prob)
        return x


def up_convolution(layer_input, output_shape, in_channels, dropout_keep_prob):
    with tf.variable_scope('up_convolution'):
        x = deconvolution_3d(layer_input, [2, 2, 2, in_channels // 2, in_channels], output_shape, [1, 2, 2, 2, 1])
        x = prelu(x)
        x = tf.nn.dropout(x, dropout_keep_prob)
        return x


def v_net(tf_input, dropout_keep_prob, input_channels, output_channels=1, n_channels=16):

    with tf.variable_scope('contracting_path'):

        # if the input has more than 1 channel it has to be expanded because broadcasting only works for 1 input channel
        if input_channels == 1:
            c0 = tf.tile(tf_input, [1, 1, 1, 1, n_channels])
        else:
            with tf.variable_scope('level_0'):
                c0 = convolution_3d(tf_input, [5, 5, 5, input_channels, n_channels], [1, 1, 1, 1, 1])
                c0 = tf.nn.dropout(c0, dropout_keep_prob)
                c0 = prelu(c0)

        with tf.variable_scope('level_1'):
            c1 = convolution_block(c0, n_channels, 1, dropout_keep_prob)
            c12 = down_convolution(c1, n_channels, dropout_keep_prob)

        with tf.variable_scope('level_2'):
            c2 = convolution_block(c12, n_channels * 2, 2, dropout_keep_prob)
            c22 = down_convolution(c2, n_channels * 2, dropout_keep_prob)

        with tf.variable_scope('level_3'):
            c3 = convolution_block(c22, n_channels * 4, 3, dropout_keep_prob)
            c32 = down_convolution(c3, n_channels * 4, dropout_keep_prob)

        with tf.variable_scope('level_4'):
            c4 = convolution_block(c32, n_channels * 8, 3, dropout_keep_prob)
            c42 = down_convolution(c4, n_channels * 8, dropout_keep_prob)

        with tf.variable_scope('level_5'):
            c5 = convolution_block(c42, n_channels * 16, 3, dropout_keep_prob)
            c52 = up_convolution(c5, tf.shape(c4), n_channels * 16, dropout_keep_prob)

    with tf.variable_scope('expanding_path'):

        with tf.variable_scope('level_4'):
            e4 = convolution_block_2(c52, c4, n_channels * 8, 3, dropout_keep_prob)
            e42 = up_convolution(e4, tf.shape(c3), n_channels * 8, dropout_keep_prob)

        with tf.variable_scope('level_3'):
            e3 = convolution_block_2(e42, c3, n_channels * 4, 3, dropout_keep_prob)
            e32 = up_convolution(e3, tf.shape(c2), n_channels * 4, dropout_keep_prob)

        with tf.variable_scope('level_2'):
            e2 = convolution_block_2(e32, c2, n_channels * 2, 2, dropout_keep_prob)
            e22 = up_convolution(e2, tf.shape(c1), n_channels * 2, dropout_keep_prob)

        with tf.variable_scope('level_1'):
            e1 = convolution_block_2(e22, c1, n_channels, 1, dropout_keep_prob)
            with tf.variable_scope('output_layer'):
                logits = convolution_3d(e1, [1, 1, 1, n_channels, output_channels], [1, 1, 1, 1, 1])

    return logits



if __name__ == '__main__':

    """ TRAIN THE VNET"""

    checkpoint_dir = "/tmp/"
    print_every = 1000
    save_every = 10000
    num_inputs = 20
    num_classes = 1
    image_dim_x = 580
    image_dim_y = 420
    image_dim_z = 1
    batch_size = 1
    input_channels = 1
    output_channels = 1

    with tf.name_scope("hyperparameters"):
        regularization = tf.placeholder(tf.float32, name="regularization")
        learning_rate = tf.placeholder(tf.float32, name="learning-rate")

    with tf.name_scope("inputs"):
        y = tf.placeholder(tf.float32, [batch_size, image_dim_x, image_dim_y], name="y-input")
        tf_input = tf.placeholder(dtype=tf.float32,
                                  shape=(batch_size, image_dim_x, image_dim_y, image_dim_z, input_channels),
                                  name='image')

    with tf.name_scope("model"):

        # this is the v-net implementation
        logits = v_net(tf_input, input_channels, 1, output_channels)
        print("Completed creating the model")

    # This is a logistic classifier, so the loss function is the logistic loss.
    with tf.name_scope("loss-function"):
        loss = dice(logits, y)

    # Use the ADAM optimizer to minimize the loss.
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    # For writing training checkpoints and reading them back in.
    saver = tf.train.Saver()
    tf.gfile.MakeDirs(checkpoint_dir)

    with tf.Session() as sess:
        # Write the graph definition to a file. We'll load this in the test.py script.
        tf.train.write_graph(sess.graph_def, checkpoint_dir, "graph.pb", False)

        sess.run(init)

        currentdir = os.getcwd()
        datadir1 = os.path.join(currentdir, 'train')
        mask_files, image_files = get_image_filenames(datadir1)
        index = 0

        step = 0
        # epochs only one at moment
        while (index < len(image_files)):

            # get the batch data
            filelist = image_files[index:index + batch_size];
            x_train, y_train = get_batch_data(filelist, batch_size, image_dim_x, image_dim_y);
            print("Training data obtained")
            x_train.reshape([batch_size, input_channels, image_dim_x, image_dim_y])
            y_train.reshape([batch_size, input_channels, image_dim_x, image_dim_y])
            index = index + batch_size + 1;

            # Run the optimizer over the entire training set at once. For larger datasets
            # you would train in batches of 100-1000 examples instead of the entire thing.
            feed = {tf_input: x_train, y: y_train, learning_rate: 1e-2, regularization: 1e-5}
            sess.run(train_op, feed_dict=feed)

            # Print the loss once every so many steps. Because of the regularization,
            # at some point the loss won't become smaller anymore. At that point, it's
            # safe to press Ctrl+C to stop the training.
            # if step % print_every == 0:
            # train_accuracy, loss_value = sess.run([accuracy, loss], feed_dict=feed)
            # print("step: %4d, loss: %.4f, training accuracy: %.4f" % \
            #        (step, loss_value, train_accuracy))

            step += 1

            # Save the model. You should only press Ctrl+C after you see this message.
            if step % save_every == 0:
                checkpoint_file = os.path.join(checkpoint_dir, "model")
                saver.save(sess, checkpoint_file)

    print("*** SAVED MODEL ***")