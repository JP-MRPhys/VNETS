import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

# n_activations_prev_layer = patch_volume_prev * in_channels
# n_activations_current_layer = patch_volume * out_channels
# sqrt(3/(n_activations_prev_layer + n_activations_current_layer)) (assuming prev_patch==curr_patch)
def xavier_normal_dist_conv3d(shape):
    return tf.truncated_normal(shape, mean=0,
                               stddev=tf.sqrt(3. / (tf.reduce_prod(shape[:3]) * tf.reduce_sum(shape[3:]))))


def xavier_uniform_dist_conv3d(shape):
    with tf.variable_scope('xavier_glorot_initializer'):
        denominator = tf.cast((tf.reduce_prod(shape[:3]) * tf.reduce_sum(shape[3:])), tf.float32)
        lim = tf.sqrt(6. / denominator)
        return tf.random_uniform(shape, minval=-lim, maxval=lim)


# parametric leaky relu
def prelu(x):
    alpha = tf.get_variable('alpha', shape=x.get_shape()[-1], dtype=x.dtype, initializer=tf.constant_initializer(0.1))
    return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)

def convolution_block(parent, kernal_size, stride, padding, keep_prob, name, batch_normalisation, tanh):
    # returns a conv block using the kernel_size stride and padding, keep_prob is the next keep prob for drop out layer

    with tf.variable_scope(name):
        init = tf.truncated_normal_initializer(stddev = 0.25)
        weights = tf.get_variable(name = 'weights', shape = kernal_size, dtype = 'float32', initializer = init)
        conv = tf.nn.conv2d(parent, weights, stride, padding = 'SAME')
        bias = tf.get_variable(name = 'bias', shape = [kernal_size[-1]], dtype = 'float32', initializer = init)
        conv_with_bias = tf.nn.bias_add(conv, bias)

        if (batch_normalisation):
            print("Applied batch normalisation for the layer" + name)
            conv_with_bias=slim.batch_norm(tf.nn.relu(conv_with_bias), activation_fn=None)

        if (tanh):
            print("Apply tanh activation")
            conv_out=tf.nn.tanh(conv_with_bias)
        else:
            conv_out=prelu(conv_with_bias)
        #conv_dropout=tf.nn.dropout(conv, keep_prob=keep_prob)

        return conv_out





def convolution(parent, kernal_size, stride, padding, keep_prob, name):

    # returns a conv block using the kernel_size stride and padding, keep_prob is the next keep prob for drop out layer

    with tf.variable_scope(name):
        # normal_input=slim.batch_norm(parent) input are normalised
        init = tf.truncated_normal_initializer(stddev = 0.0005)
        weights = tf.get_variable(name = 'weights', shape = kernal_size, dtype = 'float32', initializer = init)
        conv = tf.nn.conv2d(parent, weights, stride, padding = 'SAME')

        bias = tf.get_variable(name = 'bias', shape = [kernal_size[-1]], dtype = 'float32', initializer = init)
        conv_with_bias = tf.nn.bias_add(conv, bias)
        conv_batch=slim.batch_norm(tf.nn.relu(conv_with_bias), activation_fn=None)
        conv=tf.nn.relu(conv_batch)
        conv_dropout=tf.nn.dropout(conv, keep_prob=keep_prob)

        return conv_dropout

def convolution_3d(layer_input, filter, strides, padding, name):

    with tf.variable_scope(name):

        w = tf.Variable(initial_value=xavier_uniform_dist_conv3d(shape=filter), name='weights')
        b = tf.Variable(tf.constant(1.0, shape=[filter[-1]]), name='biases')
        conv= tf.nn.conv3d(layer_input, w, strides, padding) + b

        return conv

def _get_bilinear_filter(filter_shape, upscale_factor):
    kernel_size = filter_shape[1]
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            value = (1 - abs((x - centre_location) / upscale_factor)) * (
                1 - abs((y - centre_location) / upscale_factor))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)
    for i in range(filter_shape[2]):
        for j in range(filter_shape[3]):
            weights[:, :, i, j] = bilinear
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init,
                                       shape=weights.shape)
    return bilinear_weights

def _get_bilinear_filter_3d(filter_shape, upscale_factor):
    kernel_size = filter_shape[1]
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1], filter_shape[2]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            for z in range(filter_shape[2]):
                value = (1 - abs((x - centre_location) / upscale_factor)) * (
                    1 - abs((y - centre_location) / upscale_factor)) * (
                    1 - abs((z - centre_location) / upscale_factor))
                bilinear[x, y, z] = value
    weights = np.zeros(filter_shape)
    for i in range(filter_shape[3]):
        weights[:, :, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init,
                                       shape=weights.shape)
    return bilinear_weights


def upsampling(parent, shape, output_channel, input_channel, upscale_factor, name):


        kernel_size = 2 * upscale_factor - upscale_factor % 2

        print(kernel_size)

        stride = upscale_factor
        strides = [1, stride, stride, 1]

        with tf.variable_scope(name):

            output_shape = [shape[0], shape[1], shape[2], output_channel]
            filter_shape = [kernel_size, kernel_size, output_channel, input_channel]
            weights = _get_bilinear_filter(filter_shape, upscale_factor)
            deconv = tf.nn.conv2d_transpose(parent, weights, output_shape, strides=strides, padding='SAME')
            bias_init = tf.constant(0.0, shape=[output_channel])
            bias = tf.get_variable('bias', initializer=bias_init)
            dconv_with_bias = tf.nn.bias_add(deconv, bias)
            dconv_with_bias=slim.batch_norm(dconv_with_bias)
            dconv=tf.nn.relu(dconv_with_bias)
            #dconv=tf.nn.dropout(dconv, 0.25)


        return dconv

def upsampling_3d(parent, shape, output_channel, input_channel, upscale_factor, name):

    kernel_size = 2 * upscale_factor - upscale_factor % 2
    stride = upscale_factor
    strides = [1, stride, stride, stride, 1]
    with tf.variable_scope(name):
        output_shape = [shape[0], shape[1], shape[2], shape[3], output_channel]
        print (shape)

        filter_shape = [kernel_size, kernel_size, kernel_size, output_channel, input_channel]
        print(filter_shape)
        weights = _get_bilinear_filter_3d(filter_shape, upscale_factor)
        deconv = tf.nn.conv3d_transpose(parent, weights, output_shape,
                                        strides = strides, padding='SAME')

        bias_init = tf.constant(0.0, shape=[output_channel])
        bias = tf.get_variable('bias', initializer = bias_init)
        dconv_with_bias = tf.nn.bias_add(deconv, bias)

        dcovn=tf.nn.relu(dconv_with_bias)


    return tf.nn.dropout(dcovn,0.20)


def deconvolution_3d(layer_input, filter, output_shape, strides, padding, name):

    with tf.variable_scope(name):



        w = tf.Variable(initial_value=xavier_uniform_dist_conv3d(shape=filter), name='weights')
        b = tf.Variable(tf.constant(1.0, shape=[filter[-2]]), name='biases')

        dcov=tf.nn.conv3d_transpose(layer_input, w, output_shape, strides, padding) + b;

    return dcov


def conv2d_transpose(input_, output_shape,
                     k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
                     name="conv2d_transpose", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def max_pooling_3d(layer_input, filter, strides, name):

    padding='SAME'

    with tf.variable_scope(name):

        max=tf.nn.max_pool3d(layer_input, filter, strides, padding);
    return max


def max_pooling(layer_input, filter, strides, name):

    padding='SAME'

    with tf.variable_scope(name):

        max=tf.nn.max_pool(layer_input, filter, strides, padding);

    return max


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias