from utils.Layers import convolution_block, upsampling, prelu
from utils.Layers import  upsampling
import tensorflow as tf
import numpy as np

def print_shape(tensor):
    print(tensor.get_shape().as_list())


def create_u_net_small(input, num_classes, keep_prob):
    # u net with 5 block layers

    # Pass the image to obtain
    # input=tf.nn.lrn(input)   #normalisation step for intensity scaling

    conv1_1 = convolution_block(input, [3, 3, 1, 64], [1, 1, 1, 1], 'SAME', keep_prob, 'g_conv1_1',
                                batch_normalisation=True, tanh=False)
    conv1_2 = convolution_block(conv1_1, [3, 3, 64, 64], [1, 2, 2, 1], 'SAME', keep_prob, name='g_conv1_2',batch_normalisation=True,tanh=False)

    conv2_1 = convolution_block(conv1_2, [3, 3, 64, 128], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv2_1',batch_normalisation=True,tanh=False)
    conv2_2 = convolution_block(conv2_1, [3, 3, 128, 128], [1, 2, 2, 1], 'SAME', keep_prob, name='g_conv2_2',batch_normalisation=True,tanh=False)

    conv3_1 = convolution_block(conv2_2, [3, 3, 128, 256], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv3_1', batch_normalisation=True,tanh=False)
    conv3_2 = convolution_block(conv3_1, [3, 3, 256, 256], [1, 2, 2, 1], 'SAME', keep_prob, name='g_conv3_2', batch_normalisation=True,tanh=False)

    """
    conv4_1=convolution_block(conv3_2, [3,3,256,512], [1,1,1,1],'SAME', keep_prob, name='g_conv4_1')
    conv4_2=convolution_block(conv4_1,[3,3,512,512], [1,2,2,1],'SAME', keep_prob, name='g_conv4_2')
    conv5_1=convolution_block(conv4_2, [3,3,512,1024],  [1,1,1,1],'SAME', keep_prob, name='g_conv5_1')
    conv5_2=convolution_block(conv5_1,[3,3,1024,1024],  [1,2,2,1],'SAME', keep_prob, name='g_conv5_2')
    up_6 = upsampling(conv5_2, tf.shape(conv4_2), 512, 1024,2, name='g_up6')
    concat_6 = tf.concat([up_6, conv4_2], axis=3)
    conv6_1 = convolution_block(concat_6, [3,3,1024,512], [1,1,1,1], 'SAME', keep_prob, name='g_conv6_1')
    conv6_2 = convolution_block(conv6_1, [3,3,512,512], [1,1,1,1], 'SAME', keep_prob, name='g_conv6_2')
    """

    up_7 = upsampling(conv3_2, tf.shape(conv2_2), 128, 256, 2, name='g_up7')
    concat_7 = tf.concat([up_7, conv2_2], axis=3)
    conv7_1 = convolution_block(concat_7, [3, 3, 256, 128], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv7_1',batch_normalisation=True,tanh=False)
    conv7_2 = convolution_block(conv7_1, [3, 3, 128, 128], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv7_2', batch_normalisation=True,tanh=False)

    up_8 = upsampling(conv7_2, tf.shape(conv1_2), 64, 128, 2, name='g_up8')
    concat_8 = tf.concat([up_8, conv1_2], axis=3)
    conv8_1 = convolution_block(concat_8, [3, 3, 128, 64], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv8_1', batch_normalisation=True,  tanh=False)
    conv8_2 = convolution_block(conv8_1, [3, 3, 64, 64], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv8_2',batch_normalisation=True, tanh=False)

    # up_9 = upsampling(conv8_2, tf.shape(conv1_2), 32, 64, 2,  name='g_up9')
    # concat_9 = tf.concat([up_9, conv1_2], axis=3)

    # conv9_1 = convolution_block(concat_9, [3,3,64,32], [1,1,1,1], 'SAME', keep_prob, name='g_conv9_1', batch_normalisation=True)
    # conv9_2 = convolution_block(conv9_1, [3,3,32, 32], [1,1,1,1], 'SAME', keep_prob, name='g_conv9_2', batch_normalisation=True)

    conv_10 = convolution_block(conv8_2, [1, 1, 64, num_classes], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv10', batch_normalisation=False, tanh=True)

    print("Completed creating U-NET (small) ")


    ##with tf.variable_scope('g_logits'):
    ##    logits=tf.nn.softmax(conv_10, name='g_output')  # get logits here

    return conv_10


def create_u_net(input, num_classes, keep_prob):
    # Pass the image to obtain
    input=tf.nn.lrn(input)   #normalisation step for intensity scaling

    conv1_1 = convolution_block(input, [3, 3, 1, 64], [1, 1, 1, 1], 'SAME', keep_prob, 'g_conv1_1',         batch_normalisation=True, tanh=False)
    conv1_2 = convolution_block(conv1_1, [3, 3, 64, 64], [1, 2, 2, 1], 'SAME', keep_prob, name='g_conv1_2', batch_normalisation=True, tanh=False)

    conv2_1 = convolution_block(conv1_2, [3, 3, 64, 128], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv2_1', batch_normalisation=True, tanh=False)
    conv2_2 = convolution_block(conv2_1, [3, 3, 128, 128], [1, 2, 2, 1], 'SAME', keep_prob, name='g_conv2_2',batch_normalisation=True, tanh=False)

    conv3_1 = convolution_block(conv2_2, [3, 3, 128, 256], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv3_1', batch_normalisation=True, tanh=False)
    conv3_2 = convolution_block(conv3_1, [3, 3, 256, 256], [1, 2, 2, 1], 'SAME', keep_prob, name='g_conv3_2', batch_normalisation=True, tanh=False)


    conv4_1=convolution_block(conv3_2, [3,3,256,512], [1,1,1,1],'SAME', keep_prob, name='g_conv4_1', batch_normalisation=True, tanh=False)
    conv4_2=convolution_block(conv4_1,[3,3,512,512], [1,2,2,1],'SAME', keep_prob, name='g_conv4_2', batch_normalisation=True, tanh=False)

    conv5_1=convolution_block(conv4_2, [3,3,512,1024],  [1,1,1,1],'SAME', keep_prob, name='g_conv5_1', batch_normalisation=True, tanh=False)
    conv5_2=convolution_block(conv5_1,[3,3,1024,1024],  [1,2,2,1],'SAME', keep_prob, name='g_conv5_2', batch_normalisation=True, tanh=False)

    up_6 = upsampling(conv5_2, tf.shape(conv4_2), 512, 1024,2, name='g_up6')
    concat_6 = tf.concat([up_6, conv4_2], axis=3)
    conv6_1 = convolution_block(concat_6, [3,3,1024,512], [1,1,1,1], 'SAME', keep_prob, name='g_conv6_1', batch_normalisation=True, tanh=False)
    conv6_2 = convolution_block(conv6_1, [3,3,512,512], [1,1,1,1], 'SAME', keep_prob, name='g_conv6_2', batch_normalisation=True, tanh=False)

    up_7 = upsampling(conv6_2, tf.shape(conv3_2), 256, 512, 2, name='g_up7')
    concat_7 = tf.concat([up_7, conv3_2], axis=3)
    conv7_1 = convolution_block(concat_7, [3, 3, 512, 256], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv7_1',batch_normalisation=True, tanh=False)
    conv7_2 = convolution_block(conv7_1, [3, 3, 256, 256], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv7_2', batch_normalisation=True, tanh=False)

    up_8 = upsampling(conv7_2, tf.shape(conv2_2),128, 256, 2, name='g_up8')
    concat_8 = tf.concat([up_8, conv2_2], axis=3)
    conv8_1 = convolution_block(concat_8, [3, 3, 256, 128], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv8_1',batch_normalisation=True, tanh=False)
    conv8_2 = convolution_block(conv8_1, [3, 3, 128,128], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv8_2',  batch_normalisation=True, tanh=False)

    up_9 = upsampling(conv8_2, tf.shape(conv1_2), 64, 128, 2,  name='g_up9')
    concat_9 = tf.concat([up_9, conv1_2], axis=3)
    conv9_1 = convolution_block(concat_9, [3,3,128,64], [1,1,1,1], 'SAME', keep_prob, name='g_conv9_1', batch_normalisation=True, tanh=False)
    conv9_2 = convolution_block(conv9_1, [3,3,64, 64], [1,1,1,1], 'SAME', keep_prob, name='g_conv9_2', batch_normalisation=True, tanh=False)

    conv_10 = convolution_block(conv9_2, [1, 1, 64, num_classes], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv10', batch_normalisation=False, tanh=True)

    print("Completed creating U-NET ")
    print("Number of classes is:" + str(num_classes))


    ##with tf.variable_scope('g_logits'):
    #logits=tf.nn.softmax(conv_10, name='generator_output')  # get logits here

    return conv_10


def create_conditional_u_net(input, y, num_classes, keep_prob):
    # Pass the image to obtain
    input = tf.nn.lrn(input)  # normalisation step for intensity scaling

    image = tf.concat([input, y], axis=3)
    print("Generator input image")
    print_shape(image)

    conv1_1 = convolution_block(image, [3, 3, 2, 64], [1, 1, 1, 1], 'SAME', keep_prob, 'g_conv1_1',
                                batch_normalisation=True, tanh=False)
    conv1_2 = convolution_block(conv1_1, [3, 3, 64, 64], [1, 2, 2, 1], 'SAME', keep_prob, name='g_conv1_2',
                                batch_normalisation=True, tanh=False)

    conv2_1 = convolution_block(conv1_2, [3, 3, 64, 128], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv2_1',
                                batch_normalisation=True, tanh=False)
    conv2_2 = convolution_block(conv2_1, [3, 3, 128, 128], [1, 2, 2, 1], 'SAME', keep_prob, name='g_conv2_2',
                                batch_normalisation=True, tanh=False)

    conv3_1 = convolution_block(conv2_2, [3, 3, 128, 256], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv3_1',
                                batch_normalisation=True, tanh=False)
    conv3_2 = convolution_block(conv3_1, [3, 3, 256, 256], [1, 2, 2, 1], 'SAME', keep_prob, name='g_conv3_2',
                                batch_normalisation=True, tanh=False)

    conv4_1 = convolution_block(conv3_2, [3, 3, 256, 512], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv4_1',
                                batch_normalisation=True, tanh=False)
    conv4_2 = convolution_block(conv4_1, [3, 3, 512, 512], [1, 2, 2, 1], 'SAME', keep_prob, name='g_conv4_2',
                                batch_normalisation=True, tanh=False)

    conv5_1 = convolution_block(conv4_2, [3, 3, 512, 1024], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv5_1',
                                batch_normalisation=True, tanh=False)
    conv5_2 = convolution_block(conv5_1, [3, 3, 1024, 1024], [1, 2, 2, 1], 'SAME', keep_prob, name='g_conv5_2',
                                batch_normalisation=True, tanh=False)

    up_6 = upsampling(conv5_2, tf.shape(conv4_2), 512, 1024, 2, name='g_up6')
    concat_6 = tf.concat([up_6, conv4_2], axis=3)
    conv6_1 = convolution_block(concat_6, [3, 3, 1024, 512], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv6_1',
                                batch_normalisation=True, tanh=False)
    conv6_2 = convolution_block(conv6_1, [3, 3, 512, 512], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv6_2',
                                batch_normalisation=True, tanh=False)

    up_7 = upsampling(conv6_2, tf.shape(conv3_2), 256, 512, 2, name='g_up7')
    concat_7 = tf.concat([up_7, conv3_2], axis=3)
    conv7_1 = convolution_block(concat_7, [3, 3, 512, 256], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv7_1',
                                batch_normalisation=True, tanh=False)
    conv7_2 = convolution_block(conv7_1, [3, 3, 256, 256], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv7_2',
                                batch_normalisation=True, tanh=False)

    up_8 = upsampling(conv7_2, tf.shape(conv2_2), 128, 256, 2, name='g_up8')
    concat_8 = tf.concat([up_8, conv2_2], axis=3)
    conv8_1 = convolution_block(concat_8, [3, 3, 256, 128], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv8_1',
                                batch_normalisation=True, tanh=False)
    conv8_2 = convolution_block(conv8_1, [3, 3, 128, 128], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv8_2',
                                batch_normalisation=True, tanh=False)

    up_9 = upsampling(conv8_2, tf.shape(conv1_2), 64, 128, 2, name='g_up9')
    concat_9 = tf.concat([up_9, conv1_2], axis=3)
    conv9_1 = convolution_block(concat_9, [3, 3, 128, 64], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv9_1',
                                batch_normalisation=True, tanh=False)
    conv9_2 = convolution_block(conv9_1, [3, 3, 64, 64], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv9_2',
                                batch_normalisation=True, tanh=False)

    conv_10 = convolution_block(conv9_2, [1, 1, 64, num_classes], [1, 1, 1, 1], 'SAME', keep_prob, name='g_conv10',
                                batch_normalisation=False, tanh=True)

    print("Completed creating U-NET ")
    print("Number of classes is:" + str(num_classes))


    ##with tf.variable_scope('g_logits'):
    # logits=tf.nn.softmax(conv_10, name='generator_output')  # get logits here

    return conv_10


def loss(logits, labels):
    """Returns the loss function for supervised training inputs are logits from your networks and labeles in same dimension (use one hot encooding if necessary)"""

    return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)


def train_single(self, training_datadir):
    # get the file names
    filenames = self.get_image_filenames(training_datadir)

    with tf.device('/gpu:0'):
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as self.sess:
            self.train_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())

            self.sess.run(self.init)

            counter = 0
            learningrate = 0.005

            for epoch in range(0, self.num_epochs):

                if (epoch % 3 == 0):
                    learningrate = learningrate / 3

                np.random.shuffle(filenames)
                Average_loss_G = 0
                Average_loss_D = 0

                for image_file in filenames[:400]:
                    training_images = self.get_image(image_file)

                    [t, x, y, z] = training_images.shape

                    if (
                            x == self.w and y == self.h and z == self.d):  # check the input and the output dim otherwise reshape the image and feed to the network (WIP)
                        # print("Training image" + image_file)

                        summary1, opt, loss_D = self.sess.run([self.merged_summary, self.OptimizerD, self.d_loss],
                                                              feed_dict={self.input_image: training_images,
                                                                         self.learning_rate: learningrate})

                        opt, loss_G = self.sess.run([self.OptimizerG, self.g_loss],
                                                    feed_dict={self.input_image: training_images,
                                                               self.learning_rate: learningrate})

                        summary2, opt, loss_G = self.sess.run([self.merged_summary, self.OptimizerG, self.g_loss],
                                                              feed_dict={self.input_image: training_images,
                                                                         self.learning_rate: learningrate})

                        counter += 1
                        Average_loss_D = (Average_loss_D + loss_D) / 2
                        Average_loss_G = (Average_loss_G + loss_G) / 2
                        self.train_writer.add_summary(summary1, counter)
                        self.train_writer.add_summary(summary2)

                print("Epoch: ", str(epoch) + " learning rate:" + str(learningrate) + " Generator loss: " + str(
                    Average_loss_G) + "Discriminator loss: " + str(Average_loss_D))

                if (epoch % 5 == 0):
                    # learningrate=learningrate/10
                    self.saver.save(self.sess, self.model_name)

            print("Training completed ")
            self.save_model(self.model_name)


if __name__ == '__main__':


    input_image = tf.placeholder(shape=[None, 192,256,1], dtype=tf.float32)
    data2 = create_u_net_small(input_image, keep_prob=1, num_classes=10)
