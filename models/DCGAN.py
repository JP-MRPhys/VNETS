"""
DC-GAN to leaning probablity distribution for MRI image objects

Features:
1. Generator (U-NET) with strided convolutiono (i.e. up-sampling)
   f. relu except h (for output layer) activation
2. Discriminator
   a. replaced pooling layers by strided convolutions, batch norm except the discriminator input layer
   b. No fully connected layer at output (replaced with Flatten output)
   c. leaky-relu activation
3. Adversarial Training.
4. Adding Perceptual (VGG) loss and pixelwise loss to reduce "halluincation" in the generator outputs, loss function as-per DAGAN, Yang et al, IEEE, 2018
5. For employing Conditional -DC-GAN (changes required as per comments in the code)
"""

from utils.Layers import convolution_block, upsampling, prelu, linear
import tensorflow as tf
import os
import nibabel as nib
import numpy as np
import tensorflow.contrib.slim as slim
import random
from models.unets import create_conditional_u_net
from utils.utils import *
from utils.fileIO import *
from models.VGG16 import vgg16_cnn_emb
# Just disables the warning, doesn't enable AVX/FMA
import os


class DCGAN:
    def __init__(self, vggdir, name):
        # network parameters
        self.vggdir = vggdir
        # self.traindir=traindir
        # self.labeldir=labeldir
        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        self.num_epochs = 100
        self.display_step = 20
        self.global_step = 0
        self.w = 64  # x
        self.h = 64  # y
        self.z_dim = 100
        self.w2 = self.w / 2  #
        self.h2 = self.h / 2  #
        self.d = 1  # z or channels
        self.X_train = tf.placeholder(tf.float32, [None, None, None, self.d], name='X_train')
        self.X_conditioning = tf.placeholder(tf.float32, [None, None, None, self.d], name='X_train_conditioning')
        self.batch_size = 5;
        self.num_classes = 10  # anging number of features to 5
        self.g_gamma = 0  # 0.025weight for perceptual loss
        self.g_alpha = 0  # 0.1weight for pixel loss
        self.g_beta = 0.  # 0.1weight for frequency loss
        self.g_adv = 1  # weight for frequency loss

        self.training_dir = []
        self.labels_dir = []

        self.training_dir.append('/home/jehill/data_disk/ML_data/SEGMENTATION/PHASE2/Task03_Liver/imagesTr/')
        self.training_dir.append('/home/jehill/data_disk/ML_data/SEGMENTATION/PHASE2/Task02_Heart/imagesTr/')
        """"

        self.training_dir.append('/home/jehill/data_disk/ML_data/SEGMENTATION/PHASE2/Task04_Hippocampus/imagesTr/')
        self.training_dir.append('/home/jehill/data_disk/ML_data/SEGMENTATION/PHASE2/Task05_Prostate/imagesTr/')
        self.training_dir.append('/home/jehill/data_disk/ML_data/SEGMENTATION/PHASE2/Task06_Lung/imagesTr/')
        self.training_dir.append(
            '/home/jehill/data_disk/ML_data/SEGMENTATION/PHASE2/Task07_Pancreas/Task07_Pancreas/imagesTr/')

        """

        self.labels_dir.append('/home/jehill/data_disk/ML_data/SEGMENTATION/PHASE2/Task03_Liver/labelsTr/')
        self.labels_dir.append('/home/jehill/data_disk/ML_data/SEGMENTATION/PHASE2/Task02_Heart/labelsTr/')
        """
        self.labels_dir.append('/home/jehill/data_disk/ML_data/SEGMENTATION/PHASE2/Task04_Hippocampus/labelsTr/')
        self.labels_dir.append('/home/jehill/data_disk/ML_data/SEGMENTATION/PHASE2/Task05_Prostate/labelsTr/')
        self.labels_dir.append('/home/jehill/data_disk/ML_data/SEGMENTATION/PHASE2/Task06_Lung/labelsTr/')
        self.labels_dir.append(
            '/home/jehill/data_disk/ML_data/SEGMENTATION/PHASE2/Task07_Pancreas/Task07_Pancreas/labelsTr/')

        """

        # now create the network
        self.keep_prob = 0.5  # that the drop
        self.drop_out = self.keep_prob

        # Initialize Network weights
        self.initializer = tf.truncated_normal_initializer(stddev=0.2)

        # Input for Generator
        # self.z_in = tf.placeholder(tf.float32, [None, self.w, self.h, self.d], name='z_in')
        print(str(self.w2))
        print(str(self.h2))

        # Inputs for Discriminator and conditioning mask

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        # self.input_image = tf.placeholder(shape=[None, self.w, self.h, self.d], dtype=tf.float32, name='input_image')
        # self.conditioning_input = tf.placeholder(shape=[None, self.w, self.h, self.d], dtype=tf.float32,name="conditioning_input")  # must be of the same dimension as the input_image

        # self.X_train=distort_img(self.X_train)

        self.input_image = tf.image.resize_images(self.X_train, [np.int(self.w), np.int(self.h)])
        self.conditioning_input = tf.image.resize_images(self.X_conditioning, [np.int(self.w), np.int(self.h)])
        self.input_image_resize = tf.image.resize_images(self.input_image, [np.int(self.w2), np.int(self.h2)])
        self.input_image_244 = tf.image.resize_images(self.input_image, [244, 244])  # resize the the input image to VGG

        # self.conditioning_input2 = tf.image.resize_images(self.conditioning_input, tf.constant([self.w/2])) #generator images are half size so we resize our images
        self.conditioning_input_resize = tf.image.resize_images(self.conditioning_input, [np.int(self.w2), np.int(
            self.h2)])  # these needs to fixed using variable

        # Creating Images for ranom vectors (replaced with a U-NET)
        # self.generator_logits = self.generator(self.input_image, self.conditioning_input)
        # self.Gz = tf.reduce_mean(self.generator_logits, 3, keepdims=True, name='generator_output')

        self.Gz = self.generator_2(self.z)
        self.Gz_244 = tf.image.resize_images(self.Gz, [244, 244])
        # self.Gz = tf.image.resize_images(self.Gz, [self.w, self.h])

        # self.segmented_image = tf.reduce_max(self.generator_logits,3, keep_dims=True, name='segmented_image')
        # self.segmented_image_dim = tf.expand_dims(self.segmented_image,axis=3)
        # self.segmented_image = tf.reduce_max(self.generator_logits, axis=3, keepdims=True,name='segmented_image')
        # self.segmented_image_2 = tf.argmax(self.generator_logits,-1)

        # Probabilities for real images
        # self.Dx, self.Dx_logits = self.discriminator(self.input_image, self.conditioning_input)
        self.Dx, self.Dx_logits = self.discriminator(self.input_image)

        # Probabilities for generator images
        print("Discriminator Shape 2:")
        # self.Dz, self.Dz_logits = self.discriminator(self.Gz, self.conditioning_input, reuse=True)
        self.Dz, self.Dz_logits = self.discriminator(self.Gz, reuse=True)

        # VGG data
        self.net_vgg_conv4_good, _ = vgg16_cnn_emb(self.input_image_244, reuse=False)
        self.net_vgg_conv4_gen, _ = vgg16_cnn_emb(self.Gz_244, reuse=True)

        # Adversarial training using cross entropy for G and D loss, plus additional losses
        # Discriminator loss

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dx_logits, labels=tf.ones_like(self.Dx)))

        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dz_logits, labels=tf.zeros_like(self.Dz)))

        self.d_loss = self.d_loss_fake + self.d_loss_real

        # Generator loss (adversarial)
        self.g_loss_1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dz_logits, labels=tf.ones_like(self.Dz)))

        # additional losses

        # generator loss (pixel-wise)
        g_nmse_a = tf.sqrt(tf.reduce_sum(tf.squared_difference(self.Gz, self.input_image), axis=[1, 2, 3]))
        g_nmse_b = tf.sqrt(tf.reduce_sum(tf.square(self.input_image), axis=[1, 2, 3]))
        g_nmse = tf.reduce_mean(g_nmse_a / g_nmse_b)

        # generator loss (frequency)
        fft_good_abs = tf.map_fn(fft_abs_for_map_fn, self.input_image)
        fft_gen_abs = tf.map_fn(fft_abs_for_map_fn, self.Gz)
        g_fft = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(fft_good_abs, fft_gen_abs), axis=[1, 2]))

        # generator loss (perceptual-using VGG)
        g_perceptual = tf.reduce_mean(
            tf.reduce_mean(tf.squared_difference(self.net_vgg_conv4_good.outputs, self.net_vgg_conv4_gen.outputs),
                           axis=[1, 2, 3]))

        # generator loss (total)
        self.g_loss = self.g_adv * self.g_loss_1 + self.g_alpha * g_nmse + self.g_gamma * g_perceptual + self.g_beta * g_fft

        # get the gradients for the generator and discriminator
        self.tvars = tf.trainable_variables()
        self.d_gradients = [var for var in self.tvars if 'd_' in var.name]
        self.g_gradients = [var for var in self.tvars if 'g_' in var.name]

        """   
        print("List of the discriminator gradients")
        for grad in self.d_gradients:
            print(grad)

        print("List of the Generator gradients")
        for grad in self.g_gradients:
            print(grad)

        """

        # Use the Adam Optimizers for discriminator and generator
        # LR = self.learning_rate
        # BTA = 0.5

        self.OptimizerD = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.d_loss,
                                                                                            var_list=self.d_gradients)
        self.OptimizerG = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.g_loss,
                                                                                            var_list=self.g_gradients)

        # summary and writer for tensorboard visulization

        # tf.summary.image("Segmentation", tf.to_float(self.segmented_image))
        tf.summary.image("Generator fake output", self.Gz)
        tf.summary.image("Input image", self.input_image)
        # tf.summary.image("Mask image", self.conditioning_input_resize)

        tf.summary.histogram("Descriminator logits (Real)", self.Dx_logits)
        tf.summary.histogram("Descriminator logits (Fake)", self.Dz_logits)

        tf.summary.scalar("Discriminator loss real", self.d_loss_real)
        tf.summary.scalar("Generator loss fake", self.d_loss_fake)
        tf.summary.scalar("Total Discriminator loss", self.d_loss)
        tf.summary.scalar("Generator loss", self.g_loss)

        self.merged_summary = tf.summary.merge_all()

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.logdir = './' + name  # if not exist create logdir
        self.model_dir = self.logdir + 'final_model'
        self.model_name = name
        self.model_name2 = name

        print("Completed creating the tensor-flow model")

    # we employ y for conditioning by concat it with the input
    # def discriminator(self, image, conditioning_layer, reuse=False):
    def discriminator(self, image, reuse=False):

        PADDING = "SAME"
        STRIDE = [2, 2]

        # we employ y for conditioning by concat it with the input
        # input = tf.concat((image, conditioning_layer), 3)
        input = image

        # Conv Layer 1, No batch normalization, leaky relu activation
        d1_conv = slim.convolution2d(input, 16, [2, 2], stride=STRIDE, padding=PADDING, \
                                     biases_initializer=None, activation_fn=prelu, \
                                     reuse=reuse, scope='d_conv1', weights_initializer=self.initializer)

        # Conv Layer 2, batch normalization, leaky relu activation
        d2_conv = slim.convolution2d(d1_conv, 32, [2, 2], stride=STRIDE, padding=PADDING, \
                                     normalizer_fn=slim.batch_norm, activation_fn=prelu, \
                                     reuse=reuse, scope='d_conv2', weights_initializer=self.initializer)

        # Conv Layer 3, batch normalization, leaky relu activation
        d3_conv = slim.convolution2d(d2_conv, 64, [2, 2], stride=STRIDE, padding=PADDING, \
                                     normalizer_fn=slim.batch_norm, activation_fn=prelu, \
                                     reuse=reuse, scope='d_conv3', weights_initializer=self.initializer)

        # Conv Layer 3, batch normalization, leaky relu activation
        d4_conv = slim.convolution2d(d3_conv, 128, [2, 2], stride=STRIDE, padding=PADDING, \
                                     activation_fn=prelu, reuse=reuse, scope='d_conv4',
                                     weights_initializer=self.initializer)

        # Conv Layer 3, batch normalization, leaky relu activation
        d5_conv = slim.convolution2d(d4_conv, 256, [2, 2], stride=STRIDE, padding=PADDING, \
                                     activation_fn=prelu, reuse=reuse, scope='d_conv5',
                                     weights_initializer=self.initializer)

        d6_conv = slim.convolution2d(d4_conv, self.num_classes, [1, 1], stride=STRIDE, padding=PADDING, \
                                     activation_fn=prelu, reuse=reuse, scope='d_conv6',
                                     weights_initializer=self.initializer)  # for first working version 7 we employed d4_conv

        # Dense Layer (Fully connected), sigmoid activation
        d5_dense = slim.flatten(d6_conv, scope='d_output')

        return tf.nn.sigmoid(d5_dense), d5_dense

    def generator(self, input_image, conditioning_layer):

        """
        :param input_image: with dimension of w*h*d here d: is depth (gray-scale) can be channels (for RGB)
        :param conditioning_layer: conditional maks
        :return: logits from U-NET
        """

        G = create_conditional_u_net(input_image, conditioning_layer, self.num_classes, self.keep_prob)
        # G1=tf.identity(G, name='segmentation_output') #using a tensor op to used while inference
        # g=tf.nn.softmax(G, name='segmentation_output2')

        return G

    def generator_2(self, z):

        """
        :param z: random array input dimension (batch_size, z_dim)
        :return: image (Gz)
        """

        z_, self.h0_w, self.h0_b = linear(z, 64 * 4 * 4 * 8, 'g_h0_lin', with_w=True)

        z_resize = tf.reshape(z_, [-1, 4, 4, 64 * 8])  # add a-relu
        z_resize = tf.nn.relu(z_resize)
        # up_1 = upsampling(z_resize, [self.batch_size, 8, 8], 512, 1024, 2, name='g_up2')
        up_2 = upsampling(z_resize, [self.batch_size, 8, 8], 256, 512, 2, name='g_up3')
        up_3 = upsampling(up_2, [self.batch_size, 16, 16], 128, 256, 2, name='g_up4')
        up_4 = upsampling(up_3, [self.batch_size, 32, 32], 32, 128, 2, name='g_up5')
        up_5 = upsampling(up_4, [self.batch_size, 64, 64], 1, 32, 2, name='g_up6')
        # up_6 = upsampling(up_5, [self.batch_size, 128, 128], 16, 32, 2, name='g_up7')
        # up_7 = upsampling(up_6, [self.batch_size, 256, 256], 1,16 , 2, name='g_up8')

        return tf.nn.tanh(up_5)

    def segment(self, input_image_filename):
        # print("Add the prediction logic here")
        # load the model
        # run the session in predict_op
        # may need to do batch inference (i.e. segment image slice-by-slice and time and then merge the segmentation)
        # save the segmented image as a nifti file

        # check if nifti filename or not
        print("Segment file name")

        print("Image to be segmented" + input_image_filename)
        image = self.get_image(input_image_filename)
        [t, x, y, z] = image.shape;

        if (
                x == self.w and y == self.h and z == self.d):  # check the input and the output dim otherwise reshape the image and feed to the network (WIP)
            # sess = self.load_model(self.model_name2)

            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as new_sess:

                saver = tf.train.import_meta_graph((self.model_name2 + '.meta'))
                # saver.restore(new_sess, self.model_dir)
                saver.restore(new_sess, tf.train.latest_checkpoint("./"))

                print("re store the session completed")

                if (new_sess._closed):
                    print("tensorflow session is closed not segmenting")

                else:

                    segmented_image = new_sess.run([self.Gz], feed_dict={self.X_train: np.float32(image)})
                    return segmented_image
        else:
            print("Input is not of the same as training re-shape the data and feed to the network")

    def train(self):

        with tf.device('/gpu:0'):
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as self.sess:

                self.train_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())

                # load weights for VGG
                # load weights
                npz = np.load(self.vggdir + 'vgg16_weights.npz')
                vgg_weights = []
                for idx, val in enumerate(sorted(npz.items())[0:20]):
                    print("  Loading pretrained VGG16, CNN part %s" % str(val[1].shape))
                    vgg_weights.append(self.net_vgg_conv4_good.all_params[idx].assign(val[1]))

                print("Completed loading the weights")
                self.sess.run(vgg_weights)
                self.sess.run(self.init)

                # self.net_vgg_conv4_good.print_params(False)

                counter = 0
                learningrate = 0.001

                for epoch in range(0, self.num_epochs):

                    if (epoch % 2 == 0):
                        learningrate = learningrate / 10

                    for i in range(0, len(self.training_dir)):

                        # get the file names
                        filenames = get_image_filenames_task2(self.training_dir[i])
                        print("Number training file " + str(len(filenames)))

                        np.random.shuffle(filenames)
                        Average_loss_G = 0
                        Average_loss_D = 0

                        for file in filenames:

                            training_images, training_labels = get_training_set(i, filenames[i])
                            training_images = apply_random_deformation(training_images)
                            [batch_length, x, y, z] = training_images.shape

                            if (
                                    x == self.w and y == self.h and z == self.d):  # check the input and the output dim otherwise reshape the image and feed to the network (WIP)

                                # print("Training image" + image_file)

                                for idx in range(0, batch_length, self.batch_size):
                                    z_samples = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim)).astype(
                                        np.float32)

                                    # print("Current index" + str(idx))

                                    batch_images = training_images[idx:idx + self.batch_size, :, :, :]
                                    batch_labels = training_labels[idx:idx + self.batch_size, :, :, :]

                                    # print(batch_images.shape)

                                    summary1, opt, loss_D = self.sess.run(
                                        [self.merged_summary, self.OptimizerD, self.d_loss],
                                        feed_dict={self.X_train: batch_images,
                                                   self.learning_rate: learningrate,
                                                   self.z: z_samples})

                                    opt, loss_G = self.sess.run([self.OptimizerG, self.g_loss_1],
                                                                feed_dict={self.z: z_samples,
                                                                           self.learning_rate: learningrate})

                                    # emphrical solution to the avoid gradients vansihing issues by training generator twice, different from paper
                                    summary2, opt, loss_G = self.sess.run(
                                        [self.merged_summary, self.OptimizerG, self.g_loss_1],
                                        feed_dict={self.z: z_samples,
                                                   self.learning_rate: learningrate})

                                    counter += 1

                                    Average_loss_D = (Average_loss_D + loss_D) / 2
                                    Average_loss_G = (Average_loss_G + loss_G) / 2
                                    self.train_writer.add_summary(summary1, counter)
                                    self.train_writer.add_summary(summary2)

                        print("Epoch: ", str(epoch) + " learning rate:" + str(learningrate) + " Generator loss: " + str(
                            Average_loss_G) + "Discriminator loss: " + str(Average_loss_D))

                        # if (epoch % 5 == 0):
                        # learningrate=learningrate/10
                        # self.saver.save(self.sess, self.model_name)

                print("Training completed ")
                # self.save_model(self.model_name)

    def print_shape(self, tensor):
        print(tensor.get_shape().as_list())


if __name__ == '__main__':
    VGG_dir = './trained_model/VGG/'
    network = DCGAN(VGG_dir, 'generator2')
network.train()