"""
Un-supervised semantic segmentation of cardiac images using DC-GAN

Features:

DC-GAN
1. Generator (U-NET) with struded convolution,
   a. U-NET with strided convolution
   b. convolution layer with relu activation
   c. combined output of the classes to input file image
   d. no drop-out layer
   e. skip layers (except the last layers) WIP
   f. tanh activation (to be checked)

2. Discriminator
   a. replaced pooling layers by strided convolutions, batch norm except the discriminator input layer
   b. No fully connected layer (replaced with Flatten output)
   c. leaky-relu activation

3. Adversarial Training on single time of images (code for the entire time series avaliable too)

4. Transfer learning (on other cardiac data-sets) WIP

5. Use the current model for infernece to obtained the segmented output function WIP

6. Use the segmentation compute the LV volumes, systolic and diastolic function (WIP)

"""



from utils.Layers import convolution_block, upsampling, prelu
from utils.Layers import  upsampling
import tensorflow as tf
import os
import nibabel as nib
import numpy as np
import tensorflow.contrib.slim as slim
import random
from models.unets import create_u_net



class cardiacnets:
    def __init__(self, x,y,z, name):
        # network parameters

         self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
         self.num_epochs=100

         self.display_step = 20
         self.global_step = 0
         self.w=x
         self.h=y
         self.d=z
         self.X_train = tf.placeholder(tf.float32, [None,self.w, self.h, self.d], name='X_train')
         self.batch_size=1;
         self.num_classes=30 #anging number of features to 5

         #now create the network
         self.keep_prob=0.5  #that the drop
         self.drop_out=self.keep_prob

        # Initialize Network weights
         self.initializer = tf.truncated_normal_initializer(stddev=0.2)

            # Input for Generator
         #self.z_in = tf.placeholder(tf.float32, [None, self.w, self.h, self.d], name='z_in')


         # Input for Discriminator
         self.input_image = tf.placeholder(shape=[None, self.w, self.h, self.d], dtype=tf.float32, name='input_image')

        # Creating Images for ranom vectors of size z_in
         self.generator_logits = self.generator(self.input_image)
         self.Gz = tf.reduce_mean(self.generator_logits,3, keep_dims=True, name='generator_output')

         self.segmented_image=tf.argmax(self.generator_logits, axis=3, name='segmented_image')
         self.segmented_image2 = tf.identity(self.generator_logits, name='segmented_logits')


         print("Generator Shape:")
         self.print_shape(self.generator_logits)
         self.print_shape(self.Gz)
         self.print_shape(self.segmented_image)

         # Probabilities for real images
         self.Dx, self.Dx_logits = self.discriminator(self.input_image)
         print("Discriminator Shape:")
         self.print_shape(self.Dx)
         self.print_shape(self.Dx_logits)

        # Probabilities for generator images
         print("Discriminator Shape 2:")
         self.Dz, self.Dz_logits = self.discriminator(self.Gz, reuse=True)
         self.print_shape(self.Dz)
         self.print_shape(self.Dz_logits)

        # Adversrial training we shall use cross entropy but sums can used too see below comments
         #Discriminator loss
         self.d_loss_real=  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dx_logits, labels=tf.ones_like(self.Dx)))
         self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dz_logits, labels=tf.zeros_like(self.Dz)))

         self.d_loss=self.d_loss_fake+self.d_loss_real

         # Generator loss
         self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dz_logits, labels=tf.ones_like(self.Dz)))

         self.tvars = tf.trainable_variables()
         self.d_gradients=  [var for var in self.tvars if 'd_' in var.name]
         self.g_gradients = [var for var in self.tvars if 'g_' in var.name]



         print("List of the discriminator gradients")
         for grad in self.d_gradients:
            print(grad)


         print("List of the Generator gradients")
         for grad in self.g_gradients:
            print(grad)

        # Use the Adam Optimizers for discriminator and generator
         #LR = self.learning_rate
         BTA = 0.5

         self.OptimizerD = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.d_loss, var_list=self.d_gradients)
         self.OptimizerG = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.g_loss, var_list=self.g_gradients)


         #summary and writer for tensorboard visulization

         #self.label1 = tf.squeeze(self.segmented_output[:,:,:,0:3])




         #tf.summary.image("Label 1/2/3", self.label1)

         #tf.summary.image("Segmentation", self.segmented_image)
         tf.summary.image("Generator fake output", self.Gz)
         tf.summary.image("Input image", self.input_image)

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
         self.model_dir=self.logdir + 'final_model'
         self.model_name= name #'cardiac-segnet-v1'
         self.model_name2=name #'cardiac-segnet-v1'

         print ("Completed creating the tensorflow model")


    """

        
         # Optimize the discriminator and the generator
         self.d_log_real = tf.log(self.Dx)
         self.d_log_fake = tf.log(1. - self.Dg)
         self.g_log = tf.log(self.Dg)
         self.d_loss = -tf.reduce_mean(self.d_log1 + self.d_log2)
         self.g_loss = -tf.reduce_mean(self.g_log)

         tvars = tf.trainable_variables()

            # Use the Adam Optimizers for discriminator and generator
         LR = 0.0002
         BTA = 0.5
         self.trainerD = tf.train.AdamOptimizer(learning_rate=LR, beta1=BTA)
         self.trainerG = tf.train.AdamOptimizer(learning_rate=LR, beta1=BTA)

            # Gradients for discriminator and generator
         self.gradients_discriminator = self.trainerD.compute_gradients(self.d_loss, tvars[9:])
         self.gradients_generator = self.trainerG.compute_gradients(self.g_loss, tvars[0:9])

            # Apply the gradients
         self.update_D = self.trainerD.apply_gradients(self.gradients_discriminator)
         self.update_G = self.trainerG.apply_gradients(self.gradients_generator)
         
         self.merged_summary = tf.summary.merge_all()
         self.init = tf.global_variables_initializer()
         self.saver = tf.train.Saver()
         self.logdir='./logdir/' #if not exist create logdir

         
         
         """
    def discriminator(self, image, reuse=False):

            PADDING = "SAME"
            STRIDE = [2, 2]

            # Conv Layer 1, No batch normalization, leaky relu activation
            d1_conv = slim.convolution2d(image, 16, [2, 2], stride=STRIDE, padding=PADDING, \
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
                                          activation_fn=prelu, reuse=reuse, scope='d_conv4', weights_initializer=self.initializer)


            # Conv Layer 3, batch normalization, leaky relu activation
            d5_conv = slim.convolution2d(d4_conv, 256, [2,2], stride=STRIDE, padding=PADDING, \
                                          activation_fn=prelu, reuse=reuse, scope='d_conv5', weights_initializer=self.initializer)

            d6_conv = slim.convolution2d(d4_conv, self.num_classes, [1,1], stride=STRIDE, padding=PADDING, \
                                          activation_fn=prelu, reuse=reuse, scope='d_conv6', weights_initializer=self.initializer)


            # Dense Layer (Fully connected), sigmoid activation
            d5_dense = slim.flatten(d6_conv,  scope='d_output')


            print("Discriminator Shapes")

            self.print_shape(d1_conv)
            self.print_shape(d2_conv)
            self.print_shape(d3_conv)
            #self.print_shape(d4_conv)
            self.print_shape(d5_dense)

            return tf.nn.sigmoid(d5_dense), d5_dense


    def generator(self, input):

        # we input a cardiac image and generate a mask so need to add the across the third dimension to get the a image shape output

        G=create_u_net(input, self.num_classes, self.keep_prob)
        #G1=tf.identity(G, name='segmentation_output') #using a tensor op to used while inference

        #g=tf.nn.softmax(G, name='segmentation_output2')

        return G


    def segment(self, input_image_filename):
        #print("Add the prediction logic here")
        #load the model
        #run the session in predict_op
        #may need to do batch inference (i.e. segment image slice-by-slice and time and then merge the segmentation)
        #save the segmented image as a nifti file


        #check if nifti filename or not
        print("Segment file name")


        print("Image to be segmented" + input_image_filename)
        image=self.get_image(input_image_filename)
        print(image.shape)
        [t, x, y, z] = image.shape;

        if (x == self.w and y == self.h and z == self.d):  # check the input and the output dim otherwise reshape the image and feed to the network (WIP)
                #sess = self.load_model(self.model_name2)

                with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as new_sess:

                    saver = tf.train.import_meta_graph((self.model_name2 + '.meta'))
                    #saver.restore(new_sess, self.model_dir)
                    saver.restore(new_sess, tf.train.latest_checkpoint("./"))

                    print("re store the session completed")

                    if (new_sess._closed):
                        print ("tensorflow session is closed not segmenting")

                    else:

                        segmented_image = new_sess.run([self.Gz], feed_dict={self.X_train: np.float32(image)})
                        return segmented_image

        else:
                print("Input is not of the same as training re-shape the data and feed to the network")



    def save_model(self, modelname):

        print ("Saving the model after training")
        if (os.path.exists(self.model_dir)):
            os.makedirs(self.model_dir)

        self.saver.save(self.sess, os.path.join(self.model_dir, self.model_name))
        print("Completed saving the model")



    def load_model(self, model_name):

        print ("Checking for the model")

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as new_sess:

            saver =tf.train.import_meta_graph((model_name + '.meta'))
            #saver.restore(self.sess, self.model_dir)
            saver.restore(new_sess,tf.train.latest_checkpoint("./"))
            print ("Session restored")
            return new_sess


    def get_image_filenames(self,datadir):

        nii_files = []

        for dirName, subdirList, fileList in os.walk(datadir):
            for filename in fileList:
                name = os.path.join(dirName, filename)

                not_segment_image=True
                if "segmentation" in filename.lower():
                    not_segment_image=False;

                if (not_segment_image): #only train the not segmented images

                   if ".nii" in filename.lower() and "sax" in filename:  # we only want the short axis images
                        nii_files.append(name)
                   else:
                        continue

        return nii_files

    def get_image(self,image_filename):

        #load image data from a nifti file-name and the compute the
        data = nib.load(image_filename)
        image = np.array(data.get_data())
        image=np.swapaxes(image,0,3);
        image=np.swapaxes(image,2,3);


        #normalise the intentisity between -1 and 1
        range=image.max()-image.min();
        image=2*((image-image.min())/range)-1;

        return image

    def transfer_learning(self, filenames_to_train, inputmodelname,outputmodel_name):


        if (self.load_model(inputmodelname)):

            for filename in filenames_to_train:
                image=self.get_image(filename)
                self.train_single(image)
                print ("Transfer learning complete")
                self.save_model(outputmodel_name)
        else:
            print("Model not found")



    def train_single(self, training_datadir):

     #get the file names
        filenames=self.get_image_filenames(training_datadir)

        with tf.device('/gpu:0'):
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as self.sess:
               self.train_writer = tf.summary.FileWriter(self.logdir , tf.get_default_graph())

               self.sess.run(self.init)

               counter=0
               learningrate=0.005

               for epoch in range(0,self.num_epochs):

                    if (epoch % 3==0):
                        learningrate=learningrate/3

                    random.shuffle(filenames)
                    Average_loss_G = 0
                    Average_loss_D = 0

                    for image_file in filenames[:400]:
                        training_images = self.get_image(image_file)

                        [t,x,y,z]=training_images.shape

                        if (x == self.w and y == self.h and z == self.d):  #check the input and the output dim otherwise reshape the image and feed to the network (WIP)
                            #print("Training image" + image_file)


                                    summary1, opt ,loss_D = self.sess.run([self.merged_summary,self.OptimizerD, self.d_loss], feed_dict={self.input_image: training_images,self.learning_rate: learningrate})


                                    opt,loss_G = self.sess.run([self.OptimizerG, self.g_loss], feed_dict={self.input_image: training_images, self.learning_rate: learningrate})

                                    summary2, opt, loss_G = self.sess.run([self.merged_summary, self.OptimizerG, self.g_loss], feed_dict={self.input_image: training_images, self.learning_rate: learningrate})

                                    counter += 1
                                    Average_loss_D=  (Average_loss_D+loss_D)/2
                                    Average_loss_G = (Average_loss_G + loss_G)/2
                                    self.train_writer.add_summary(summary1, counter)
                                    self.train_writer.add_summary(summary2)

                    print("Epoch: ", str(epoch) + " learning rate:" + str(learningrate) + " Generator loss: " + str(Average_loss_G) + "Discriminator loss: " + str(Average_loss_D))

                    if (epoch % 5 == 0):
                       #learningrate=learningrate/10
                       self.saver.save(self.sess, self.model_name)


               print("Training completed ")
               self.save_model(self.model_name)


    def print_shape(self,tensor):
        print(tensor.get_shape().as_list())


if __name__ == '__main__':


    cnet=cardiacnets(192,256,1, 'cardiac_segnet-v7')
    cnet.train_single(training_datadir='/home/jehill/Documents/ML_PROJECTS/VNETS/DATA/train/')
    #cnet.load_model(cnet.model_name)
    #cnet.load_model('cardiac-seg-net-single.meta')

    #file=cnet.get_image_filenames('/home/jehill/Documents/ML_PROJECTS/VNETS/DATA/train/');

    #img=cnet.segment(file[10])
