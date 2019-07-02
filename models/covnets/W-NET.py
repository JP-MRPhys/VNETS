#Implementation of W-NET (encoder and decorder) for unsupervised image segmention
#Data must 3D (spatial) + number of channels
#deconvlution3D is upsampling


from utils.Layers import convolution, max_pooling
from utils.Layers import  upsampling
import tensorflow as tf
import os
import nibabel as nib
import numpy as np
import tensorflow.contrib.slim as slim
import random


class WNET:
    def __init__(self, x,y,z):
        # network parameters
         self.learning_rate = 0.001
         self.num_epochs = 50000
         self.display_step = 20

         self.global_step = 0
         self.w=x
         self.h=y
         self.d=z
         self.X_train = tf.placeholder(tf.float32, [None,self.w, self.h, self.d], name='X_train')
         self.batch_size=30;
         self.num_classes=5

         #now create the network
         self.keep_prob=0.30  #that the drop
         self.drop_out=self.keep_prob
         self.logits = self.create_encorder()
         print("Completed creating encorder graph")
         self.decorder_output=self.create_decoder(self.logits)
         print("Completed creating decoder graph")

         self.print_shape(self.decorder_output)
         self.loss=tf.Variable(0.0, tf.float32)


         with tf.name_scope("loss"):
             y_true = tf.reshape(self.X_train, [-1, self.w * self.h * self.d])
             for i in range(0,self.num_classes):

                 y_pred = tf.reshape(self.decorder_output[-1, :,:,i], [-1, self.w*self.h*self.d])
                 label_loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
                 self.loss += label_loss;

             self.loss=self.loss/self.num_classes;

         tf.summary.scalar("SEE loss", self.loss)

         with tf.name_scope("Optimization"):
             optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
             self.train_op = optimizer.minimize(loss=self.loss)

         self.predict_op= y_pred


         self.merged_summary = tf.summary.merge_all()
         self.init = tf.global_variables_initializer()
         self.saver = tf.train.Saver()
         self.logdir='./logdir/' #if not exist create logdir




    def create_encorder(self):

        #create the encorder side


        input=tf.nn.lrn(self.X_train)   #normalisation step

        conv1_1=convolution(input,[3,3,1,64],[1,1,1,1],'SAME', self.keep_prob, 'conv1_1')
        conv1_2=convolution(conv1_1,[3,3,64,64], [1,1,1,1],'SAME', self.keep_prob, name='conv1_2')
        max1=max_pooling(conv1_2, [1,2,2,1], [1,2,2,1], name='max1')

        conv2_1=convolution(max1, [3,3,64,128], [1,1,1,1],'SAME', self.keep_prob, name='conv2_1') 
        conv2_2=convolution(conv2_1,[3,3,128,128], [1,1,1,1],'SAME', self.keep_prob, name='conv2_2')
        max2=max_pooling(conv2_2, [1,2,2,1], [1, 2, 2,1], name='max2')

        conv3_1=convolution(max2, [3,3,128,256], [1,1,1,1],'SAME', self.keep_prob, name='conv3_1') 
        conv3_2=convolution(conv3_1,[3,3,256,256], [1,1,1,1],'SAME', self.keep_prob, name='conv3_2') 
        max3=max_pooling(conv3_2, [1,2,2,1], [1, 2, 2,1], name='max3')

        conv4_1=convolution(max3, [3,3,256,512], [1,1,1,1],'SAME', self.keep_prob, name='conv4_1') 
        conv4_2=convolution(conv4_1,[3,3,512,512], [1,1,1,1],'SAME', self.keep_prob, name='conv4_2') 
        max4=max_pooling(conv4_2, [1, 2, 2, 1], [1, 2,2,1], name='max4')

        conv5_1=convolution(max4, [3,3,512,1024],  [1,1,1,1],'SAME', self.keep_prob, name='conv5_1') 
        conv5_2=convolution(conv5_1,[3,3,1024,1024],  [1,1,1,1],'SAME', self.keep_prob, name='conv5_2')

        up_6 = upsampling(conv5_2, tf.shape(conv4_2), 512, 1024,2, name='up6') 
        concat_6 = tf.concat([up_6, conv4_2], axis=3) 
        conv6_1 = convolution(concat_6, [3,3,1024,512], [1,1,1,1], 'SAME', self.keep_prob, name='conv6_1')
        conv6_2 = convolution(conv6_1, [3,3,512,512], [1,1,1,1], 'SAME', self.keep_prob, name='conv6_2')


        up_7 = upsampling(conv6_2, tf.shape(conv3_2), 256, 512, 2, name='up7') 
        concat_7 = tf.concat([up_7, conv3_2], axis=3) 
        conv7_1 = convolution(concat_7, [3,3,512,256], [1,1,1,1], 'SAME', self.keep_prob, name='conv7_1')
        conv7_2 = convolution(conv7_1, [3,3,256,256], [1,1,1,1], 'SAME', self.keep_prob, name='conv7_2')


        up_8 = upsampling(conv7_2, tf.shape(conv2_2), 128, 256, 2, name='up8')
        concat_8 = tf.concat([up_8, conv2_2], axis=3) 
        conv8_1 = convolution(concat_8, [3,3,256,128], [1,1,1,1], 'SAME', self.keep_prob, name='conv8_1')
        conv8_2 = convolution(conv8_1, [3,3,128,128], [1,1,1,1], 'SAME', self.keep_prob, name='conv8_2')


        up_9 = upsampling(conv8_2, tf.shape(conv1_2), 64, 128, 2,  name='up9')
        concat_9 = tf.concat([up_9, conv1_2], axis=3)

        conv9_1 = convolution(concat_9, [3,3,128,64], [1,1,1,1], 'SAME', self.keep_prob, name='conv9_1')
        conv9_2 = convolution(conv9_1, [3,3,64, 64], [1,1,1,1], 'SAME', self.keep_prob, name='conv9_2')

        conv_10 = convolution(conv9_2,[1,1,64,1], [1,1,1,1], 'SAME', self.keep_prob, name='conv10')

        print("Down-sampling")
        self.print_shape(conv1_2)
        self.print_shape(max1)
        self.print_shape(conv2_1)
        self.print_shape(max2)
        self.print_shape(conv3_2)
        self.print_shape(max3)
        self.print_shape(conv4_1)
        self.print_shape(max4)
        self.print_shape(conv5_2)

        print("Upsampling")
        self.print_shape(concat_6)
        self.print_shape(conv6_2)
        self.print_shape(concat_7)
        self.print_shape(conv7_2)
        self.print_shape(concat_8)
        self.print_shape(conv8_2)
        self.print_shape(concat_9)
        self.print_shape(conv9_2)
        self.print_shape(conv_10)

        logits=tf.nn.softmax(conv_10)  # get logits here

        return logits

    def create_decoder(self, input_tensor):
            # create the decorder side 

            conv11_1 = convolution(input_tensor, [3,3, 1, 64], [1, 1, 1,1], 'SAME', self.keep_prob, name='conv11_1')
            max11 = max_pooling(conv11_1, [1, 2, 2, 1], [1, 2, 2, 1], name='max11')

            conv12_1 = convolution(max11, [3,3, 64,128], [1, 1, 1,1], 'SAME', self.keep_prob, name='conv12_1')
            conv12_2 = convolution(conv12_1, [3, 3,128, 128], [1, 1, 1,1], 'SAME', self.keep_prob, name='conv12_2')
            max12 = max_pooling(conv12_2, [1, 2, 2, 1], [1, 2, 2, 1], name='max12')

            conv13_1 = convolution(max12, [3, 3,128, 256], [1, 1, 1,1], 'SAME', self.keep_prob, name='conv13_1')
            conv13_2 = convolution(conv13_1, [3, 3,256, 256], [1, 1, 1,1], 'SAME', self.keep_prob, name='conv13_2')
            max13 = max_pooling(conv13_2, [1, 2, 2, 1], [1, 2, 2, 1], name='max13')

            conv14_1 = convolution(max13, [3, 3,256, 512], [1, 1, 1,1], 'SAME', self.keep_prob, name='conv14_1')
            conv14_2 = convolution(conv14_1, [3, 3,512, 512], [1, 1, 1,1], 'SAME', self.keep_prob, name='conv14_2')
            max14 = max_pooling(conv14_2, [1, 2, 2, 1], [1, 2, 2, 1], name='max14')

            conv15_1 = convolution(max14, [3,3, 512, 1024], [1,1,1,1], 'SAME', self.keep_prob, name='conv15_1')
            conv15_2 = convolution(conv15_1, [3, 3, 1024, 1024], [1,1,1,1], 'SAME', self.keep_prob, name='conv15_2')

            up_16 = upsampling(conv15_2, tf.shape(conv14_2), 512, 1024, 2, name='up16') 
            concat_16 = tf.concat([up_16, conv14_2], axis=3)
            conv16_1 = convolution(concat_16, [3, 3,1024, 512], [1,1,1,1], 'SAME', self.keep_prob, name='conv16_1')
            conv16_2 = convolution(conv16_1, [3,3, 512, 512], [1,1,1,1], 'SAME', self.keep_prob, name='conv16_2')

            up_17 = upsampling(conv16_2, tf.shape(conv13_2), 256, 512, 2, name='up17') 
            concat_17 = tf.concat([up_17, conv13_2], axis=3)
            conv17_1 = convolution(concat_17, [3, 3,512, 256], [1,1,1,1], 'SAME', self.keep_prob, name='conv17_1')
            conv17_2 = convolution(conv17_1, [3, 3,256, 256], [1,1,1,1], 'SAME', self.keep_prob,  name='conv17_2')

            up_18 = upsampling(conv17_2, tf.shape(conv12_2), 128, 256, 2, name='up18') 
            concat_18 = tf.concat([up_18, conv12_2], axis=3)
            conv18_1 = convolution(concat_18, [3, 3, 256, 128], [1, 1, 1, 1], 'SAME', self.keep_prob,name='conv18_1')
            conv18_2 = convolution(conv18_1, [3, 3,128, 128], [1, 1, 1, 1], 'SAME', self.keep_prob, name='conv18_2')

            up_19 = upsampling(conv18_2, tf.shape(conv11_1), 64, 128, 2, name='up19')
            concat_19 = tf.concat([up_19, conv11_1], axis=3)

            conv19_1 = convolution(concat_19, [3, 3,128, 64], [1, 1, 1, 1], 'SAME', self.keep_prob, name='conv19_1') 
            conv19_2 = convolution(conv19_1, [3, 3,64, 64], [1, 1, 1, 1], 'SAME', self.keep_prob, name='conv19_2') 
            conv20 = convolution(conv19_2, [1, 1,  64, self.num_classes], [1,1,1,1], 'SAME', self.keep_prob, name='conv20')

            logits=tf.nn.softmax(conv20)


            return logits


    def train(self, data_iterator):

        with tf.Session() as sess:
            sess.run(self.init)
            train_writer = tf.summary.FileWriter(self.logdir, graph=tf.get_default_graph())

            # if exists(logdir):
            #     saver.restore(sess, logdir)

            iterator = data_iterator 

            for i in range(self.num_epoch + 1):
                next_items = iterator.get_next()
                batch_x = self.sess.run(next_items)
                _ = sess.run(self.train_op, feed_dict={self.X_train: batch_x})

                if i % self.display_step == 0:
                    loss_, summary = self.sess.run([self.loss, self.merged_summary], feed_dict={self.X_train: batch_x})
                    print("Iteration number: ", str(i), " Loss: ", str(loss_))
                    train_writer.add_summary(summary)
                    #self.saver.save(sess, self.logdir, global_step=self.global_step + i)
        #self.sess=sess 


    def train_manual(self, training_datadir):

     #get the file names
        filenames=self.get_image_filenames(training_datadir)

        with tf.device('/gpu:0'):
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
               train_writer = tf.summary.FileWriter(self.logdir + '/train',sess.graph)
               sess.run(self.init)
               for epoch in range(0,self.num_epochs):

                    random.shuffle(filenames)
                    average_loss=0
                    for image_file in filenames[:100]:
                        next_items = self.nifti_to_numpy(image_file)
                        [t, x,y,z]=next_items.shape

                        if (x == self.w and y == self.h and z == self.d):  #check the input and the output dim otherwise reshape the image and feed to the network (WIP)

                            #print("Training image" + image_file)

                            for i in range(0,t,self.batch_size):
                                    data=next_items[i:i+self.batch_size,:,:,:] #we need to feed a feed of the images of each time series in a batched way
                                    loss, summary = sess.run([self.loss, self.merged_summary], feed_dict={self.X_train: data})
                                    average_loss=(average_loss+loss)/2
                    print("Epoch: ", str(epoch), "Average Loss for this epoch: ", str(average_loss))

               if i % self.display_step == 0:
                     print("Epoch: ", str(epoch), " Loss: ", str(loss))
                     train_writer.add_summary(summary)

                     # save the model every 200 epoch's
               if (epoch % 200 == 0):
                    self.saver.save(sess,'W-NET')



    def segment(self, input_image_filename):
        #print("Add the prediction logic here")
        #load the model
        #run the session in predict_op
        #may need to do batch inference (i.e. segment image slice-by-slice and time and then merge the segmentation)

        image=self.nifti_to_numpy(input_image_filename)
        [t, x, y, z] = image.shape;

        if (x == self.w and y == self.h and z == self.d):  # check the input and the output dim otherwise reshape the image and feed to the network (WIP)

            print("Image to be segmented" + input_image_filename)
            segmented_image = self.sess.run([self.predict], feed_dict={self.X_train: image})
            return segmented_image

    def get_image_filenames(self,datadir):

        nii_files = [] 

        for dirName, subdirList, fileList in os.walk(datadir):
            for filename in fileList:
                name = os.path.join(dirName, filename)
                if ".nii" in filename.lower() and "sax" in filename:  # we only want the short axis images
                    nii_files.append(name)
                else:
                    continue

        return nii_files 

    def nifti_to_numpy(self,image_filename):

        #load image data from a nifti file-name and the compute the

        data = nib.load(image_filename)
        image = np.array(data.get_data())
        image=np.swapaxes(image,0,3);
        image=np.swapaxes(image,2,3);



        #normalise the intentisity between -1 and 1
        range=image.max()-image.min();



        image=2*((image-image.min())/range)-1;

        print('max' + str(image.max()) + 'min:' + str(image.min()))

        #print(image.shape)

        return image

    def print_shape(self,tensor):
        print(tensor.get_shape().as_list())


if __name__ == '__main__':

    network=WNET(192,256,1)
    network.train_manual(training_datadir='/home/jehill/Documents/ML_PROJECTS/VNETS/DATA/train/')




