import os
import tensorflow as tf
from utils.fileIO import *


def save_model(saver, sess, model_name, model_dir):
    print("Saving the model after training")
    if (os.path.exists(model_dir)):
        os.makedirs(model_dir)
    saver.save(sess, os.path.join(model_dir, model_name))
    print("Completed saving the model")


def load_model(model_name):
    print("Checking for the model")

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as new_sess:
        saver = tf.train.import_meta_graph((model_name + '.meta'))
        # saver.restore(self.sess, self.model_dir)
        saver.restore(new_sess, tf.train.latest_checkpoint("./"))
        print("Session restored")
        return new_sess


def transfer_learning(filenames_to_train, inputmodelname, outputmodel_name):
    
    if (load_model(inputmodelname)):

        print("model found and session restored: WIP in progress below function's may not work in stable manner")

        for filename in filenames_to_train:
            image = get_image(filename)
            # load weights
            # train_single(image)
            print("Transfer learning complete")
            save_model(outputmodel_name)
    else:
        print("Model not found")