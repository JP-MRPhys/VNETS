
import os
import nibabel as  nib
import numpy as np

"""
Utility functions to read in the training data
"""


def get_image_filenames(datadir):
    nii_files = []

    for dirName, subdirList, fileList in os.walk(datadir):
        for filename in fileList:
            name = os.path.join(dirName, filename)

            not_segment_image = True
            if "segmentation" in filename.lower():
                not_segment_image = False;

            if (not_segment_image):  # only train the not segmented images

                if ".nii" in filename.lower() and "sax" in filename:  # we only want the short axis images
                    nii_files.append(name)
                else:
                    continue

    return nii_files


def get_image(image_filename):
    # load image data from a nifti file-name and the compute the
    data = nib.load(image_filename)
    image = np.array(data.get_data())
    image = np.swapaxes(image, 0, 2)  # for task2 we flip the image in 0 and 2 dimension
    shape = image.shape;
    image = image.reshape(shape[0], shape[1], shape[2], self.d)
    # image = np.swapaxes(image, 2, 3);

    print(str(image.max()))

    # normalise the intentisity between -1 and 1
    range = image.max() - image.min();
    image = 2 * ((image - image.min()) / range) - 1;

    return image


def get_label(label_filename):
    # load image data from a nifti file-name and the compute the
    data = nib.load(label_filename)
    image = np.array(data.get_data())
    image = np.swapaxes(image, 0, 2)  # for task2 we flip the image in 0 and 2 dimension
    image = image.reshape(image.shape[0], image.shape[1], image.shape[2], self.d)

    print(str(image.max()))

    # normalise the intentisity between -1 and 1
    # range = image.max() - image.min();
    ##image = 2 * ((image - image.min()) / range) - 1;

    return image


def get_image_filenames_task2(self, datadir):
    nii_files = []

    for dirName, subdirList, fileList in os.walk(datadir):
        for filename in fileList:
            # name = os.path.join(dirName, filename)
            name = filename
            # print(name)

            not_segment_image = True
            if "segmentation" in filename.lower():
                not_segment_image = False;

            if "._la" in filename.lower():
                not_segment_image = False;

            if "._lun" in filename.lower():
                not_segment_image = False;

            if "._pan" in filename.lower():
                not_segment_image = False;

            if "._li" in filename.lower():
                not_segment_image = False;

            if "._lu" in filename.lower():
                not_segment_image = False;

            if "._pro" in filename.lower():
                not_segment_image = False;

            if (not_segment_image):  # only train the not segmented images

                if ".nii.gz" in filename.lower() in filename:  # we only want the short axis images
                    nii_files.append(name)
                    print(name)

                else:
                    continue

    return nii_files


def get_training_filenames(self):
    nii_files = []
    labels_files = []
    for i in range(0, len(self.training_dir)):
        print(self.training_dir[i])

        for dirName, subdirList, fileList in os.walk(self.training_dir[i]):
            for filename in fileList:
                # name = os.path.join(dirName, filename)
                name = filename
                # print(name)

                not_segment_image = True
                if "segmentation" in filename.lower():
                    not_segment_image = False;

                if "._la" in filename.lower():
                    not_segment_image = False;

                if "._li" in filename.lower():
                    not_segment_image = False;

                if "._pro" in filename.lower():
                    not_segment_image = False;

                if "._lu" in filename.lower():
                    not_segment_image = False;

                if "._pan" in filename.lower():
                    not_segment_image = False;

                if (not_segment_image):  # only train the not segmented images/correct name images

                    if ".nii.gz" in filename.lower() in filename:  # we only want the short axis images
                        nii_files.append(os.path.join(self.training_dir[i], name))
                        labels_files.append(os.path.join(self.labels_dir[i], name))
                        print(name)
                    else:
                        continue

    return nii_files, labels_files


def get_training_images(images_name, labels_name):
    # images_name = (os.path.join(self.traindir, filename))
    # labels_name = (os.path.join(self.labeldir, filename))

    train_image = get_image(images_name)
    label_image = get_label(labels_name)

    # print(filename)
    # print(train_image.shape)
    # print(label_image.shape)

    return train_image, label_image


def get_training_set(self, i, filename):
    images_name = (os.path.join(self.training_dir[i], filename))
    labels_name = (os.path.join(self.labels_dir[i], filename))

    train_image = get_image(images_name)
    label_image = get_label(labels_name)
    """"
    print(filename)
    print(train_image.shape)
    print(label_image.shape)
    """

    return train_image, label_image