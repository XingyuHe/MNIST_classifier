import numpy as np
import pickle 
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy
# from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *


dog_model_run_1 = open("dog_model_run_1", 'r')
parameters = pickle.load(dog_model_run_1)

num_px = 150
print(num_px)

def image_preprocess():
    from skimage.transform import resize
    # We preprocess the image to fit your algorithm.
    fname = "test_images/test12.jpg"
    image = np.array(plt.imread(fname))
    print(image.shape, "original image shape")
    plt.imshow(image)
    
    my_image_show = resize(image, (num_px, num_px, 3))
    print(my_image_show.shape, "shrink image shape")
    plt.imshow(my_image_show)


    my_image_flatten = my_image_show.reshape((1, num_px * num_px * 3)).T
    print(my_image_flatten.shape, "processed image shape")
    my_image_input = my_image_flatten/255.

    # plt.show()

    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
    my_image = my_image/255.

    return my_image

# my_image_input = image_preprocess()
# p, probas = predict(my_image_input, 0, parameters)
# print(probas)
# plt.show()

f = h5py.File('dog_datasets.hdf5', 'r')
test_set_x = f['test_dataset_x'][:]
test_set_y = f['test_dataset_y'][:]
# train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T   
# test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
# train_x = train_x_flatten/255.
# test_x = test_x_flatten/255.

p, probas = predict(test_set_x.T, test_set_y.T, parameters)

