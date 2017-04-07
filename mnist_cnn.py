from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pyplt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_images = mnist.train.images
train_labels = mnist.train.labels

train_set = np.array(train_images, dtype=np.float32)
num_samples = len(train_set)
train_set = train_set.reshape([num_samples, 28, 28])

# print(train_labels[6])
# pyplt.imshow(train_set[6], cmap='gray')
# pyplt.show()

