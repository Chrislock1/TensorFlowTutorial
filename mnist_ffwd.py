from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pyplt

from tensorflow.examples.tutorials.mnist import input_data

# Downloading dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_images = mnist.train.images
train_labels = mnist.train.labels

test_images = mnist.test.images
test_labels = mnist.test.labels

