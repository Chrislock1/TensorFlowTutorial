from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pyplt

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def run():
    # Downloading dataset
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, fake_data=FLAGS.fake_data)

    train_images = mnist.train.images
    train_labels = mnist.train.labels

    test_images = mnist.test.images
    test_labels = mnist.test.labels

    sess = tf.InteractiveSession()

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='input_image')
        y_ = tf.placeholder(tf.float32, [None, 10], name='target_output')

    with tf.name_scope('images'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input_images', image_shaped_input, 10)

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial_value=initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial_value=initial)

    def variable_summaries(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def dense_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases)
            with tf.name_scope('fwd_output'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    y = dense_layer(x, 784, 10, 'layer1', act=tf.nn.sigmoid)

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.log_dir + '/model.ckpt')

    def feed_dict(run):
        if run or FLAGS.fake_data:
            xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
        else:
            xs, ys = mnist.test.images, mnist.test.labels

        return {x: xs, y_: ys}

    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    print('Accuracy: %s' % acc)


def main(_):
    run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False, help='If true, uses fake data for unit testing')
    parser.add_argument('--max_steps', type=int, default=5000, help='Number of steps to run trainer')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--data_dir', type=str, default='C:/Users/fredr/PycharmProjects/TensorFlowTutorial/MNIST_data',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='C:/Users/fredr/PycharmProjects/TensorFlowTutorial/log/mnist_with_summaries',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
