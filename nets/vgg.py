# https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py
from __future__ import absolute_import, division, print_function
import tensorflow as tf

slim = tf.contrib.slim


class Vgg16:
    def __init__(self):
        pass

    def forward(inputs, num_outputs, scope=None):
        with tf.variable_scope(scope, 'net', [inputs], reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                weights_initializer=tf.truncated_normal_initializer(
                                    stddev=0.01),
                                weights_regularizer=slim.l2_regularizer(1e-4),
                                biases_initializer=tf.zeros_initializer()):
                net = slim.repeat(inputs, 2, slim.conv2d,
                                  64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')

                net = slim.repeat(net, 2, slim.conv2d, 128,
                                  [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')

                net = slim.repeat(net, 2, slim.conv2d, 256,
                                  [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')

                net = slim.repeat(net, 2, slim.conv2d, 512,
                                  [3, 3], scope='conv4')
                sc = slim.max_pool2d(net, [1, 1], stride=2, scope='sc4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')

                net = slim.repeat(net, 2, slim.conv2d, 512,
                                  [3, 3], scope='conv5')
                net = net + sc
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                net = slim.conv2d(net, 512, [3, 3], scope='conv6')
                net = slim.conv2d(net, num_outputs, [1, 1], stride=1,
                                  activation_fn=None, normalizer_fn=None, scope='logits')

        return net

    def restore(self):
        pass
