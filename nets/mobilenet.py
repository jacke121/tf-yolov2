from __future__ import absolute_import, division, print_function
import tensorflow as tf

slim = tf.contrib.slim


def depthsep_conv2d(inputs, num_outputs, kernel, stride, scope=None):
    net = slim.separable_conv2d(inputs, None, kernel,
                                depth_multiplier=1,
                                stride=stride,
                                scope=scope + '_depthwise')

    net = slim.conv2d(net, num_outputs, [1, 1],
                      stride=1,
                      scope=scope + '_pointwise')

    return net


def forward(inputs, num_outputs, scope=None):
    with tf.variable_scope(scope, 'MobilenetV1', [inputs], reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True)):

            net = slim.conv2d(inputs, 32, [3, 3],
                              stride=2, scope='Conv2d_0')

            net = depthsep_conv2d(net, 64, [3, 3],
                                  stride=1, scope='Conv2d_1')
            net = depthsep_conv2d(net, 128, [3, 3],
                                  stride=2, scope='Conv2d_2')
            net = depthsep_conv2d(net, 128, [3, 3],
                                  stride=1, scope='Conv2d_3')
            net = depthsep_conv2d(net, 256, [3, 3],
                                  stride=2, scope='Conv2d_4')
            net = depthsep_conv2d(net, 256, [3, 3],
                                  stride=1, scope='Conv2d_5')
            net = depthsep_conv2d(net, 512, [3, 3],
                                  stride=2, scope='Conv2d_6')
            net = depthsep_conv2d(net, 512, [3, 3],
                                  stride=1, scope='Conv2d_7')
            net = depthsep_conv2d(net, 512, [3, 3],
                                  stride=1, scope='Conv2d_8')
            net = depthsep_conv2d(net, 512, [3, 3],
                                  stride=1, scope='Conv2d_9')
            net = depthsep_conv2d(net, 512, [3, 3],
                                  stride=1, scope='Conv2d_10')
            net = depthsep_conv2d(net, 512, [3, 3],
                                  stride=1, scope='Conv2d_11')
            net = depthsep_conv2d(net, 1024, [3, 3],
                                  stride=2, scope='Conv2d_12')
            net = depthsep_conv2d(net, 1024, [3, 3],
                                  stride=1, scope='Conv2d_13')

            net = slim.conv2d(net, num_outputs, [1, 1],
                              activation_fn=None, normalizer_fn=None, scope='logits')

    return net
