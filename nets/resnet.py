from __future__ import absolute_import, division, print_function
import tensorflow as tf
from resnet_utils import resnet_block

slim = tf.contrib.slim


def res50(inputs, num_outputs, scope=None):
    # resnet forwarding
    with tf.variable_scope(scope, 'res50', [inputs], reuse=tf.AUTO_REUSE):
        net = slim.conv2d(inputs, 64, [7, 7], stride=2,
                          activation_fn=None, normalizer_fn=None, scope='conv1')
        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

        # num_outputs = base_depth * 4 each resnet block
        net = resnet_block(net, base_depth=64, num_units=3,
                           stride=1, scope='block1')
        net = resnet_block(net, base_depth=128, num_units=4,
                           stride=2, scope='block2')
        net = resnet_block(net, base_depth=256, num_units=6,
                           stride=2, scope='block3')
        net = resnet_block(net, base_depth=512, num_units=3,
                           stride=2, scope='block4')

        net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')

        # reduce features to prediction
        net = slim.conv2d(net, num_outputs, [1, 1], stride=1,
                          activation_fn=None, normalizer_fn=None, scope='logits')

    return net
