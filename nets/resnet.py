# https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from resnet_utils import resnet_block

slim = tf.contrib.slim


class Res50:
    def __init__(self):
        pass

    def forward(inputs, num_outputs, scope=None):
        pass

    def restore(self):
        pass
