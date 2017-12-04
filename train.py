from __future__ import absolute_import, division, print_function
import tensorflow as tf
import config as cfg
from network import Network
from blob import prep_test_images_blob
from utils.anchors import get_anchors

slim = tf.contrib.slim

# jit/xla config
tfcfg = tf.ConfigProto()
tfcfg.gpu_options.per_process_gpu_memory_fraction = 0.85
tfcfg.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

# also load checkpoint or init variables
net = Network(session=tf.Session(config=tfcfg))

# load anchors and data
print('loading anchors and dataset')
anchors = get_anchors(target_size=(cfg.inp_size, cfg.inp_size))
