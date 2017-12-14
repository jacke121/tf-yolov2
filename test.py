from __future__ import absolute_import, division, print_function
import tensorflow as tf
from network import Network

slim = tf.contrib.slim

tfcfg = tf.ConfigProto()
tfcfg.gpu_options.per_process_gpu_memory_fraction = 0.8
tfcfg.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

net = Network(session=tf.Session(config=tfcfg), is_training=False)
