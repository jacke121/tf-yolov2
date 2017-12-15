from __future__ import absolute_import, division, print_function
import os
import time
from datetime import timedelta
import numpy as np
import cv2
import tensorflow as tf
import config as cfg
from network import Network
from postprocess import postprocess, draw_targets
from utils.anchors import get_anchors

slim = tf.contrib.slim

tfcfg = tf.ConfigProto()
tfcfg.gpu_options.allow_growth = True
tfcfg.gpu_options.per_process_gpu_memory_fraction = 0.6
tfcfg.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

net = Network(session=tf.Session(config=tfcfg), is_training=False)

image = cv2.imread('./test/img00001.jpg')
scaled_image = cv2.resize(image, (cfg.inp_size, cfg.inp_size))
anchors = get_anchors(target_size=(cfg.inp_size, cfg.inp_size))

start_t = time.time()

box_pred, iou_pred, cls_pred = net.predict(scaled_image[np.newaxis], anchors)

box_pred, cls_inds, scores = postprocess(box_pred[0], iou_pred[0], cls_pred[0],
                                         image.shape[0:2], thresh=0.5, force_cpu=False)

image = draw_targets(image, box_pred, cls_inds, scores)

print('time: ' + str(timedelta(seconds=time.time() - start_t)))

cv2.imshow('testing', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
