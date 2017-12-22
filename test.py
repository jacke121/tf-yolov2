from __future__ import absolute_import, division, print_function
import os
import time
from datetime import timedelta
import numpy as np
import cv2
import tensorflow as tf
import config as cfg
from network import Network
from py_postprocess import postprocess, draw_targets

slim = tf.contrib.slim

tfcfg = tf.ConfigProto()
tfcfg.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

net = Network(session=tf.Session(config=tfcfg), is_training=False)

image_name = '01.jpg'
image = cv2.imread(os.path.join(cfg.workspace, 'test', image_name))

tsize = 416
scaled_image = cv2.cvtColor(
    image, cv2.COLOR_BGR2RGB) - [123.68, 116.78, 103.94]
scaled_image = cv2.resize(scaled_image, (tsize, tsize))
anchors = cfg.anchors * tsize

start_t = time.time()

box_pred, iou_pred, cls_pred = net.predict(scaled_image[np.newaxis], anchors)

box_pred, cls_inds, scores = postprocess(box_pred[0], iou_pred[0], cls_pred[0],
                                         image.shape[0:2], thresh=0.5, force_cpu=False)

image = draw_targets(image, box_pred, cls_inds, scores)

print('usage time: ' + str(timedelta(seconds=np.round(time.time() - start_t))))

cv2.imshow(image_name, image)
cv2.waitKey(0)
cv2.destroyAllWindows()
