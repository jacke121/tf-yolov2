from __future__ import absolute_import, division, print_function
import os
import numpy as np
import tensorflow as tf
import config as cfg
from network import Network
from blob import BlobLoader
from utils.anchors import get_anchors

slim = tf.contrib.slim

train_anno_dir = os.path.join(cfg.data_dir, 'annotation')
assert os.path.exists(train_anno_dir)

# load anchors and data
print('loading anchors and dataset')
anchors = get_anchors(target_size=(cfg.inp_size, cfg.inp_size))
blob = BlobLoader(anno_dir=train_anno_dir, batch_size=cfg.batch_size)

# jit/xla config
tfcfg = tf.ConfigProto()
tfcfg.gpu_options.per_process_gpu_memory_fraction = 0.85
tfcfg.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

# also load checkpoint or init variables
net = Network(session=tf.Session(config=tfcfg))

for epoch in range(cfg.num_epochs):
    for _ in range(blob.num_anno // cfg.batch_size + 1):
        batch_images, batch_boxes, batch_classes = blob.next_batch()
        step, loss = net.train(batch_images, batch_boxes,
                               batch_classes, anchors)

        # if step % 5000 == 0:
        #     print('step: {:06} - total loss: {:.6f}'.format(step, loss))

    # if epoch % 10 or epoch == cfg.num_epochs - 1:
    #     net.save()
