from __future__ import absolute_import, division, print_function
import argparse
import os
import numpy as np
import tensorflow as tf
import config as cfg
from network import Network
from blob import BlobLoader
from utils.anchors import get_anchors

slim = tf.contrib.slim

train_anno_dir = os.path.join(cfg.data_dir, 'annotation_val')

# add gpu/cpu options??
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-5)
args = parser.parse_args()

# tf configuration
tfcfg = tf.ConfigProto()
tfcfg.gpu_options.per_process_gpu_memory_fraction = 0.7
tfcfg.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

net = Network(session=tf.Session(config=tfcfg), is_training=True,
              lr=args.lr, adamop=True, pretrained=True)

# load anchors and data
print('loading anchors and dataset')
anchors = get_anchors(target_size=(cfg.inp_size, cfg.inp_size))
blob = BlobLoader(anno_dir=train_anno_dir, batch_size=args.batch)
num_iters = blob.num_anno // args.batch
step = 0

print('start training')
for epoch in range(1, args.epochs + 1):
    iter = 0

    for batch_images, batch_boxes, batch_classes, num_boxes_batch in blob.next_batch():
        iter += 1

        step, loss = net.train(batch_images, batch_boxes,
                               batch_classes, anchors, num_boxes_batch)

        if step % 100 == 0 or iter == num_iters:
            print('epoch: {0:03} - step: {1:06} - total_loss: {2}'
                  .format(epoch, step, loss))

    if epoch % 10 == 0 or epoch == args.epochs:
        net.save_ckpt(step)
print('training done')
