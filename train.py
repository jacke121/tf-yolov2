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

# ie. Annotations and JPEGImages in pascal/voc
train_anno_dir = os.path.join(cfg.data_dir, 'annotation')
train_images_dir = os.path.join(cfg.data_dir, 'images')

# add gpu/cpu options??
# 2*batch_size images per batch with left-right flipping
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)  # num training epochs
parser.add_argument('--batch', type=int, default=4)  # num images per batch
parser.add_argument('--lr', type=float, default=1e-3)  # learning rate
args = parser.parse_args()

# tf configuration
tfcfg = tf.ConfigProto()
tfcfg.gpu_options.allow_growth = True
tfcfg.gpu_options.per_process_gpu_memory_fraction = 0.6
tfcfg.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

net = Network(session=tf.Session(config=tfcfg), is_training=True,
              lr=args.lr, adamop=False, pretrained=True)

# load anchors and data
print('loading anchors and dataset')
anchors = get_anchors(target_size=(cfg.inp_size, cfg.inp_size))
blob = BlobLoader(anno_dir=train_anno_dir,
                  images_dir=train_images_dir, batch_size=args.batch)
num_iters = blob.num_anno // args.batch
step = 0

# losses collection contain dict of losses from network
# {'step', 'bbox_loss', 'iou_loss', 'cls_loss', 'total_loss'}
losses_collection = []

print('start training')
for epoch in range(1, args.epochs + 1):
    iter = 0

    # double images per batch with left-right flipping
    for batch_images, batch_boxes, batch_classes, num_boxes_batch in blob.next_batch():
        iter += 1

        step, bbox_loss, iou_loss, cls_loss = net.train(batch_images, batch_boxes,
                                                        batch_classes, anchors, num_boxes_batch)

        if step % 100 == 0 or iter == num_iters:
            total_loss = bbox_loss + iou_loss + cls_loss

            # add to collection
            losses_collection.append({'step': step, 'bbox_loss': bbox_loss,
                                      'iou_loss': iou_loss, 'cls_loss': cls_loss, 'total_loss': total_loss})

            print('epoch: {0:03} - step: {1:06} - bbox_loss: {2} - iou_loss: {3} - cls_loss: {4}'
                  .format(epoch, step, bbox_loss, iou_loss, cls_loss))

    if epoch % 10 == 0 or epoch == args.epochs:
        net.save_ckpt(step)
print('training done')

# dumpt losses_collection to file
import json
with open('./logs/losses_collection.json', 'w') as fout:
    json.dump(losses_collection, fout)
