from __future__ import absolute_import, division, print_function
import argparse
import os
import numpy as np
import tensorflow as tf
import config as cfg
from network import Network
from blob import BlobLoader
from utils.anchors import get_anchors
from py_compute_targets import compute_targets_batch

slim = tf.contrib.slim

train_anno_dir = os.path.join(cfg.data_dir, 'annotation')

# add gpu/cpu options??
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()

# tf configuration
tfcfg = tf.ConfigProto()
tfcfg.gpu_options.per_process_gpu_memory_fraction = 0.9
tfcfg.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

# losses
num_boxes_ph = tf.placeholder(tf.float32)

# focal loss?
cls_label_ph = tf.placeholder(tf.float32)
cls_pred_ph = tf.placeholder(tf.float32)
cls_loss = tf.losses.mean_squared_error(cls_label_ph, cls_pred_ph)

iou_label_ph = tf.placeholder(tf.float32)
iou_pred_ph = tf.placeholder(tf.float32)
iou_loss = tf.losses.mean_squared_error(iou_label_ph, iou_pred_ph)

bbox_label_ph = tf.placeholder(tf.float32)
bbox_pred_ph = tf.placeholder(tf.float32)
bbox_loss = tf.losses.mean_squared_error(bbox_label_ph, bbox_pred_ph)

total_loss = (cls_loss + iou_loss + bbox_loss) / num_boxes_ph

global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
optimizer = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(
    loss=total_loss, global_step=global_step)

net = Network(session=tf.Session(config=tfcfg))
net.load_ckpt(pretrained=True)

# load anchors and data
print('loading anchors and dataset')
anchors = get_anchors(target_size=(cfg.inp_size, cfg.inp_size))
blob = BlobLoader(anno_dir=train_anno_dir, batch_size=args.batch)
print('done')

for epoch in range(1, args.epochs + 1):
    for _ in range(blob.num_anno // args.batch + 1):
        batch_images, batch_boxes, batch_classes = blob.next_batch()

        num_boxes = sum([boxes.shape[0] for boxes in batch_boxes])

        bbox_pred, iou_pred, cls_pred = net.run(images=batch_images)

        _cls, _cls_mask, _iou, _iou_mask, _bbox, _bbox_mask = compute_targets_batch(net.out_h, net.out_w,
                                                                                    bbox_pred, iou_pred,
                                                                                    batch_boxes, batch_classes, anchors)

        step, loss, _ = net.sess.run([global_step, total_loss, optimizer],
                                     feed_dict={num_boxes_ph: num_boxes,
                                                cls_label_ph: _cls * _cls_mask,
                                                cls_pred_ph: cls_pred * _cls_mask,
                                                iou_label_ph: _iou * _iou_mask,
                                                iou_pred_ph: iou_pred * _iou_mask,
                                                bbox_label_ph: _bbox * _bbox_mask,
                                                bbox_pred_ph: bbox_pred * _bbox_mask})

        if step % 5000 == 0:
            print('step: {0:06} - total loss: {1:.6f}'.format(step, loss))

    if epoch % 10 == 0 or epoch == args.epochs:
        net.saver.save(net.sess,
                       save_path=os.path.join(cfg.ckpt_dir, cfg.model),
                       global_step=global_step)
        print('saved checkpoint\n')
