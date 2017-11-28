from __future__ import absolute_import, division, print_function
import tensorflow as tf
import config as cfg
from network import Network
from blob import prep_train_images_blob
from utils.anchors import get_anchors

slim = tf.contrib.slim

# jit/xla config
tfcfg = tf.ConfigProto()
# tfcfg.gpu_options.per_process_gpu_memory_fraction = 0.9
tfcfg.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

net = Network(session=tf.Session(config=tfcfg))  # also load checkpoint or init variables

# load anchors and data
anchors = get_anchors(target_size=(cfg.inp_size, cfg.inp_size))

blob_images, blob_boxes, blob_classes = prep_train_images_blob()  # scaled data to inp_size
num_images = len(blob_images)

print('loaded {} images from dataset'.format(num_images))

# training
for epoch in range(cfg.num_epochs):
	print('epoch: {}'.format(epoch + 1))

    for i in range(num_images):  # only 1 image per batch
        global_step, total_loss = net.train(
            image=blob_images[i], boxes=blob_boxes[i], classes=blob_classes[i], anchors=anchors)

        if global_step % 2000 == 0:  # print status each 2000 steps
        	print('step: {}, total loss: {:.3f}'.format(global_step, total_loss))

    net.save()
    print('saved checkpoint')
