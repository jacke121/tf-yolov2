from __future__ import absolute_import, division, print_function
import tensorflow as tf
import config as cfg
from network import Network
from blob import prep_test_images_blob
from utils.anchors import get_anchors

slim = tf.contrib.slim

# jit/xla config
tfcfg = tf.ConfigProto()
tfcfg.gpu_options.per_process_gpu_memory_fraction = 0.8
tfcfg.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

# also load checkpoint or init variables
net = Network(session=tf.Session(config=tfcfg))

# load anchors and data
print('loading anchors and dataset')
anchors = get_anchors(target_size=(cfg.inp_size, cfg.inp_size))

# scaled, normalized data
blob_images, blob_boxes, blob_classes = prep_test_images_blob()
num_images = len(blob_images)

print('loaded {} images from dataset'.format(num_images))

for epoch in range(cfg.num_epochs):
    print('epoch: {}'.format(epoch))

    for i in range(num_images):
        step, loss = net.train(
            image=blob_images[i], boxes=blob_boxes[i], classes=blob_classes[i], anchors=anchors)

        if step % 200 == 0:
            print('step {} - total loss {}'.format(step, loss))

    # net.save()
    # print('saved checkpoint')
