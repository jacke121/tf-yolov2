# tf-yolo2 configurations
from __future__ import absolute_import, division, print_function
import os
import numpy as np

# yolo model name
model = 'detrac'

# working directories
data_dir = '/home/dat/data/detrac'

yolo_dir = os.path.dirname(os.path.abspath(__file__))
ckpt_dir = os.path.join(yolo_dir, 'ckpt')

save_path = os.path.join(ckpt_dir, model)

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# target image size for blob
# target_size = 416
# max_size = 480

inp_size = 416
num_anchors = 5

# object labels and class colors
label_names = ['others', 'car', 'bus', 'van']
num_classes = len(label_names)
label_colors = {}
for label in label_names:
    label_colors[label] = (np.random.randint(
        0, 128), np.random.randint(0, 128), np.random.randint(0, 128))

# yolo configuration
iou_thresh = 0.7
cls_scale = 1
object_scale = 5
noobject_scale = 1
box_scale = 1

# training configuration
num_epochs = 50
batch_size = 8
learn_rate = 1e-3
