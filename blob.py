from __future__ import absolute_import, division, print_function
import os
import numpy as np
import cv2
import xml.etree.cElementTree as ctree
import config as cfg

label2cls = {}
for idx, label in enumerate(cfg.label_names):
    label2cls[label] = idx


def prep_image_train(xml):
    # images in shape of [height, width, num_channels]
    # bboxes in shape of [num_gt_boxes, (x1, y1, x2, y2)], scaled in cfg.inp_size
    # bclasses in shape of [num_gt_boxes, (cls)]
    root = ctree.parse(os.path.join(
        cfg.train_data_dir, 'annotation', xml)).getroot()
    image_name = root.find('image').text
    image_height = float(root.find('height').text)
    image_width = float(root.find('width').text)
    classes = []
    boxes = []
    for box in root.findall('object'):
        classes.append(label2cls[box.find('class').text])
        bndbox = box.find('bndbox')
        boxes.append([float(bndbox.find('xmin').text),
                      float(bndbox.find('ymin').text),
                      float(bndbox.find('xmax').text),
                      float(bndbox.find('ymax').text)])

    # scale box coords to target size
    boxes = np.array(boxes, dtype=np.float32) * float(cfg.inp_size)
    boxes[0::2] /= height
    boxes[1::2] /= width

    classes = np.array(classes, dtype=np.float32)

    image = cv2.imread(os.path.join(cfg.train_data_dir, 'images', image_name))
    image = cv2.resize(image, (cfg.inp_size, cfg.inp_size)) / 255.0

    return image, boxes, classes


def prep_train_images_blob():
    blob_images = []
    blob_boxes = []
    blob_classes = []
    for xml in os.listdir(os.path.join(cfg.train_data_dir, 'annotation')):
        image, boxes, classes = prep_image_train(xml)
        blob_images.append(image)
        blob_boxes.append(boxes)
        blob_classes.append(classes)
        
    return blob_images, blob_boxes, blob_classes
