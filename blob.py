from __future__ import absolute_import, division, print_function
import os
import numpy as np
import cv2
import xml.etree.cElementTree as ctree
import config as cfg

label2cls = {}
for idx, label in enumerate(cfg.label_names):
    label2cls[label] = idx


def prep_image(anno_dir, images_dir, xml, target_size):
    # image in shape of [height, width, num_channels]
    # boxes in shape of [num_gt_boxes, (xmin, ymin, xmax, ymax)], scaled in target_size
    # classes in shape of [num_gt_boxes, (cls)]
    root = ctree.parse(os.path.join(anno_dir, xml)).getroot()
    image_name = root.find('filename').text
    image_size = root.find('size')
    image_height = float(image_size.find('height').text)
    image_width = float(image_size.find('width').text)
    classes = []
    boxes = []
    for box in root.findall('object'):
        classes.append(label2cls[box.find('name').text])
        bndbox = box.find('bndbox')
        # using numpy's axis (not pascal/voc)
        boxes.append([float(bndbox.find('ymin').text),
                      float(bndbox.find('xmin').text),
                      float(bndbox.find('ymax').text),
                      float(bndbox.find('xmax').text)])

    # scale box coords to target size
    boxes = np.array(boxes, dtype=np.float32)
    boxes[:, 0::2] *= target_size[0] / image_height
    boxes[:, 1::2] *= target_size[1] / image_width

    classes = np.array(classes, dtype=np.int8)

    image = cv2.imread(os.path.join(cfg.data_dir, images_dir, image_name))
    # cv2 using BGR channels for imread, convert to RGB and normalize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_size[1], target_size[0]))
    # image -= [123.68, 116.78, 103.94]
    image = image / 128.0 - 1

    # if np.random.randint(0, 2):  # randomly left-right flipping
    #     image = cv2.flip(image, 1)
    #     boxes[:, 1::2] = target_size[1] - boxes[:, 1::2]

    return image, boxes, classes


class BlobLoader:
    def __init__(self, anno_dir, images_dir, batch_size, target_size):
        self.anno_dir = anno_dir
        self.images_dir = images_dir
        self.anno = os.listdir(os.path.join(cfg.data_dir, anno_dir))
        self.num_anno = len(self.anno)
        self.batch_size = batch_size
        self.target_size = target_size
        self.start_idx = 0

    def next_batch(self):
        np.random.shuffle(self.anno)

        while True:
            batch_images = []
            batch_boxes = []
            batch_classes = []

            end_idx = min(self.start_idx + self.batch_size, self.num_anno)
            for xml in self.anno[self.start_idx:end_idx]:
                image, boxes, classes = prep_image(
                    self.anno_dir, self.images_dir, xml, self.target_size)
                batch_images.append(image)
                batch_boxes.append(boxes)
                batch_classes.append(classes)

            batch_images = np.asarray(batch_images, dtype=np.float32)

            # add padding, list np.ndarray -> tf.tensor
            num_images = batch_images.shape[0]
            max_boxes_im = max([len(clss) for clss in batch_classes])
            num_boxes_batch = sum([len(clss) for clss in batch_classes])

            batch_boxes_pad = np.zeros(
                (num_images, max_boxes_im, 4), dtype=np.float32)
            batch_classes_pad = np.full(
                (num_images, max_boxes_im), -1, dtype=np.int8)  # -1 mean dontcare

            for i in range(num_images):
                num_boxes_im = len(batch_classes[i])
                batch_boxes_pad[i, 0:num_boxes_im, :] = batch_boxes[i]
                batch_classes_pad[i, 0:num_boxes_im] = batch_classes[i]

            yield batch_images, batch_boxes_pad, batch_classes_pad, num_boxes_batch

            # delete whatever yielded
            del batch_images
            del batch_boxes
            del batch_boxes_pad
            del batch_classes
            del batch_classes_pad
            del num_boxes_batch

            self.start_idx = end_idx if end_idx < self.num_anno else 0

            # complete epoch
            if self.start_idx == 0:
                break
