from __future__ import absolute_import, division, print_function
import os
import numpy as np
import tensorflow as tf
import config as cfg
from utils.anchors import get_anchors
from utils.cython_bbox import yolo2bbox, bbox_overlaps, anchor_overlaps

slim = tf.contrib.slim


def leaky(features, name=None):
    return tf.nn.leaky_relu(features, alpha=0.1, name=name)


# import network defined in nets
def forward(inputs, num_outputs, scope=None):  # define network architecture in here
    # network forwarding, using vgg16 pretrained model (from tf slim) with duplicated first layer
    with tf.variable_scope(scope, 'net', [inputs], reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            weights_initializer=tf.truncated_normal_initializer(
                                stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(5e-4),
                            biases_initializer=tf.zeros_initializer()):
            # use conv_s2 instead of max_pool2d, 12 convolution layers + 1 pooling layer
            # inputs 416x416x3
            net = slim.repeat(inputs, 2, slim.conv2d,
                              64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 208x208x64

            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 104x104x128

            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 52x52x256

            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            shortcut = slim.max_pool2d(
                net, [1, 1], stride=2, scope='shortcut4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')  # 26x26x512

            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = net + shortcut
            shortcut = slim.max_pool2d(
                net, [1, 1], stride=2, scope='shortcut5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')  # 13x13x512

            net = slim.conv2d(net, 512, [3, 3], scope='conv6')
            net = net + shortcut
            # reduce features to prediction
            net = slim.conv2d(net, num_outputs, [1, 1], stride=1,
                              activation_fn=None, normalizer_fn=None, scope='conv7')  # 13x13x(A*(5+C))

    return net


def compute_targets(h, w, bbox_pred_np, iou_pred_np, gt_boxes, gt_classes, anchors):
    # only 1 image in processing
    hw, num_anchors, _ = bbox_pred_np.shape

    # compute ground-truth and regression weights (mask)
    # classes
    _cls = np.zeros((hw, num_anchors, cfg.num_classes), dtype=np.float32)
    _cls_mask = np.zeros((hw, num_anchors, 1), dtype=np.float32)
    # iou(coef)
    # iou(bbox_pred, object)
    _iou = np.zeros((hw, num_anchors, 1), dtype=np.float32)
    _iou_mask = np.zeros((hw, num_anchors, 1), dtype=np.float32)  # p(object)
    # box_coords
    _box = np.zeros((hw, num_anchors, 4), dtype=np.float32)
    _box_mask = np.zeros((hw, num_anchors, 1), dtype=np.float32)

    # transform and scale bbox_pred to inp_size
    bbox_np = yolo2bbox(
        np.ascontiguousarray(bbox_pred_np[np.newaxis], dtype=np.float),
        np.ascontiguousarray(anchors, dtype=np.float), h, w)
    bbox_np = np.reshape(bbox_np, newshape=[-1, 4]) * cfg.inp_size

    # compute iou between bbox_np and gt_boxes, overlaps on coords
    box_ious = bbox_overlaps(
        np.ascontiguousarray(bbox_np, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    box_ious = np.reshape(box_ious, newshape=[hw, num_anchors, -1])

    # select max iou for filtering negative bbox with iou_thresh
    max_iou_bbox = np.max(box_ious, axis=2).reshape(_cls_mask.shape)
    bbox_negative_inds = max_iou_bbox <= cfg.iou_thresh
    _iou_mask[bbox_negative_inds] = cfg.noobject_scale * \
        (0 - iou_pred_np[bbox_negative_inds])

    # scale gt_boxes to output_size
    gt_boxes[:, 0::2] *= (h / cfg.inp_size)
    gt_boxes[:, 1::2] *= (w / cfg.inp_size)

    cx = 0.5 * (gt_boxes[:, 0] + gt_boxes[:, 2])
    cy = 0.5 * (gt_boxes[:, 1] + gt_boxes[:, 3])
    cell_inds = np.floor(cx) * w + np.floor(cy)
    cell_inds = cell_inds.astype(np.int)

    # compute target_boxes for regression bbox_pred
    # target_boxes ~ gt_boxes
    target_boxes = np.empty(gt_boxes.shape, dtype=np.float32)
    target_boxes[:, 0] = cx - np.floor(cx)
    target_boxes[:, 1] = cy - np.floor(cy)
    target_boxes[:, 2] = gt_boxes[:, 2] - gt_boxes[:, 0]  # height
    target_boxes[:, 3] = gt_boxes[:, 3] - gt_boxes[:, 1]  # width

    # match best anchor for each gt_boxes, overlaps on area
    anchor_ious = anchor_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    anchor_inds = np.argmax(anchor_ious, axis=0)

    # compute ground-truth and regression weights
    for i, cell_ind in enumerate(cell_inds):
        if cell_ind >= hw or cell_ind < 0:
            continue
        a = anchor_inds[i]
        # classes
        _cls_mask[cell_ind, a, :] = cfg.cls_scale
        _cls[cell_ind, a, gt_classes[i]] = 1
        # iou(coef)
        _iou_truth = box_ious[cell_ind, a, i]
        _iou_mask[cell_ind, a, :] = cfg.object_scale * \
            (_iou_truth - iou_pred_np[cell_ind, a, :])
        _iou[cell_ind, a, :] = _iou_truth
        # coords
        _box_mask[cell_ind, a, :] = cfg.box_scale
        target_boxes[i, 2:4] /= anchors[a]
        _box[cell_ind, a, :] = target_boxes[i]

    return _cls, _cls_mask, _iou, _iou_mask, _box, _box_mask


def compute_targets_batch(h, w, bbox_pred_np, iou_pred_np, gt_boxes, gt_classes, anchors):
    targets = [compute_targets(h, w, bbox_pred_np[b], iou_pred_np[b], gt_boxes[b],
                               gt_classes[b], anchors) for b in range(bbox_pred_np.shape[0])]

    _cls = np.stack(target[0] for target in targets)
    _cls_mask = np.stack(target[1] for target in targets)
    _iou = np.stack(target[2] for target in targets)
    _iou_mask = np.stack(target[3] for target in targets)
    _box = np.stack(target[4] for target in targets)
    _box_mask = np.stack(target[5] for target in targets)

    return _cls, _cls_mask, _iou, _iou_mask, _box, _box_mask


class Network:
    def __init__(self, session, lr=1e-3, adamop=False, pretrained=False):
        self.name = cfg.model

        # tf session
        self.sess = session

        # tf input placeholders, batch size should be 2
        # [#im, height, width, depth]
        self.images_ph = tf.placeholder(
            tf.float32, shape=[None, cfg.inp_size, cfg.inp_size, 3])
        # [#im, #box_im, (x_tl, y_tl, x_br, y_br)]
        self.boxes_ph = tf.placeholder(tf.float32, shape=None)
        # [#im, #box_im] of classes
        self.classes_ph = tf.placeholder(tf.int8, shape=None)
        # [#anchors, (height, width)]
        self.anchors_ph = tf.placeholder(
            tf.float32, shape=[cfg.num_anchors, 2])

        logits = forward(self.images_ph,
                         num_outputs=cfg.num_anchors * (cfg.num_classes + 5),
                         scope='vgg_16')
        logits_h, logits_w = logits.get_shape().as_list()[1:3]
        # flatten logits' cells
        logits = tf.reshape(logits,
                            shape=[-1, logits_h * logits_w, cfg.num_anchors, cfg.num_classes + 5])

        # compute targets
        bbox_pred = tf.concat(
            [tf.sigmoid(logits[:, :, :, 0:2]), tf.exp(logits[:, :, :, 2:4])], axis=3)
        self.iou_pred = tf.sigmoid(logits[:, :, :, 4:5])  # keep dims
        self.cls_pred = tf.nn.softmax(logits[:, :, :, 5:])  # on last dimension

        _cls, _cls_mask, _iou, _iou_mask, _box, _box_mask = tf.py_func(compute_targets_batch,
                                                                       [logits_h, logits_w, bbox_pred, self.iou_pred,
                                                                        self.boxes_ph, self.classes_ph, self.anchors_ph],
                                                                       [tf.float32] * 6, name='targets')

        # network's losses
        self.cls_loss = tf.losses.mean_squared_error(labels=_cls * _cls_mask,
                                                     predictions=self.cls_pred * _cls_mask)
        # cls_loss softmax_cross_entropy loss?
        self.iou_loss = tf.losses.mean_squared_error(labels=_iou * _iou_mask,
                                                     predictions=self.iou_pred * _iou_mask)
        self.box_loss = tf.losses.mean_squared_error(labels=_box * _box_mask,
                                                     predictions=bbox_pred * _box_mask)
        self.total_loss = self.cls_loss + self.iou_loss + self.box_loss

        # network's optimizer
        self.global_step = tf.Variable(
            initial_value=0, trainable=False, name='global_step')
        # GradientDescentOptimizer better?
        if adamop:  # using adam optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(
                loss=self.total_loss, global_step=self.global_step)
        else:  # using SGD optimizer
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(
                loss=self.total_loss, global_step=self.global_step)

        # network's predictions in shape [#im, hw, num_anchors, :]
        self.box_pred = tf.py_func(yolo2bbox,  # scale bbox_pred to 0~1
                                   [bbox_pred, self.anchors_ph,
                                    logits_h, logits_w],
                                   tf.float32, name='box_pred')

        # network ckpt saver
        self.saver = tf.train.Saver()

        self.load_ckpt(pretrained)

    def load_ckpt(self, pretrained=False):
        # restore model with ckpt/pretrain or init
        try:
            print('trying to restore last checkpoint')
            last_ckpt_path = tf.train.latest_checkpoint(
                checkpoint_dir=cfg.ckpt_dir)
            self.saver.restore(self.sess, save_path=last_ckpt_path)
            print('restored checkpoint from:', last_ckpt_path)
        except:
            print('init variables')
            restored_vars = []
            global_vars = tf.global_variables()

            if pretrained:  # restore from tf slim vgg16 model
                vgg_16_ckpt = os.path.join(cfg.workspace, 'model/vgg_16.ckpt')
                if os.path.exists(vgg_16_ckpt):
                    print('from model/vgg_16.ckpt')

                    import re
                    from tensorflow.python.pywrap_tensorflow import NewCheckpointReader

                    reader = NewCheckpointReader(vgg_16_ckpt)
                    restored_var_names = sorted([name + ':0'
                                                 for name in reader.get_variable_to_dtype_map().keys()
                                                 if re.match('^.*conv.*weights$', name)])

                    # update restored variables from pretrained vgg16 model
                    restored_vars = [var for var in global_vars
                                     if var.name in restored_var_names]

                    restored_var_names = [var.name[:-2]
                                          for var in restored_vars]

                    value_ph = tf.placeholder(tf.float32, shape=None)
                    for i in range(len(restored_var_names)):
                        self.sess.run(tf.assign(restored_vars[i], value_ph),
                                      feed_dict={value_ph: reader.get_tensor(restored_var_names[i])})

            initialized_vars = list(set(global_vars) - set(restored_vars))
            self.sess.run(tf.variables_initializer(initialized_vars))

    def train(self, batch_images, batch_boxes, batch_classes, anchors):
        global_step, total_loss, _ = self.sess.run(
            [self.global_step, self.total_loss, self.optimizer],
            feed_dict={self.images_ph: batch_images,
                       self.boxes_ph: batch_boxes,
                       self.classes_ph: batch_classes,
                       self.anchors_ph: anchors})

        return global_step, total_loss

    def save_ckpt(self):
        self.saver.save(sess=self.sess, save_path=os.path.join(cfg.ckpt_dir, self.name),
                        global_step=self.global_step)
        print('saved checkpoint')

    def predict(self, scaled_images, anchors):
        box_pred, iou_pred, cls_pred = self.sess.run(
            [self.box_pred, self.iou_pred, self.cls_pred],
            feed_dict={self.images_ph: scaled_images,
                       self.anchors_ph: anchors})

        return box_pred, iou_pred, cls_pred


if __name__ == '__main__':
    Network(tf.Session(), pretrained=True)
