from __future__ import absolute_import, division, print_function
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
    # network forwarding
    with tf.variable_scope(scope, 'net', [inputs], reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=leaky,
                            normalizer_fn=slim.batch_norm,
                            weights_initializer=tf.truncated_normal_initializer(
                                stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(5e-4),
                            biases_initializer=tf.zeros_initializer()):
            # use conv_s2 instead of max_pool2d, 12 convolution layers + 1 pooling layer
            net = slim.conv2d(
                inputs, 64, [5, 5], stride=2, scope='conv0')  # 208x208x64
            net = slim.max_pool2d(net, [2, 2], scope='pool0')

            net = slim.conv2d(
                net, 64, [3, 3], stride=1, scope='conv1_0')  # 104x104x64
            net = slim.conv2d(
                net, 64, [3, 3], stride=1, scope='conv1_1')  # 104x104x64

            net = slim.conv2d(
                net, 128, [3, 3], stride=2, scope='conv2_0')  # 52x52x128
            net = slim.conv2d(
                net, 128, [3, 3], stride=1, scope='conv2_1')  # 52x52x128

            shortcut = slim.conv2d(
                net, 256, [1, 1], stride=2, scope='shortcut2_3')
            net = slim.conv2d(
                net, 256, [3, 3], stride=2, scope='conv3_0')  # 26x26x256
            net = slim.conv2d(
                net + shortcut, 256, [3, 3], stride=1, scope='conv3_1')  # 26x26x256

            shortcut = slim.conv2d(
                net, 512, [1, 1], stride=2, scope='shortcut3_4')
            net = slim.conv2d(
                net, 512, [3, 3], stride=2, scope='conv4_0')  # 13x13x512
            net = slim.conv2d(
                net + shortcut, 512, [3, 3], stride=1, scope='conv4_1')  # 13x13x512

            # reduce features to prediction
            net = slim.conv2d(net, num_outputs, [1, 1], stride=1,
                              activation_fn=None, normalizer_fn=None, scope='conv5')  # 13x13x45

    return net


def compute_targets(h, w, bbox_pred_np, iou_pred_np, gt_boxes, gt_classes, anchors):
    # only 1 image per batch
    hw, num_anchors = bbox_pred_np.shape[0:2]  # a is num of anchors

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
        np.ascontiguousarray(bbox_pred_np, dtype=np.float),
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
        _cls_mask[cell_ind, a, :] = 1
        _cls[cell_ind, a, gt_classes[i]] = 1
        # iou(coef)
        _iou_mask[cell_ind, a, :] = cfg.object_scale * \
            (1 - iou_pred_np[cell_ind, a, :])
        _iou[cell_ind, a, :] = box_ious[cell_ind, a, i]
        # box_coords
        _box_mask[cell_ind, a, :] = 1
        target_boxes[i, 2:4] /= anchors[a]
        _box[cell_ind, a, :] = target_boxes[i]

    return _cls, _cls_mask, _iou, _iou_mask, _box, _box_mask


class Network:
    def __init__(self, session):
        # tf session
        self.sess = session

        # tf input placeholders, batch size must be 1
        self.images_ph = tf.placeholder(
            tf.float32, shape=[1, cfg.inp_size, cfg.inp_size, 3])
        # [#box_im, (x_tl, y_tl, x_br, y_br)]
        self.boxes_ph = tf.placeholder(tf.float32, shape=[None, 4])
        # [#box_im] of classes
        self.classes_ph = tf.placeholder(tf.int8, shape=[None])
        # target_size (height, width) anchors
        self.anchors_ph = tf.placeholder(
            tf.float32, shape=[cfg.num_anchors, 2])

        logits = forward(self.images_ph, num_outputs=cfg.num_anchors *
                         (cfg.num_classes + 5), scope='yolo')
        logits_h, logits_w = logits.get_shape().as_list()[1:3]

        # flatten logits' cells
        logits = tf.reshape(
            logits, shape=[-1, cfg.num_anchors, cfg.num_classes + 5])

        # compute targets
        bbox_pred = tf.concat(
            [tf.sigmoid(logits[:, :, 0:2]), tf.exp(logits[:, :, 2:4])], axis=2)
        iou_pred = tf.sigmoid(logits[:, :, 4:5])

        _cls, _cls_mask, _iou, _iou_mask, _box, _box_mask = tf.py_func(compute_targets,
                                                                       [logits_h, logits_w, bbox_pred, iou_pred,
                                                                        self.boxes_ph, self.classes_ph, self.anchors_ph],
                                                                       [tf.float32] * 6, name='targets')

        # network's losses
        self.cls_loss = tf.losses.softmax_cross_entropy(
            onehot_labels=tf.reshape(_cls, shape=[-1, cfg.num_classes]),
            logits=tf.reshape(logits[:, :, 5:] * _cls_mask, shape=[-1, cfg.num_classes]))

        # score = prob(object) * iou(bbox, object)
        self.iou_loss = tf.losses.mean_squared_error(labels=_iou * _iou_mask,
                                                     predictions=iou_pred * _iou_mask)

        self.bbox_loss = tf.losses.mean_squared_error(labels=_box,
                                                      predictions=bbox_pred * _box_mask)

        self.total_loss = self.cls_loss + self.iou_loss + self.bbox_loss

        # network's optimizer
        self.global_step = tf.Variable(
            initial_value=0, trainable=False, name='global_step')

        self.optimizer = tf.train.AdamOptimizer(learning_rate=cfg.learn_rate).minimize(
            loss=self.total_loss, global_step=self.global_step)

        # network's predictions in shape [hw, num_anchors, :]
        self.bbox_pred = tf.py_func(yolo2bbox,  # scale bbox_pred to 0~1
                                    [bbox_pred, self.anchors_ph,
                                        logits_h, logits_w],
                                    tf.float32, name='boxes_pred')

        self.iou_pred = iou_pred

        self.cls_pred = tf.nn.softmax(logits[:, :, 5:])  # on last dimension

        # network ckpt saver
        self.saver = tf.train.Saver()

        self.load_checkpoint()

    def load_checkpoint(self):
        # restore/init tf graph
        try:
            print('trying to restore last checkpoint')
            last_ckpt_path = tf.train.latest_checkpoint(
                checkpoint_dir=cfg.ckpt_dir)
            self.saver.restore(self.sess, save_path=last_ckpt_path)
            print('restored checkpoint from:', last_ckpt_path)
        except:
            print('init variables instead of restoring')
            self.sess.run(tf.global_variables_initializer())

    def train(self, image, boxes, classes, anchors):
        global_step, total_loss, _ = self.sess.run(
            [self.global_step, self.total_loss, self.optimizer],
            feed_dict={self.images_ph: image[np.newaxis],
                       self.boxes_ph: boxes,
                       self.classes_ph: classes,
                       self.anchors_ph: anchors})

        return global_step, total_loss

    def save(self):
        self.saver.save(sess=self.sess, save_path=cfg.save_path,
                        global_step=self.global_step)

    def predict(self, scaled_image):
        # image must be scaled in inp_size
        bbox_pred, iou_pred, cls_pred = self.sess.run(
            [self.bbox_pred, self.iou_pred, self.cls_pred],
            feed_dict={self.images_ph: scaled_image[np.newaxis]})

        return bbox_pred, iou_pred, cls_pred


if __name__ == '__main__':
    Network(tf.Session())
