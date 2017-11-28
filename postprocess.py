from __future__ import absolute_import, division, print_function
import numpy as np
import cv2
import config as cfg
from nms_wrapper import nms


def clip_boxes(boxes, im_shape):
    # Clip boxes[xmin, ymin, xmax, ymax] to image boundaries
    if boxes.shape[0] == 0:
        return boxes
    # 0 <= x1 < im_shape[0]
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[0] - 1), 0)
    # 0 <= y1 < im_shape[1]
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[1] - 1), 0)
    # 0 <= x2 < im_shape[0]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[0] - 1), 0)
    # 0 <= y2 < im_shape[1]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[1] - 1), 0)

    return boxes


def nms_detections(boxes_pred, scores, nms_thresh):
    dets = np.hstack((boxes_pred, scores[:, np.newaxis])).astype(np.float32)

    return nms(dets, nms_thresh, force_cpu=False)


def preprocess(bbox_pred, iou_pred, cls_pred, im_shape, thresh):
    # flatten logits' cells with anchors
    bbox_pred = np.reshape(bbox_pred, newshape=[-1, 4])
    bbox_pred[:, 0::2] *= float(im_shape[0])
    bbox_pred[:, 1::2] *= float(im_shape[1])
    bbox_pred = bbox_pred.astype(np.int)

    iou_pred = np.reshape(iou_pred, newshape=[-1])
    cls_pred = np.reshape(cls_pred, newshape=[-1, cfg.num_classes])

    cls_inds = np.argmax(cls_pred, axis=1)
    scores = iou_pred * cls_pred[np.arange(cls_pred.shape[0]), cls_inds]

    # select positive boxes with score larger than thresh
    keep_inds = np.where(scores > thresh)[0]
    bbox_pred = bbox_pred[keep_inds]
    cls_inds = cls_inds[keep_inds]
    scores = scores[keep_inds]

    # apply nms to remove overlap boxes
    keep_inds = np.zeros(len(bbox_pred), dtype=np.int)
    for c in range(cfg.num_classes):
        inds = np.where(cls_inds == c)[0]
        if len(inds) == 0:
            continue

        keep = nms_detections(boxes_pred[inds], scores[inds], nms_thresh=0.3)
        keep_inds[inds[keep]] = 1

    keep_inds = np.where(keep_inds > 0)[0]
    bbox_pred = bbox_pred[keep_inds]
    cls_inds = cls_inds[keep_inds]
    scores = scores[keep_inds]

    bbox_pred = clip_boxes(bbox_pred, im_shape)

    return bbox_pred, cls_inds, scores


def draw_targets(image, bbox_pred, cls_inds, scores):
    for b in range(bbox_pred.shape[0]):
        box_cls = cls_inds[b]
        if box_cls == 0:  # skip for others/background boxes
            continue

        box_color = cfg.label_colors[box_cls]

        cv2.rectangle(image,
                      (bbox_pred[b, 1], bbox_pred[b, 0]),
                      (bbox_pred[b, 3], bbox_pred[b, 1]),
                      box_color, 2)

        cv2.putText(image,
            '{}_{:.3f}'.format(cfg.label_names[box_cls], scores[b]),
            (bbox_pred[b, 1], bbox_pred[b, 0] - 5),  # text above 5 pixels
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75, box_color)

    return image
