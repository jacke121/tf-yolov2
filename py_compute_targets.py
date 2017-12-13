from __future__ import absolute_import, division, print_function
import numpy as np
import config as cfg
from utils.cython_bbox import yolo2bbox, bbox_overlaps, anchor_overlaps


def compute_targets(h, w, bbox_pred, iou_pred, gt_boxes, gt_classes, anchors):
    hw, num_anchors = bbox_pred.shape[0:2]

    _cls = np.zeros((hw, num_anchors, cfg.num_classes), dtype=np.float32)
    _cls_mask = np.zeros((hw, num_anchors, 1), dtype=np.float32)
    _iou = np.zeros((hw, num_anchors, 1), dtype=np.float32)
    _iou_mask = np.zeros((hw, num_anchors, 1), dtype=np.float32)
    _bbox = np.zeros((hw, num_anchors, 4), dtype=np.float32)
    _bbox_mask = np.zeros((hw, num_anchors, 1), dtype=np.float32)

    # scale bbox_pred to inp_size
    box_pred = yolo2bbox(bbox_pred, anchors, h, w)
    box_pred = np.reshape(box_pred, [-1, 4]) * cfg.inp_size

    box_ious = bbox_overlaps(np.ascontiguousarray(box_pred, dtype=np.float),
                             np.ascontiguousarray(gt_boxes, dtype=np.float))
    box_ious = np.reshape(box_ious, [hw, num_anchors, -1])

    neg_boxpred_inds = np.max(box_ious, axis=2) <= cfg.iou_thresh
    _iou_mask[neg_boxpred_inds] = cfg.noobject_scale * \
        (0 - iou_pred[neg_boxpred_inds])

    # scale gt_boxes to out_size
    gt_boxes[:, 0::2] *= (h / cfg.inp_size)
    gt_boxes[:, 1::2] *= (w / cfg.inp_size)

    # locate gt_boxes' cells
    cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    cell_inds = np.floor(cx) * w + np.floor(cy)
    cell_inds = cell_inds.astype(np.int)

    # compute target boxes for regression
    target_boxes = np.empty(gt_boxes.shape, dtype=np.float32)
    target_boxes[:, 0] = cx - np.floor(cx)
    target_boxes[:, 1] = cy - np.floor(cy)
    target_boxes[:, 2] = gt_boxes[:, 2] - gt_boxes[:, 0]
    target_boxes[:, 3] = gt_boxes[:, 3] - gt_boxes[:, 1]

    # match gt_boxes and anchors
    anchor_ious = anchor_overlaps(np.ascontiguousarray(anchors, dtype=np.float),
                                  np.ascontiguousarray(gt_boxes, dtype=np.float))
    anchor_inds = np.argmax(anchor_ious, axis=0)

    for i, cell_ind in enumerate(cell_inds):
        # for each gt_boxes
        if cell_ind >= hw or cell_ind < 0:
            continue
        a = anchor_inds[i]

        _cls[cell_ind, a, gt_classes[i]] = 1
        _cls_mask[cell_ind, a, :] = cfg.cls_scale

        _iou_truth = box_ious[cell_ind, a, i]
        _iou[cell_ind, a, :] = _iou_truth
        _iou_mask[cell_ind, a, :] = cfg.object_scale * \
            (_iou_truth - iou_pred[cell_ind, a, :])

        target_boxes[i, 2::4] /= anchors[a]
        _bbox[cell_ind, a, :] = target_boxes[i]
        _bbox_mask[cell_ind, a, :] = cfg.box_scale

    return _cls, _cls_mask, _iou, _iou_mask, _bbox, _bbox_mask


def compute_targets_batch(h, w, bbox_pred, iou_pred, gt_boxes, gt_classes, anchors):
    targets = [compute_targets(h, w, bbox_pred[b], iou_pred[b], gt_boxes[b],
                               gt_classes[b], anchors) for b in range(bbox_pred.shape[0])]

    _cls = np.stack(target[0] for target in targets)
    _cls_mask = np.stack(target[1] for target in targets)
    _iou = np.stack(target[2] for target in targets)
    _iou_mask = np.stack(target[3] for target in targets)
    _bbox = np.stack(target[4] for target in targets)
    _bbox_mask = np.stack(target[5] for target in targets)

    return _cls, _cls_mask, _iou, _iou_mask, _bbox, _bbox_mask
