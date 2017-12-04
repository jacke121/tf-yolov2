# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Sergey Karayev
# --------------------------------------------------------

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def bbox_overlaps(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float in order (x1, y1, x2, y2)
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t ih, iw, box_area
    cdef DTYPE_t ua
    cdef unsigned int n, k
    for n in range(N):
        box_area = (
            (boxes[n, 2] - boxes[n, 0] + 1) *
            (boxes[n, 3] - boxes[n, 1] + 1)
        )
        for k in range(K):
            ih = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if ih > 0:
                iw = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if iw > 0:
                    inter_area = ih * iw
                    ua = float(
                        (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
                        (query_boxes[k, 3] - query_boxes[k, 1] + 1) +
                        box_area - inter_area
                    )
                    overlaps[n, k] = inter_area / ua
    return overlaps

def anchor_overlaps(
        np.ndarray[DTYPE_t, ndim=2] anchors,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    For each query box compute the intersection ratio covered by anchors
    ----------
    Parameters
    ----------
    anchors: (N, 2) ndarray of float in order (height, width)
    query_boxes: (K, 4) ndarray of float in order (x1, y1, x2, y2)
    Returns
    -------
    overlaps: (N, K) ndarray of intersec between boxes and query_boxes
    """
    cdef unsigned int N = anchors.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t ih, iw, anchor_area, inter_area
    cdef DTYPE_t boxh, boxw
    cdef unsigned int n, k
    for n in range(N):
        anchor_area = anchors[n, 0] * anchors[n, 1]
        for k in range(K):
            boxh = query_boxes[k, 2] - query_boxes[k, 0] + 1
            boxw = query_boxes[k, 3] - query_boxes[k, 1] + 1
            ih = min(anchors[n, 0], boxh)
            iw = min(anchors[n, 1], boxw)
            inter_area = ih * iw
            overlaps[n, k] = inter_area / (anchor_area + boxh * boxw - inter_area)
    return overlaps

def yolo2bbox(  # similar to bbox_transform_inv of faster-rcnn
        np.ndarray[DTYPE_t, ndim=4] bbox_pred,
        np.ndarray[DTYPE_t, ndim=2] anchors, 
        int H, int W):
    """
    Parameters
    ----------
    bbox_pred: 4-dim float ndarray [bsize, HxW, num_anchors, 4] of (sig(tx), sig(ty), exp(th), exp(tw))
    anchors: [num_anchors, 2] of (ph, pw)
    H, W: height, width of features map
    Returns
    -------
    bbox_out: (HxW, num_anchors, 4) ndarray of bbox (x1, y1, x2, y2) rescaled to (0, 1)
    """
    cdef unsigned int bsize = bbox_pred.shape[0]
    cdef unsigned int num_anchors = anchors.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=3] bbox_scale = np.zeros((bsize, H*W, num_anchors, 4), dtype=DTYPE)
    
    for b in range(bsize):
        for row in range(H):
            for col in range(W):
                ind = row * W + col
                for a in range(num_anchors):
                    cx = row + bbox_pred[b, ind, a, 0]
                    cy = col + bbox_pred[b, ind, a, 1]
                    bh = bbox_pred[b, ind, a, 2] * anchors[a, 0] * 0.5
                    bw = bbox_pred[b, ind, a, 3] * anchors[a, 1] * 0.5
                    bbox_scale[b, ind, a, 0] = (cx - bh) / H
                    bbox_scale[b, ind, a, 1] = (cy - bw) / W
                    bbox_scale[b, ind, a, 2] = (cx + bh) / H
                    bbox_scale[b, ind, a, 3] = (cy + bw) / W

    return bbox_scale
