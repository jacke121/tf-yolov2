cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef box_overlaps_op(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes)

cdef anchor_overlaps_op(
        np.ndarray[DTYPE_t, ndim=2] anchors,
        np.ndarray[DTYPE_t, ndim=2] query_boxes)

cdef bbox_transform_op(
        np.ndarray[DTYPE_t, ndim=4] bbox_pred,
        np.ndarray[DTYPE_t, ndim=2] anchors, 
        int H, int W)
