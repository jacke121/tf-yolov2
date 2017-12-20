from __future__ import absolute_import, division, print_function
from nms.cpu_nms import cpu_nms
from nms.gpu_nms import gpu_nms


def nms(dets, thresh, force_cpu=False):
    if dets.shape[0] == 0:
        return []
    if force_cpu:
        return cpu_nms(dets, thresh)
    else:
        return gpu_nms(dets, thresh)
