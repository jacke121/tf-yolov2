# yolov2 with tensorflow

## dataset:
change dataset directory in config.py, folder contain 'images' and 'annotation' subfolder
  
annotation using pascal/voc xml format

using default yolov2's anchor in 416x416, can be scaled to different size

## training:
using numpy's axis (not pascal/voc's axis): (ymin,xmin) = (xmin,ymin) and (ymax,xmax) = (xmax,ymax)

network using VGG16 pretrained model (removed fc layers) from tf-slim with adding 2 conv layers (conv6, logits). vgg_16.ckpt put in model/

python3 train.py --epochs NUM_EPOCHS --batch NUM_IMAGES --lr LEARN_RATE

losses collection (step, bbox, iou, class, total) will be saved in logs/losses_collection.txt

edit adamop to use AdamOptimizer instead of SGD with momentum (default) and pretrained to use VGG16 model from tf-slim instead of initialization from scratch

## validation - testing:

## demo:

## todo:
multiscale images training

fixed out of GPU's memory while training, exhausted with tf.ConfigProto GPU options

evaluate model and visualization with matplotlib

using compute targets (groundtruth and mask) with cython to improve training speed

using postprocess with cython to improve speed
