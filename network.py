from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
import config as cfg
from py_compute_targets import compute_targets_batch
from utils.bbox import bbox_transform

slim = tf.contrib.slim


def leaky(features, name=None):
    return tf.nn.leaky_relu(features, alpha=0.1, name=name)


# network's feed-forward used from nets/
def forward(inputs, num_outputs, scope=None):
    with tf.variable_scope(scope, 'net', [inputs], reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
            net = slim.repeat(inputs, 2, slim.conv2d, 64,
                              [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            net = slim.repeat(net, 2, slim.conv2d, 128,
                              [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')

            net = slim.repeat(net, 2, slim.conv2d, 256,
                              [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')

            net = slim.repeat(net, 2, slim.conv2d, 512,
                              [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')

            net = slim.repeat(net, 2, slim.conv2d, 512,
                              [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            net = slim.conv2d(net, 512, [3, 3], scope='conv6')
            net = slim.conv2d(net, num_outputs, [1, 1],
                              activation_fn=None, normalizer_fn=None, scope='logits')

    return net


class Network:
    def __init__(self, session, is_training=True, lr=1e-3, adamop=False, pretrained=False):
        self.sess = session
        self.is_training = is_training

        # network's placeholders
        self.images_ph = tf.placeholder(
            tf.float32, shape=[None, cfg.inp_size, cfg.inp_size, 3])

        self.anchors_ph = tf.placeholder(
            tf.float32, shape=[cfg.num_anchors, 2])

        # compute targets
        logits = forward(self.images_ph,
                         num_outputs=cfg.num_anchors * (cfg.num_classes + 5),
                         scope='vgg_16')

        h, w = logits.get_shape().as_list()[1:3]

        logits = tf.reshape(logits,
                            [-1, h * w, cfg.num_anchors, cfg.num_classes + 5])

        bbox_pred = tf.concat([tf.sigmoid(logits[:, :, :, 0:2]), tf.exp(logits[:, :, :, 2:4])],
                              axis=3)

        self.box_pred = tf.py_func(bbox_transform,
                                   [bbox_pred, self.anchors_ph, h, w],
                                   tf.float32, name='box_pred')

        self.iou_pred = tf.sigmoid(logits[:, :, :, 4:5])

        self.cls_pred = tf.nn.softmax(logits[:, :, :, 5:])

        if self.is_training:
            # network's placeholders in training
            self.boxes_ph = tf.placeholder(tf.float32, shape=None)

            self.classes_ph = tf.placeholder(tf.int8, shape=None)

            self.num_boxes_batch_ph = tf.placeholder(tf.float32, shape=None)

            _cls, _cls_mask, _iou, _iou_mask, _bbox, _bbox_mask = tf.py_func(compute_targets_batch,
                                                                             [h, w, self.box_pred, self.iou_pred,
                                                                              self.boxes_ph, self.classes_ph, self.anchors_ph],
                                                                             [tf.float32] * 6, name='targets')

            # network's losses, focal loss on cls?
            self.bbox_loss = tf.losses.mean_squared_error(labels=_bbox * _bbox_mask,
                                                          predictions=bbox_pred * _bbox_mask) / self.num_boxes_batch_ph
            self.iou_loss = tf.losses.mean_squared_error(labels=_iou * _iou_mask,
                                                         predictions=self.iou_pred * _iou_mask) / self.num_boxes_batch_ph
            self.cls_loss = tf.losses.mean_squared_error(labels=_cls * _cls_mask,
                                                         predictions=self.cls_pred * _cls_mask) / self.num_boxes_batch_ph

            self.total_loss = self.bbox_loss + self.iou_loss + self.cls_loss

            # network's optimizer
            self.global_step = tf.Variable(
                initial_value=0, trainable=False, name='global_step')

            if adamop:  # using Adam
                self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(
                    loss=self.total_loss, global_step=self.global_step)
            else:  # using SGD with momentum
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(
                    loss=self.total_loss, global_step=self.global_step)

        self.saver = tf.train.Saver(max_to_keep=3)

        self.load_ckpt(pretrained)

    def load_ckpt(self, pretrained):
        # restore model with ckpt/pretrain or init
        try:
            print('trying to restore last checkpoint')
            last_ckpt_path = tf.train.latest_checkpoint(
                checkpoint_dir=cfg.ckpt_dir)
            self.saver.restore(self.sess, save_path=last_ckpt_path)
            print('restored checkpoint from:', last_ckpt_path)
        except:
            if self.is_training:
                print('init variables')
                restored_vars = []
                global_vars = tf.global_variables()

                if pretrained:  # restore from tf slim vgg16 model
                    vgg_16_ckpt = os.path.join(
                        cfg.workspace, 'model/vgg_16.ckpt')
                    if os.path.exists(vgg_16_ckpt):
                        print('from model/vgg_16.ckpt')

                        import re
                        from tensorflow.python.pywrap_tensorflow import NewCheckpointReader

                        reader = NewCheckpointReader(vgg_16_ckpt)
                        # only restoring conv's weights
                        restored_var_names = [name + ':0'
                                              for name in reader.get_variable_to_dtype_map().keys()
                                              if re.match('^.*conv.*weights$', name)]

                        # update restored variables from pretrained vgg16 model
                        restored_vars = [var for var in global_vars
                                         if var.name in restored_var_names]

                        # update restored variables' name
                        restored_var_names = [var.name[:-2]
                                              for var in restored_vars]

                        # assignment variables
                        value_ph = tf.placeholder(tf.float32, shape=None)
                        for i in range(len(restored_var_names)):
                            self.sess.run(tf.assign(restored_vars[i], value_ph),
                                          feed_dict={value_ph: reader.get_tensor(restored_var_names[i])})

                initialized_vars = list(set(global_vars) - set(restored_vars))
                self.sess.run(tf.variables_initializer(initialized_vars))

    def train(self, batch_images, batch_boxes, batch_classes, anchors, num_boxes_batch):
        assert self.is_training

        step, bbox_loss, iou_loss, cls_loss, _ = self.sess.run([self.global_step,
                                                                self.bbox_loss, self.iou_loss, self.cls_loss,
                                                                self.optimizer],
                                                               feed_dict={self.images_ph: batch_images,
                                                                          self.boxes_ph: batch_boxes,
                                                                          self.classes_ph: batch_classes,
                                                                          self.anchors_ph: anchors,
                                                                          self.num_boxes_batch_ph: num_boxes_batch})

        return step, bbox_loss, iou_loss, cls_loss

    def save_ckpt(self, step):
        self.saver.save(self.sess,
                        save_path=os.path.join(cfg.ckpt_dir, cfg.model),
                        global_step=self.global_step)

        print('saved checkpoint at step {}'.format(step))

    def predict(self, scaled_images, anchors):
        box_pred, iou_pred, cls_pred = self.sess.run([self.box_pred, self.iou_pred, self.cls_pred],
                                                     feed_dict={self.images_ph: scaled_images,
                                                                self.anchors_ph: anchors})

        return box_pred, iou_pred, cls_pred


if __name__ == '__main__':
    Network(tf.Session(), pretrained=True)  # trying to init network
