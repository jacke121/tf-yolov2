from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
import config as cfg

slim = tf.contrib.slim


def leaky(features, name=None):
    return tf.nn.leaky_relu(features, alpha=0.1, name=name)


# import network defined in nets
def forward(inputs, num_outputs, scope=None):  # define network architecture in here
    # network forwarding, using vgg16 pretrained model (from tf slim)
    with tf.variable_scope(scope, 'net', [inputs], reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,  # yolov2 using leaky instead of relu
                            normalizer_fn=slim.batch_norm,
                            weights_initializer=tf.truncated_normal_initializer(
                                stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(5e-4),
                            biases_initializer=tf.zeros_initializer()):
            # inputs 416x416x3
            net = slim.repeat(inputs, 2, slim.conv2d,
                              64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 208x208x64

            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 104x104x128

            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            sc = slim.conv2d(net, 512, [1, 1], stride=2, scope='sc3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 52x52x256

            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = net + sc
            sc = slim.max_pool2d(net, [1, 1], stride=2, scope='sc4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')  # 26x26x512

            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = net + sc
            net = slim.max_pool2d(net, [2, 2], scope='pool5')  # 13x13x512

            net = slim.conv2d(net, 512, [3, 3], scope='conv6')
            # reduce features to prediction
            net = slim.conv2d(net, num_outputs, [1, 1], stride=1,
                              activation_fn=None, normalizer_fn=None, scope='logits')  # 13x13x(A*(5+C))

    return net


class Network:
    def __init__(self, session):
        self.sess = session

        self.images_ph = tf.placeholder(
            tf.float32, shape=[None, cfg.inp_size, cfg.inp_size, 3])

        logits = forward(self.images_ph,
                         num_outputs=cfg.num_anchors * (cfg.num_classes + 5),
                         scope='vgg_16')
        self.out_h, self.out_w = logits.get_shape().as_list()[1:3]

        logits = tf.reshape(logits,
                            shape=[-1, self.out_h * self.out_w, cfg.num_anchors, cfg.num_classes + 5])

        self.bbox_pred = tf.concat(
            [tf.sigmoid(logits[:, :, :, 0:2]), tf.exp(logits[:, :, :, 2:4])], axis=3)
        self.iou_pred = tf.sigmoid(logits[:, :, :, 4:5])
        self.cls_pred = tf.nn.softmax(logits[:, :, :, 5:])

        self.saver = tf.train.Saver()

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
                    # only restored conv's weights
                    restored_var_names = [name + ':0'
                                          for name in reader.get_variable_to_dtype_map().keys()
                                          if re.match('^.*conv.*weights$', name)]

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

    def run(self, images):
        bbox_pred, iou_pred, cls_pred = self.sess.run([self.bbox_pred, self.iou_pred, self.cls_pred],
                                                      feed_dict={self.images_ph: images})

        return bbox_pred, iou_pred, cls_pred
