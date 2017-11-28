from __future__ import absolute_import, division, print_function
import tensorflow as tf

slim = tf.contrib.slim


def _subsample(inputs, factor, scope=None):
    """Subsamples the input along the spatial dimensions.
    Args:
        inputs: A `Tensor` of size [batch, height_in, width_in, channels].
        factor: The subsampling factor.
        scope: Optional variable_scope.
    Returns:
        output: A `Tensor` of size [batch, height_out, width_out, channels] with the
          input, either intact (if factor == 1) or subsampled (if factor > 1).
    """
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def _bottleneck(inputs, depth, depth_bottleneck, stride, scope=None):
    """Bottleneck residual unit variant with BN before convolutions.
    Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    scope: Optional variable_scope.
    Returns:
        The ResNet unit's output.
    """
    with tf.variable_scope(scope, 'bottleneck', [inputs]):
        depth_in = slim.utils.last_dimension(inputs.get_shape())
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        if depth == depth_in:
            shortcut = _subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None, scope='shortcut')

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')

        residual = slim.conv2d(residual, depth_bottleneck, [3, 3], stride=stride,
                               scope='conv2')

        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               scope='conv3')

        output = residual + shortcut

    return output


def resnet_block(inputs, base_depth, num_units, stride, scope=None):
    """Helper function for creating a resnet_v2 bottleneck block.
    Args:
        inputs: A tensor of size [batch, height, width, channels].
        base_depth: The depth of the bottleneck layer for each unit.
        num_units: The number of units in the block.
        stride: The stride of the block, implemented as a stride in the last unit.
          All other units have stride=1.
        scope: The scope of the block.
    Returns:
        A resnet_v2 bottleneck block.
    """
    depth = base_depth * 4

    with tf.variable_scope(scope, 'block', [inputs]):
        net = _bottleneck(inputs, depth, base_depth, stride=stride)  # first block

        net = slim.repeat(net, num_units - 1, _bottleneck, depth, base_depth, stride=1)  # rest blocks

    return net
