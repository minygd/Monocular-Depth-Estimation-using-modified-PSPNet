import tensorflow as tf
from tensorflow.contrib.slim import nets
import numpy as np

slim = tf.contrib.slim

def downsample(input, stride, kernel_size, name, output_channels=64):
    conv_channels = output_channels - input.get_shape().as_list()[-1]
    if conv_channels > 0:
        layer1 = tf.contrib.layers.separable_conv2d(input, num_outputs=conv_channels, kernel_size=kernel_size, stride=stride, padding='SAME', depth_multiplier=1, activation_fn=None, scope=name + '_1')
        layer2 = tf.contrib.layers.avg_pool2d(input, kernel_size=kernel_size, stride=stride, padding='SAME')
        output = tf.concat([layer1, layer2], axis=-1)
    elif conv_channels < 0:
        output = tf.contrib.layers.separable_conv2d(input, num_outputs=output_channels, kernel_size=kernel_size, stride=stride, padding='SAME', depth_multiplier=1, activation_fn=None, scope=name + '_2')
    else:
        output = tf.contrib.layers.avg_pool2d(input, kernel_size=kernel_size, stride=stride, padding='SAME')

    return tf.nn.relu(tf.contrib.layers.batch_norm(output, scope=name))

def upsample(input, stride, kernel_size, name, output_channels=64):
    output_size = [stride * input.get_shape().as_list()[1], stride * input.get_shape().as_list()[2]]
    output = tf.image.resize_bilinear(input, output_size, align_corners=True)
    output = tf.contrib.layers.separable_conv2d(output, num_outputs=output_channels, kernel_size=kernel_size, stride=1, padding='SAME', depth_multiplier=1, activation_fn=None, scope=name)

    return tf.nn.relu(tf.contrib.layers.batch_norm(output, scope=name))

def non_bottle_neck_block(input, name, dilated_rate):
    channels = input.get_shape()[-1]

    output = tf.contrib.layers.separable_conv2d(input, num_outputs=channels, kernel_size=3, stride=1, padding='SAME', depth_multiplier=1, activation_fn=None, scope=name + '_1')
    output = tf.nn.relu(output)
    output = tf.contrib.layers.separable_conv2d(output, num_outputs=channels, kernel_size=3, stride=1, padding='SAME', depth_multiplier=1, activation_fn=None, scope=name + '_2')
    output = tf.contrib.layers.batch_norm(output, scope=name + '_1')
    output = tf.nn.relu(output)

    output = tf.contrib.layers.separable_conv2d(input, num_outputs=channels, kernel_size=1, stride=1, rate=(1, dilated_rate), padding='SAME', depth_multiplier=1, activation_fn=None, scope=name + '_3')
    output = tf.nn.relu(output)
    output = tf.contrib.layers.separable_conv2d(input, num_outputs=channels, kernel_size=1, stride=1, rate=(dilated_rate, 1), padding='SAME', depth_multiplier=1, activation_fn=None, scope=name + '_4')
    output = tf.contrib.layers.batch_norm(output, scope=name + '_2')

    return tf.nn.relu(output + input)

def encoder(input, output_channels=256):
    output_2 = downsample(input, stride=2, kernel_size=3, name='down1', output_channels=16)
    for i in range(2):
        output_2 = non_bottle_neck_block(output_2, name='bt1_' + str(i), dilated_rate=1)

    output_1 = downsample(output_2, stride=2, kernel_size=3, name='down2', output_channels=64)
    for i in range(2):
        output_1 = non_bottle_neck_block(output_1, name='bt2_' + str(i), dilated_rate=1)

    output = downsample(output_1, stride=2, kernel_size=3, name='down3', output_channels=output_channels)
    for i in range(1):
        output = non_bottle_neck_block(output, name='bt3_' + str(i) + '_1', dilated_rate=2)
        output = non_bottle_neck_block(output, name='bt3_' + str(i) + '_2', dilated_rate=4)
        output = non_bottle_neck_block(output, name='bt3_' + str(i) + '_3', dilated_rate=8)
        output = non_bottle_neck_block(output, name='bt3_' + str(i) + '_4', dilated_rate=16)

    return output, output_1, output_2

def PSPNet(input, output_channels=1, reuse=False, is_training=True):
    feature_downsample_num = 3
    use_pretrained_model = False

    with tf.variable_scope('Generator', reuse=reuse):
        with tf.variable_scope('dense_feature_extractor'):
            if use_pretrained_model is True:
                net, _ = nets.resnet_v1.resnet_v1_50(input, output_stride=2 ** feature_downsample_num, global_pool=False, num_classes=None, is_training=False)
            else:
                net, net_1, net_2 = encoder(input, output_channels=256)
                feature_downsample_num = 3

                gate_layer_1 = tf.contrib.layers.conv2d(net_1, num_outputs=1, kernel_size=1, stride=1, padding='SAME', activation_fn=None)
                gate_layer_2 = tf.contrib.layers.conv2d(net_2, num_outputs=1, kernel_size=1, stride=1, padding='SAME', activation_fn=None)
                """
                net_1_2d_size = np.prod(np.array(net_1.get_shape().as_list()[1: 3]))
                net_2_2d_size = np.prod(np.array(net_2.get_shape().as_list()[1: 3]))

                gate_layer_1 = tf.reduce_mean(net_1, keepdims=False, axis=[1, 2])
                gate_layer_2 = tf.reduce_mean(net_2, keepdims=False, axis=[1, 2])
                gate_layer_1 = tf.contrib.layers.fully_connected(gate_layer_1, num_outputs=net_1_2d_size, activation_fn=None)
                gate_layer_2 = tf.contrib.layers.fully_connected(gate_layer_2, num_outputs=net_2_2d_size, activation_fn=None)
                gate_layer_1 = tf.reshape(gate_layer_1, [-1] + net_1.get_shape().as_list()[1: 3] + [1])
                gate_layer_2 = tf.reshape(gate_layer_2, [-1] + net_2.get_shape().as_list()[1: 3] + [1])
                """
                gate_layer_1 = tf.nn.sigmoid(gate_layer_1)
                gate_layer_2 = tf.nn.sigmoid(gate_layer_2)

        with tf.variable_scope('pyramid_pooling_module'):
            layer1 = downsample(net, stride=2, kernel_size=3, name='down1', output_channels=64)
            layer2 = downsample(net, stride=4, kernel_size=5, name='down2', output_channels=64)
            layer3 = downsample(net, stride=8, kernel_size=9, name='down3', output_channels=64)
            global_layer = tf.reduce_mean(net, keepdims=False, axis=[1, 2])

            layer1 = tf.contrib.layers.separable_conv2d(layer1, num_outputs=64, kernel_size=1, stride=1, padding='SAME', depth_multiplier=1, activation_fn=None, scope='conv1')
            layer2 = tf.contrib.layers.separable_conv2d(layer2, num_outputs=64, kernel_size=1, stride=1, padding='SAME', depth_multiplier=1, activation_fn=None, scope='conv2')
            layer3 = tf.contrib.layers.separable_conv2d(layer3, num_outputs=64, kernel_size=1, stride=1, padding='SAME', depth_multiplier=1, activation_fn=None, scope='conv3')
            global_layer = tf.contrib.layers.fully_connected(global_layer, num_outputs=64, activation_fn=None, scope='fc')

            layer1 = upsample(layer1, stride=2, kernel_size=3, name='up1', output_channels=64)
            layer2 = upsample(layer2, stride=4, kernel_size=3, name='up2', output_channels=64)
            layer3 = upsample(layer3, stride=8, kernel_size=3, name='up3', output_channels=64)
            global_layer = tf.expand_dims(global_layer, axis=1)
            global_layer = tf.expand_dims(global_layer, axis=1)
            global_layer = tf.tile(global_layer, [1] + net.get_shape().as_list()[1: 3] + [1])

            layer = tf.concat([net, layer1, layer2, layer3, global_layer], axis=-1)

        with tf.variable_scope('prediction'):
            output = upsample(layer, stride=2, kernel_size=3, name='up1', output_channels=64)
            output = gate_layer_1 * output + (1 - gate_layer_1) * net_1

            output = upsample(output, stride=2, kernel_size=3, name='up2', output_channels=16)
            output = gate_layer_2 * output + (1 - gate_layer_2) * net_2

            output = upsample(output, stride=2, kernel_size=3, name='up3', output_channels=1)
            
            #
            output = tf.nn.leaky_relu(output)
            #
            
            if is_training is True:
                layer = tf.nn.dropout(layer, 0.5)

    return output