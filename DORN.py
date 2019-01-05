import tensorflow as tf
from tensorflow.contrib.slim import nets
import numpy as np

slim = tf.contrib.slim

fully_encoder_channels = 64
downsample_num = 3

def conv( x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

def conv_block(x, num_out_layers, kernel_size):
    conv1 = conv(x, num_out_layers, kernel_size, 1)
    conv2 = conv(conv1, num_out_layers, kernel_size, 2)
    return conv2

def DORN(input, output_channels=1, reuse=tf.AUTO_REUSE, is_training=True):
    """
    with tf.variable_scope('model', reuse=reuse):
        with tf.variable_scope('encoder', reuse=reuse):
            conv1 = conv_block(input, 32, 7) # H/2
            conv2 = conv_block(conv1, 64, 5) # H/4
            conv3 = conv_block(conv2, 128, 3) # H/8
            conv4 = conv_block(conv3, 256, 3) # H/16
            conv5 = conv_block(conv4, 512, 3) # H/32
            conv6 = conv_block(conv5, 512, 3) # H/64
            conv7 = conv_block(conv6, 512, 3) # H/128
            net = conv7
    """
    with tf.variable_scope('Generator', reuse=reuse):
        with tf.variable_scope('dense_feature_extractor', reuse=reuse):
            net, endpoints = nets.resnet_v1.resnet_v1_50(input, output_stride=2 ** downsample_num, global_pool=False, num_classes=None, is_training=False)

        with tf.variable_scope('scene_understanding_modular', reuse=reuse):
            with tf.variable_scope('full_image_encoder', reuse=reuse):
                initial_height, initial_width = net.get_shape().as_list()[1: -1]
                layer1 = tf.contrib.layers.avg_pool2d(net, kernel_size=3, stride=2, padding='SAME')
                fc_init_length = np.prod(layer1.get_shape().as_list()[1: ])
                layer1 = tf.reshape(layer1, [-1, fc_init_length])
                layer1 = tf.contrib.layers.fully_connected(layer1, num_outputs=fully_encoder_channels, activation_fn=None)
                layer1 = tf.reshape(layer1, [-1, 1, 1, fully_encoder_channels])
                layer1 = tf.contrib.layers.conv2d(layer1, num_outputs=fully_encoder_channels, kernel_size=1, stride=1, padding='VALID', activation_fn=None)
                layer1 = tf.tile(layer1, [1, initial_height, initial_width, 1])

            with tf.variable_scope('cross_channel_information_learner', reuse=reuse):
                cross_info_channels = net.get_shape().as_list()[-1]
                layer2 = tf.contrib.layers.conv2d(net, num_outputs=cross_info_channels, kernel_size=1, stride=1, padding='SAME', activation_fn=None)
                layer2 = tf.contrib.layers.batch_norm(layer2)

            with tf.variable_scope('aspp', reuse=reuse):
                aspp_channels = net.get_shape().as_list()[-1]
                layer3 = tf.contrib.layers.conv2d(net, num_outputs=aspp_channels, kernel_size=3, stride=1, padding='SAME', rate=3, activation_fn=None)
                layer4 = tf.contrib.layers.conv2d(net, num_outputs=aspp_channels, kernel_size=3, stride=1, padding='SAME', rate=6, activation_fn=None)
                layer5 = tf.contrib.layers.conv2d(net, num_outputs=aspp_channels, kernel_size=3, stride=1, padding='SAME', rate=12, activation_fn=None)

            # add conv for each extractor
            layer1 = tf.contrib.layers.conv2d(layer1, num_outputs=fully_encoder_channels, kernel_size=1, stride=1, padding='SAME', activation_fn=None, scope='conv1')
            layer2 = tf.contrib.layers.conv2d(layer2, num_outputs=fully_encoder_channels, kernel_size=1, stride=1, padding='SAME', activation_fn=None, scope='conv2')
            layer3 = tf.contrib.layers.conv2d(layer3, num_outputs=fully_encoder_channels, kernel_size=1, stride=1, padding='SAME', activation_fn=None, scope='conv3')
            layer4 = tf.contrib.layers.conv2d(layer4, num_outputs=fully_encoder_channels, kernel_size=1, stride=1, padding='SAME', activation_fn=None, scope='conv4')
            layer5 = tf.contrib.layers.conv2d(layer5, num_outputs=fully_encoder_channels, kernel_size=1, stride=1, padding='SAME', activation_fn=None, scope='conv5')

            layer = tf.concat([layer1, layer2, layer3, layer4, layer5], axis=-1)
            total_channels = layer.get_shape().as_list()[-1]
            layer = tf.contrib.layers.conv2d(layer, num_outputs=total_channels, kernel_size=1, stride=1, padding='SAME', activation_fn=None)
            layer = tf.nn.relu(tf.contrib.layers.batch_norm(layer))

        with tf.variable_scope('prediction', reuse=reuse):
            layer = tf.image.resize_bilinear(layer, input.get_shape().as_list()[1: -1], align_corners=True)
            layer = tf.contrib.layers.conv2d(layer, num_outputs=16, kernel_size=3, stride=1, padding='SAME', activation_fn=None)
            layer = tf.contrib.layers.conv2d(layer, num_outputs=output_channels, kernel_size=3, stride=1, padding='SAME', activation_fn=None)
            layer = tf.nn.relu(tf.contrib.layers.batch_norm(layer))

            #layer = tf.nn.dropout(layer, 0.8)

    return layer