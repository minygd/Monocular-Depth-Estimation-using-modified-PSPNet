import tensorflow as tf
import ReadData as read

USE_FUSED_BN = True
BN_EPSILON = 0.001
BN_MOMENTUM = 0.99

def reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [
            min(shape[1], kernel_size[0]), min(shape[2], kernel_size[1])
        ]
    return kernel_size_out

def relu_separable_bn_block(inputs, filters, name_prefix, is_training, data_format):
    bn_axis = -1 if data_format == 'channels_last' else 1

    inputs = tf.nn.relu(inputs, name=name_prefix + '_act')
    inputs = tf.layers.separable_conv2d(inputs, filters, (3, 3),
                        strides=(1, 1), padding='same',
                        data_format=data_format,
                        activation=None, use_bias=False,
                        depthwise_initializer=tf.contrib.layers.xavier_initializer(),
                        pointwise_initializer=tf.contrib.layers.xavier_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        name=name_prefix, reuse=None)
    inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name=name_prefix + '_bn', axis=bn_axis,
                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    return inputs

"""Contains the definition for Xception V1 classification network."""
def XceptionModel(input_image, num_classes, is_training = False, data_format='channels_last'):
    bn_axis = -1 if data_format == 'channels_last' else 1
    # Entry Flow
    inputs = tf.layers.conv2d(input_image, 32, (3, 3), use_bias=False, name='block1_conv1', strides=(2, 2),
                padding='valid', data_format=data_format, activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer())
    inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name='block1_conv1_bn', axis=bn_axis,
                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    inputs = tf.nn.relu(inputs, name='block1_conv1_act')

    inputs = tf.layers.conv2d(inputs, 64, (3, 3), use_bias=False, name='block1_conv2', strides=(1, 1),
                padding='valid', data_format=data_format, activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer())
    inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name='block1_conv2_bn', axis=bn_axis,
                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    inputs = tf.nn.relu(inputs, name='block1_conv2_act')

    residual = tf.layers.conv2d(inputs, 128, (1, 1), use_bias=False, name='conv2d_1', strides=(2, 2),
                padding='same', data_format=data_format, activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer())
    residual = tf.layers.batch_normalization(residual, momentum=BN_MOMENTUM, name='batch_normalization_1', axis=bn_axis,
                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

    inputs = tf.layers.separable_conv2d(inputs, 128, (3, 3),
                        strides=(1, 1), padding='same',
                        data_format=data_format,
                        activation=None, use_bias=False,
                        depthwise_initializer=tf.contrib.layers.xavier_initializer(),
                        pointwise_initializer=tf.contrib.layers.xavier_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        name='block2_sepconv1', reuse=None)
    inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name='block2_sepconv1_bn', axis=bn_axis,
                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

    inputs = relu_separable_bn_block(inputs, 128, 'block2_sepconv2', is_training, data_format)

    inputs = tf.layers.max_pooling2d(inputs, pool_size=(3, 3), strides=(2, 2),
                                    padding='same', data_format=data_format,
                                    name='block2_pool')

    inputs = tf.add(inputs, residual, name='residual_add_0')
    residual = tf.layers.conv2d(inputs, 256, (1, 1), use_bias=False, name='conv2d_2', strides=(2, 2),
                padding='same', data_format=data_format, activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer())
    residual = tf.layers.batch_normalization(residual, momentum=BN_MOMENTUM, name='batch_normalization_2', axis=bn_axis,
                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

    inputs = relu_separable_bn_block(inputs, 256, 'block3_sepconv1', is_training, data_format)
    inputs = relu_separable_bn_block(inputs, 256, 'block3_sepconv2', is_training, data_format)

    inputs = tf.layers.max_pooling2d(inputs, pool_size=(3, 3), strides=(2, 2),
                                    padding='same', data_format=data_format,
                                    name='block3_pool')
    inputs = tf.add(inputs, residual, name='residual_add_1')

    residual = tf.layers.conv2d(inputs, 728, (1, 1), use_bias=False, name='conv2d_3', strides=(2, 2),
                padding='same', data_format=data_format, activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer())
    residual = tf.layers.batch_normalization(residual, momentum=BN_MOMENTUM, name='batch_normalization_3', axis=bn_axis,
                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

    inputs = relu_separable_bn_block(inputs, 728, 'block4_sepconv1', is_training, data_format)
    inputs = relu_separable_bn_block(inputs, 728, 'block4_sepconv2', is_training, data_format)

    inputs = tf.layers.max_pooling2d(inputs, pool_size=(3, 3), strides=(2, 2),
                                    padding='same', data_format=data_format,
                                    name='block4_pool')
    inputs = tf.add(inputs, residual, name='residual_add_2')

    # Middle Flow
    for index in range(8):
        residual = inputs
        prefix = 'block' + str(index + 5)

        inputs = relu_separable_bn_block(inputs, 728, prefix + '_sepconv1', is_training, data_format)
        inputs = relu_separable_bn_block(inputs, 728, prefix + '_sepconv2', is_training, data_format)
        inputs = relu_separable_bn_block(inputs, 728, prefix + '_sepconv3', is_training, data_format)
        inputs = tf.add(inputs, residual, name=prefix + '_residual_add')

    outputs_1 = inputs

    # Exit Flow
    residual = tf.layers.conv2d(inputs, 1024, (1, 1), use_bias=False, name='conv2d_4', strides=(2, 2),
                padding='same', data_format=data_format, activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer())
    residual = tf.layers.batch_normalization(residual, momentum=BN_MOMENTUM, name='batch_normalization_4', axis=bn_axis,
                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

    inputs = relu_separable_bn_block(inputs, 728, 'block13_sepconv1', is_training, data_format)
    inputs = relu_separable_bn_block(inputs, 1024, 'block13_sepconv2', is_training, data_format)

    inputs = tf.layers.max_pooling2d(inputs, pool_size=(3, 3), strides=(2, 2),
                                    padding='same', data_format=data_format,
                                    name='block13_pool')
    inputs = tf.add(inputs, residual, name='residual_add_3')

    outputs_2 = inputs

    inputs = tf.layers.separable_conv2d(inputs, 1536, (3, 3),
                        strides=(1, 1), padding='same',
                        data_format=data_format,
                        activation=None, use_bias=False,
                        depthwise_initializer=tf.contrib.layers.xavier_initializer(),
                        pointwise_initializer=tf.contrib.layers.xavier_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        name='block14_sepconv1', reuse=None)
    inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name='block14_sepconv1_bn', axis=bn_axis,
                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    inputs = tf.nn.relu(inputs, name='block14_sepconv1_act')

    inputs = tf.layers.separable_conv2d(inputs, 2048, (3, 3),
                        strides=(1, 1), padding='same',
                        data_format=data_format,
                        activation=None, use_bias=False,
                        depthwise_initializer=tf.contrib.layers.xavier_initializer(),
                        pointwise_initializer=tf.contrib.layers.xavier_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        name='block14_sepconv2', reuse=None)
    inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name='block14_sepconv2_bn', axis=bn_axis,
                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    inputs = tf.nn.relu(inputs, name='block14_sepconv2_act')

    if data_format == 'channels_first':
        channels_last_inputs = tf.transpose(inputs, [0, 2, 3, 1])
    else:
        channels_last_inputs = inputs

    inputs = tf.layers.average_pooling2d(inputs, pool_size = reduced_kernel_size_for_small_input(channels_last_inputs, [10, 10]), strides = 1, padding='valid', data_format=data_format, name='avg_pool')

    if data_format == 'channels_first':
        inputs = tf.squeeze(inputs, axis=[2, 3])
    else:
        inputs = tf.squeeze(inputs, axis=[1, 2])

    outputs = tf.layers.dense(inputs, num_classes,
                            activation=tf.nn.softmax, use_bias=True,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.zeros_initializer(),
                            name='dense', reuse=None)

    return outputs_1, outputs_2, outputs

def conv2d(input, output_channels, kernel_size=3, stride=1, padding='same', is_bn=True, is_relu=False):
    #output = tf.contrib.layers.conv2d(input, num_outputs=output_channels, kernel_size=kernel_size, stride=stride, padding=padding, activation_fn=tf.nn.relu)
    output = tf.layers.conv2d(input, filters=output_channels, kernel_size=kernel_size, strides=stride, padding=padding)

    if is_bn is True:
        output = tf.contrib.layers.batch_norm(output)
    if is_relu is True:
        output = tf.nn.leaky_relu(output)

    return output

def upsampling(input, scaling=2):
    resize_shape = [scaling * input.get_shape().as_list()[1], scaling * input.get_shape().as_list()[2]]
    output = tf.image.resize_bilinear(input, resize_shape, align_corners=True)
    #output = conv2d(output, output_channels=output.get_shape()[-1], is_bn=True, is_relu=False)
    return output

def separateConv2d(input, output_channels, kernel_size=3, stride=1, padding='same', is_bn=True, is_relu=False):
    #output = tf.contrib.layers.separable_conv2d(input, num_outputs=output_channels, kernel_size=kernel_size, padding=padding, activation_fn=tf.nn.relu, depth_multiplier=1)
    output = tf.layers.separable_conv2d(input, filters=output_channels, kernel_size=kernel_size, padding=padding, depth_multiplier=1)

    if is_bn is True:
        output = tf.contrib.layers.batch_norm(output)
    if is_relu is True:
        output = tf.nn.leaky_relu(output)

    return output

def globalAveragePooling2d(input):
    return tf.reduce_mean(input, [1, 2], keepdims=True)

def Xception_block(input, output_channels, is_relu=True):
    if isinstance(output_channels, list) is True:
        if len(output_channels) != 2:
            print("output_channels list's length is incorrect.")
            return None
        output_channels_1, output_channels_2 = output_channels[:]
    elif isinstance(output_channels, tuple) is True:
        if len(output_channels) != 2:
            print("output_channels tuple's length is incorrect.")
            return None
        output_channels_1, output_channels_2 = output_channels
    else:
        output_channels_1 = output_channels_2 = output_channels

    if is_relu is True:
        middle_layer = tf.nn.leaky_relu(input)
    else:
        middle_layer = input

    middle_layer = separateConv2d(middle_layer, output_channels=output_channels_1, stride=1, is_bn=True, is_relu=True)
    middle_layer = separateConv2d(middle_layer, output_channels=output_channels_2, stride=1, is_bn=True, is_relu=False)
    middle_layer = tf.contrib.layers.max_pool2d(middle_layer, kernel_size=3, stride=2, padding='same')

    layer = conv2d(input, output_channels=output_channels_2, kernel_size=1, stride=2, is_bn=True, is_relu=False)
    layer = tf.add(layer, middle_layer)

    return layer

def Xception_middle_block(input, output_channels):
    layer = input

    for i in range(3):
        layer = tf.nn.leaky_relu(layer)
        layer = separateConv2d(layer, output_channels=output_channels)

    return tf.add(input, layer)

def Xception(input, output_channels):
    layer = conv2d(input, output_channels=16, stride=2, is_bn=True, is_relu=True)
    layer = conv2d(layer, output_channels=64, stride=1, is_bn=True, is_relu=True)

    layer_1 = Xception_block(layer, output_channels=128, is_relu=False)
    layer_2 = Xception_block(layer_1, output_channels=256)
    layer_3 = Xception_block(layer_2, output_channels=512)

    for i in range(3):
        layer_3 = Xception_middle_block(layer_3, output_channels=512)

    layer_4 = Xception_block(layer_3, output_channels=[512, output_channels])

    return layer_3, layer_4

def ResnetBlock(input, output_channels, is_down=True):
    if is_down is True:
        stride = 2
    else:
        stride = 1
    layer = conv2d(input, output_channels=output_channels, kernel_size=3, stride=stride, is_bn=True, is_relu=False)
    layer = conv2d(layer, output_channels=output_channels, kernel_size=3, stride=1, is_bn=False, is_relu=False)

    if is_down is True:
        input = tf.contrib.layers.max_pool2d(input, kernel_size=3, stride=2, padding='same')
        if input.get_shape().as_list()[-1] != output_channels:
            input = tf.pad(input, [[0, 0], [0, 0], [0, 0], [0, output_channels - input.get_shape().as_list()[-1]]])

    return tf.nn.relu(layer + input)

def Resnet18(input, output_channels):
    # conv1
    layer = conv2d(input, output_channels=64, kernel_size=7, stride=2, is_bn=True, is_relu=True)

    # conv2_x
    layer = tf.contrib.layers.max_pool2d(layer, kernel_size=3, stride=2, padding='same')
    for i in range(2):
        layer = ResnetBlock(layer, output_channels=64, is_down=False)

    # conv3_x
    for i in range(2):
        if i == 0:
            is_down = True
        else:
            is_down = False
        layer = ResnetBlock(layer, output_channels=128, is_down=is_down)

    # conv4_x
    for i in range(2):
        if i == 0:
            is_down = True
        else:
            is_down = False
        layer = ResnetBlock(layer, output_channels=256, is_down=is_down)

    # conv5_x
    layer_1 = layer
    for i in range(2):
        if i == 0:
            is_down = True
        else:
            is_down = False
        layer_1 = ResnetBlock(layer_1, output_channels=output_channels, is_down=is_down)

    return layer, layer_1

def attentionRefinement(input):
    layer = globalAveragePooling2d(input)
    layer = conv2d(layer, output_channels=layer.get_shape()[-1], kernel_size=1, is_bn=True, is_relu=False)
    layer = tf.nn.sigmoid(layer)
    return tf.multiply(layer, input)

def featureFusion(input_1, input_2):
    input = tf.concat([input_1, input_2], axis=-1)
    input = conv2d(input, output_channels=input.get_shape()[-1], kernel_size=1, is_bn=True, is_relu=True)
    layer = globalAveragePooling2d(input)
    layer = conv2d(layer, output_channels=layer.get_shape()[-1], kernel_size=1, is_bn=False, is_relu=True)
    layer = conv2d(layer, output_channels=layer.get_shape()[-1], kernel_size=1, is_bn=False, is_relu=False)
    layer = tf.nn.sigmoid(layer)
    return tf.add(input, tf.multiply(input, layer))

def construct_network(image, is_training=True):
    # confirm the number of input's channels is 3
    """
    if image.get_shape().as_list()[-1] == 1:
        image = tf.tile(image, [1, 1, 1, 3])
    elif image.get_shape().as_list()[-1] != 3:
        print("invalid channel input to the network.")
        return None
    """

    # spatial path
    with tf.variable_scope('spatial_path'):
        layer = image
        for i in range(3):
            output_channels = 2 ** (i + 6)
            layer = conv2d(layer, output_channels=output_channels, stride=2, is_bn=True, is_relu=True)
        spatial_channels = layer.get_shape()[-1]

    # context path
    #with tf.variable_scope('context_path'):
    layer_1, layer_2, avg_layer = XceptionModel(image, num_classes=1000, is_training=False)
    #layer_1, layer_2, _ = XceptionModel(image, num_classes=1000, is_training=False)
    #print(layer_1.get_shape(), layer_2.get_shape())
    
    with tf.variable_scope('attention_refine'):
        layer_1 = attentionRefinement(layer_1)
        layer_2 = attentionRefinement(layer_2)

        avg_layer = tf.reshape(avg_layer, [-1, 1, 1, 1000])
        avg_layer = conv2d(avg_layer, output_channels=spatial_channels, stride=1, kernel_size=1, is_bn=False, is_relu=True)
        avg_layer = tf.tile(avg_layer, [1, layer_1.get_shape().as_list()[1], layer_1.get_shape().as_list()[2], 1])

        # not sure how to fuse two tensor, just try one method
        """
        layer_3 = tf.contrib.layers.conv2d_transpose(layer_2, num_outputs=256, kernel_size=3, stride=2, padding='same')
        layer_3 = tf.concat([layer_1, layer_3], axis=-1)
        layer_4 = tf.contrib.layers.conv2d_transpose(layer_3, num_outputs=64, kernel_size=3, stride=2, padding='same')
        """

        layer_3 = tf.pad(upsampling(tf.concat([layer_1, avg_layer], axis=-1), scaling=2), [[0, 0], [1, 1], [1, 1], [0, 0]])
        layer_4 = tf.concat([layer_3, upsampling(layer_2, scaling=4)], axis=-1)

        for i in range(2):
            layer_4 = conv2d(layer_4, output_channels=spatial_channels, is_bn=True, is_relu=True)

    with tf.variable_scope('two_path_fusion'):
        layer = tf.pad(layer, [[0, 0], [1, 1], [1, 1], [0, 0]])
        output = featureFusion(layer, layer_4)
        output = tf.image.resize_bilinear(upsampling(output, scaling=8), read.resize_len)
        #output = conv2d(output, output_channels=64, is_relu=True)
        output = conv2d(output, output_channels=1, is_bn=True, is_relu=False)
        #output = tf.nn.tanh(output)

        if is_training is True:
            output = tf.nn.dropout(output, keep_prob=0.5)

        return output