import tensorflow as tf

stddev = 0.01
leaky_relu_alpha = 0.2
kernel_init = None

def down_sampler_block(input, output_channels=None, id=None):
    with tf.variable_scope('down_sampler_block_' + str(id), reuse=False):
        if output_channels == None:
            output_channels = input.get_shape()[-1] * 2
        output = tf.contrib.layers.conv2d(input, num_outputs=output_channels - input.get_shape().as_list()[-1], kernel_size=3, stride=2, padding='same')
        output_2 = tf.contrib.layers.max_pool2d(input, kernel_size=3, stride=2, padding='same')
        output = tf.concat([output, output_2], axis=-1)
        #output = tf.contrib.layers.batch_norm(output)
        return output

def non_bottle_neck_block(input, dilated_rate, is_dropout=True, id_1=None, id_2=None, kernel_size=3):
    with tf.variable_scope('non_bottle_neck_block_' + str(id_1) + '_' + str(id_2), reuse=False):
        channels = input.get_shape().as_list()[-1]
        output = tf.contrib.layers.conv2d(input, num_outputs=channels, kernel_size=(3, 1), stride=1, padding='same')
        #output = tf.contrib.layers.batch_norm(output)
        output = tf.contrib.layers.conv2d(output, num_outputs=channels, kernel_size=(1, 3), stride=1, padding='same')
        output = tf.contrib.layers.batch_norm(output)
        #output = tf.nn.leaky_relu(output)

        output = tf.contrib.layers.conv2d(output, num_outputs=channels, kernel_size=(kernel_size, 1), stride=1, padding='same')
        #output = tf.contrib.layers.batch_norm(output)
        output = tf.contrib.layers.conv2d(output, num_outputs=channels, kernel_size=(1, kernel_size), stride=1, padding='same')
        output = tf.contrib.layers.batch_norm(output)

        if is_dropout is True:
            output = tf.nn.dropout(output, keep_prob=0.9)
        return tf.nn.leaky_relu(output + input, alpha=leaky_relu_alpha)

def bottle_neck_block(input, dilated_rate, is_dropout=True, id_1=None, id_2=None, kernel_size=3):
    with tf.variable_scope('bottle_neck_block_' + str(id_1) + '_' + str(id_2), reuse=False):
        channels = input.get_shape().as_list()[-1]
        output = tf.contrib.layers.conv2d(input, num_outputs=channels, kernel_size=kernel_size, stride=1, padding='same')
        output = tf.contrib.layers.batch_norm(output)
        output = tf.contrib.layers.conv2d(output, num_outputs=channels, kernel_size=kernel_size, stride=1, padding='same')
        output = tf.contrib.layers.batch_norm(output)
        if is_dropout is True:
            output = tf.nn.dropout(output, keep_prob=0.9)
        return tf.nn.leaky_relu(output + input, alpha=leaky_relu_alpha)

def encoder(input, output_channels=128):
    output_1 = down_sampler_block(input, output_channels=32, id=1)
    for i in range(1):
        output_1 = non_bottle_neck_block(output_1, dilated_rate=1, id_1=1, id_2=i)
    """
    output_2 = down_sampler_block(output_1, output_channels=32, id=2)
    for i in range(2):
        output_2 = bottle_neck_block(output_2, dilated_rate=1, id_1=2, id_2=i)
    """
    output_3 = down_sampler_block(output_1, output_channels=64, id=3)
    for i in range(2):
        output_3 = non_bottle_neck_block(output_3, dilated_rate=1, id_1=3, id_2=i)

    output = down_sampler_block(output_3, output_channels=output_channels, id=4)
    for i in range(1):
        output = non_bottle_neck_block(output, dilated_rate=1, id_1=4, id_2=i, kernel_size=3)
        output = non_bottle_neck_block(output, dilated_rate=1, id_1=5, id_2=i, kernel_size=3)
        output = non_bottle_neck_block(output, dilated_rate=1, id_1=6, id_2=i, kernel_size=5)
        #output = non_bottle_neck_block(output, dilated_rate=1, id_1=7, id_2=i, kernel_size=7)

    tf.summary.image('2*downsampling', output_1[:, :, :, :1], 2)
    tf.summary.image('4*downsampling', output_3[:, :, :, :1], 2)
    tf.summary.image('8*downsampling', output[:, :, :, :1], 2)

    return output, output_3, output_1

def up_sampler_block(input, output_channels, id=None):
    with tf.variable_scope('up_sampler_block_' + str(id), reuse=False):
        output = unpool(input)
        output = tf.contrib.layers.conv2d_transpose(output, num_outputs=output_channels, kernel_size=3, stride=1, padding='same')
        #output = tf.contrib.layers.conv2d_transpose(input, num_outputs=output_channels, kernel_size=3, stride=2, padding='same')
        #output = tf.contrib.layers.batch_norm(output)
        #output = tf.nn.leaky_relu(output, alpha=leaky_relu_alpha)
        return output

def unpool(value, name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    sh = value.get_shape().as_list()
    dim = len(sh[1: -1])
    out = (tf.reshape(value, [-1] + sh[-dim:]))
    for i in range(dim, 0, -1):
        out = tf.concat([out, tf.zeros_like(out)], i)
    out_size = [-1] + [s * 2 for s in sh[1: -1]] + [sh[-1]]
    out = tf.reshape(out, out_size)
    return out

def decoder(input, output_channels, middle_layer):
    output_1 = up_sampler_block(input, output_channels=64, id=1)
    #output_1 = tf.concat([up_sampler_block(input, output_channels=64, id=1), middle_layer[0]], axis=-1)
    #output_1 = up_sampler_block(input, output_channels=64, id=1) + middle_layer[0]
    output_1 = non_bottle_neck_block(output_1, is_dropout=False, dilated_rate=1, id_1=8, id_2=0)
    tf.summary.image('2*upsampling', output_1[:, :, :, :1], 2)

    output_2 = up_sampler_block(output_1, output_channels=32, id=2)
    #output_2 = tf.concat([up_sampler_block(output_1, output_channels=32, id=2), middle_layer[1]], axis=-1)
    #output_2 = up_sampler_block(output_1, output_channels=32, id=2) + middle_layer[1]
    output_2 = non_bottle_neck_block(output_2, is_dropout=False, dilated_rate=1, id_1=9, id_2=0)
    tf.summary.image('4*upsampling', output_2[:, :, :, :1], 2)

    output = up_sampler_block(output_2, output_channels=output_channels, id=3)
    tf.summary.image('8*upsampling', output[:, :, :, :1], 2)

    shape = output.get_shape().as_list()[1: -1]
    output_1 = tf.image.resize_bilinear(output_1, shape, align_corners=True)
    output_2 = tf.image.resize_bilinear(output_2, shape, align_corners=True)

    return output_1, output_2, output

def construct_network_v2(input, is_training=True):
    with tf.variable_scope('encoder', reuse=False):
        output, output_2, output_1 = encoder(input, output_channels=128)
        #output_0 = tf.contrib.layers.conv2d(output, num_outputs=1, kernel_size=1, stride=1, padding='same')
        #output_0 = tf.image.resize_bilinear(output_0, tf.Variable([8 * output.get_shape().as_list()[1], 8 * output.get_shape().as_list()[2]], trainable=False))

    with tf.variable_scope('decoder', reuse=False):
        for i in range(0):
            output_1 = non_bottle_neck_block(output_1, dilated_rate=1, is_dropout=True, id_1=10, id_2=i)
            output_2 = non_bottle_neck_block(output_2, dilated_rate=1, is_dropout=True, id_1=11, id_2=i)
        output_1, output_2, output = decoder(output, output_channels=16, middle_layer=[output_2, output_1])
        #output_1, output_2, output = decoder_v2(output, output_channels=16, middle_layer=[output_2, output_1])
        output = tf.contrib.layers.conv2d(output, num_outputs=1, kernel_size=1, stride=1, padding='same')
        output = tf.contrib.layers.batch_norm(output)
        output = tf.nn.leaky_relu(output)

    #print(input.get_shape(), output_1.get_shape(), output_2.get_shape(), output.get_shape())
    #output_1 = tf.contrib.layers.conv2d(output_1, num_outputs=1, kernel_size=1, stride=1, padding='same')
    #output_2 = tf.contrib.layers.conv2d(output_2, num_outputs=1, kernel_size=1, stride=1, padding='same')
    return output_1, output_2, output
