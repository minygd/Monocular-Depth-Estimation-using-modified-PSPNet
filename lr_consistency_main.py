import tensorflow as tf
import numpy as np
import PSPNet_v3 as psp_v3
import ReadData as read
import scipy.misc as misc
import os
import time
import argparse
import left_right_loss as Loss

slim = tf.contrib.slim

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--batch_size', type=int, help='batch size', default=8)
parser.add_argument('--input_height', type=int, help='input height', default=512)
parser.add_argument('--input_width', type=int, help='input width', default=256)
args = parser.parse_args()

saver_dir = "your/path/to/save/logs/and/checkpoints"

adam_init_lr = 1e-5
adam_pretrained_init_lr = 1e-8
weight_decay = 2e-4
l2_regu_weight = 1e-5
lr_decay_step = 3
grad_clip_norm_value = 5

max_range = 101

input_width, input_height = args.input_height, args.input_width
input_channels = 3
batch_size = args.batch_size

def main(argv=None):
    tf.reset_default_graph()

    print("Setting up image reader...")
    train_dataset, valid_dataset, training_len, validation_len = read.read_data_v6('')
    train_dataset = train_dataset.shuffle(buffer_size=300).batch(batch_size).repeat()
    valid_dataset = valid_dataset.shuffle(buffer_size=300).batch(batch_size).repeat()
    train_iterator = train_dataset.make_initializable_iterator()
    valid_iterator = valid_dataset.make_initializable_iterator()
    print("train length: {}, valid length: {}".format(training_len, validation_len))

    print("Constructing network...")
    x_left = tf.placeholder(tf.float32, [None, input_height, input_width, input_channels], name='x_left')
    x_right = tf.placeholder(tf.float32, [None, input_height, input_width, input_channels], name='x_right')
    is_training = tf.placeholder(tf.bool, name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    #y_out = dorn.DORN(x, is_training=is_training)
    #y_out = psp.PSPNet(x, is_training=is_training)
    #y_out = psp_v2.PSPNet(x, is_training=is_training)
    disp_left = tf.clip_by_value(psp_v3.PSPNet(x_left, is_training=is_training, global_step=global_step), 0, 64)
    disp_right = tf.clip_by_value(psp_v3.PSPNet(x_right, is_training=is_training, global_step=global_step, reuse=True), 0, 64)
    disp_left_result = tf.add(disp_left, 0, name='disp_left')
    disp_right_result = tf.add(disp_right, 0, name='disp_right')

    tf.summary.image('x_left', x_left, 2)
    tf.summary.image('x_right', x_right, 2)
    tf.summary.image('disp_left', disp_left, 2)
    tf.summary.image('disp_right', disp_right, 2)

    mean_loss = Loss.calc_loss(x_left, x_right, disp_left, disp_right)

    adam_lr = tf.clip_by_value(tf.train.exponential_decay(adam_init_lr, global_step=global_step, decay_steps=lr_decay_step, decay_rate=0.5), adam_init_lr / 1000, adam_init_lr)
    adam_pretrained_lr = tf.clip_by_value(tf.train.exponential_decay(adam_pretrained_init_lr, global_step=global_step, decay_steps=lr_decay_step, decay_rate=0.5), adam_init_lr / 1000, adam_init_lr)

    with tf.variable_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=adam_lr)
        optimizer_pretrained = tf.train.AdamOptimizer(learning_rate=adam_pretrained_lr)

    # Resnet, Adam here is for previous models, please ignore them when using PSPNet
    # Pre-trained var is for the same purpose as above
    # Since keeping these lines does not cause error, they are not removed.
    var_num = 0
    include_var = []
    exclude_var = []
    for var in tf.trainable_variables():
        var_num += np.prod(np.array(var.get_shape().as_list()[1: ]))
        if 'resnet' not in var.name:
            if 'Adam' not in var.name:
                exclude_var.append(var)
            else:
                include_var.append(var)
        else:
            include_var.append(var)

    print("pretrained_var length: {}, random_var length: {}, parameter num: {}".format(len(include_var), len(exclude_var), int(var_num)))

    grads = optimizer.compute_gradients(mean_loss, var_list=exclude_var)
    if len(include_var) != 0:
        grads_pretrained = optimizer_pretrained.compute_gradients(mean_loss, var_list=include_var)

    for i, (g, v) in enumerate(grads):
        if g is not None:
            grads[i] = (tf.clip_by_norm(g, grad_clip_norm_value), v)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grads)
        if len(include_var) != 0:
            train_op_pretrained = optimizer_pretrained.apply_gradients(grads_pretrained)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter("/home/zhoukeyang/mono-depth/output/board", sess.graph)

    print("Setting up saver...")
    current_time = time.localtime()
    current_time_str = str(current_time[1]) + '-' + str(current_time[2]) + '-' + str(current_time[3]) + '-' + str(current_time[4])
    tf.train.write_graph(sess.graph_def, saver_dir + 'checkpoint-' + current_time_str + '/', "monodepth_model.pbtxt", as_text=True)
    saver = tf.train.Saver(tf.global_variables())

    print("Global variables initializing...")
    sess.run(train_iterator.initializer)
    sess.run(valid_iterator.initializer)
    sess.run(tf.global_variables_initializer())

    current_itr = 0
    if args.checkpoint_path != '':
        saver.restore(sess, args.checkpoint_path.split(".")[0])
        #init_fn = slim.assign_from_checkpoint_fn(model_path=args.checkpoint_path.split(".")[0], var_list=tf.trainable_variables(), ignore_missing_vars=True)
        #init_fn(sess)
        current_step = int(args.checkpoint_path.split(".")[0].split("-")[-1])
        global_step.assign(current_step)
        current_itr = current_step
        print("Training restored...")

    print("Begin running...")
    add_global = global_step.assign_add(1)

    step_per_iteration = training_len // batch_size
    count = 0

    next_bs = train_iterator.get_next()
    next_bs_valid = valid_iterator.get_next()

    for itr in range(current_itr + 1, max_range):
        init_loss_sum = 0
        for i in range(step_per_iteration):
            next_batch = sess.run(next_bs)
            train_images, train_corr_images = next_batch['image'], next_batch['image_corr']
            feed_dict = {x_left: train_images, x_right: train_corr_images, is_training: True}
            if len(include_var) != 0:
                sess.run(train_op_pretrained, feed_dict=feed_dict)
            init_loss, summary_str, lr, pred, _ = sess.run([mean_loss, summary_op, adam_lr, disp_left, train_op], feed_dict=feed_dict)
            if i % 200 == 0:
                print("Step: {} ---> Train_loss: {}".format(i, init_loss))
                summary_writer.add_summary(summary_str, count)
                count += 1
            init_loss_sum += init_loss

        sess.run(add_global)
        print("Iteration: {} ---> Train_loss: {}".format(itr, (init_loss_sum / step_per_iteration)))
        print("learning_rate: {}".format(lr))
        print("max_value: {}, min_value: {}, average: {}".format(np.max(pred), np.min(pred), np.mean(pred)))

        if itr % 1 == 0:
            next_batch = sess.run(next_bs_valid)
            valid_images, valid_corr_images = next_batch['image'], next_batch['image_corr']
            feed_dict = {x_left: valid_images, x_right: valid_corr_images, is_training: False}
            time_start = int(round(time.time() * 1000))
            init_loss, pred, summary_str, _ = sess.run([mean_loss, disp_left, summary_op, tf.no_op()], feed_dict=feed_dict)
            time_end = int(round(time.time() * 1000))

            print("{} images can be processed in 1 second.".format(int(1000 / (time_end - time_start) * batch_size * 2)))
            print("Iteration: {} ---> Validation_loss: {}".format(itr, init_loss))
            print("max_value: {}, min_value: {}, average: {}".format(np.max(pred), np.min(pred), np.mean(pred)))

        if itr % 10 == 0:
            saver.save(sess, saver_dir + 'checkpoint-' + current_time_str + '/', itr)


if __name__ == "__main__":
    tf.app.run()
