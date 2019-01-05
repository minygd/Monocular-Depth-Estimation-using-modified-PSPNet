import tensorflow as tf
import numpy as np
import bilinear_sampler as bs

slim = tf.contrib.slim

def generate_image_left(image, disp):
    return bs.bilinear_sampler_1d_h(image, -disp)

def generate_image_right(image, disp):
    return bs.bilinear_sampler_1d_h(image, disp)

def gradient_x(img):
    gx = img[:,:,:-1,:] - img[:,:,1:,:]
    return gx

def gradient_y(img):
    gy = img[:,:-1,:,:] - img[:,1:,:,:]
    return gy

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

    sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
    sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
    sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

def get_disparity_smoothness(image, disp):
    disp_gradient_x = gradient_x(disp)
    disp_gradient_y = gradient_y(disp)
    
    image_gradient_x = gradient_x(image)
    image_gradient_y = gradient_y(image)
    
    weight_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradient_x), 3, keep_dims=True))
    weight_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradient_y), 3, keep_dims=True))
    
    smoothness_x = tf.reduce_mean(weight_x * disp_gradient_x)
    smoothness_y = tf.reduce_mean(weight_y * disp_gradient_y)
    
    return smoothness_x + smoothness_y

def vertical_gradient_loss(disp):
    gy = gradient_y(disp)
    return tf.reduce_mean(tf.where(gy > 0, gy ** 2, 0)) ** 0.5

def calc_loss(left_input, right_input, left_disparity, right_disparity):
    right_generate_image = generate_image_right(left_input, left_disparity)
    left_generate_image = generate_image_left(right_input, right_disparity)

    right_generate_disp = generate_image_right(left_disparity, right_disparity)
    left_generate_disp = generate_image_left(right_disparity, left_disparity)

    right_smoothness_disp = get_disparity_smoothness(right_input, right_disparity)
    left_smoothness_disp = get_disparity_smoothness(left_input, left_disparity)
    
    right_reconstruction_loss = tf.reduce_mean(tf.abs(right_generate_image - right_input))
    left_reconstruction_loss = tf.reduce_mean(tf.abs(left_generate_image - left_input))
    reconstruction_loss = right_reconstruction_loss + left_reconstruction_loss
    
    right_SSIM_loss = tf.reduce_mean(tf.abs(SSIM(right_generate_image, right_input)))
    left_SSIM_loss = tf.reduce_mean(tf.abs(SSIM(left_generate_image, left_input)))
    SSIM_loss = right_SSIM_loss + left_SSIM_loss
    
    right_smoothness_loss = tf.reduce_mean(tf.abs(right_smoothness_disp))
    left_smoothness_loss = tf.reduce_mean(tf.abs(left_smoothness_disp))
    smoothness_loss = right_smoothness_loss + left_smoothness_loss
    
    lr_right_consistency_loss = tf.reduce_mean(tf.abs(right_generate_disp - right_disparity))
    lr_left_consistency_loss = tf.reduce_mean(tf.abs(left_generate_disp - left_disparity))
    lr_loss = lr_right_consistency_loss + lr_left_consistency_loss

    tf.summary.scalar('reconstruction_loss', reconstruction_loss)
    tf.summary.scalar('SSIM_loss', SSIM_loss)
    tf.summary.scalar('smoothness_loss', smoothness_loss)
    tf.summary.scalar('lr_loss', lr_loss)

    tf.summary.image('left_generate_image', left_generate_image, 2)
    tf.summary.image('right_generate_image', right_generate_image, 2)

    return reconstruction_loss * 0.7 + SSIM_loss * 0.3 + smoothness_loss + lr_loss

        
