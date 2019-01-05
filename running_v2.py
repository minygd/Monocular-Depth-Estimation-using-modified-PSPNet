import tensorflow as tf
import numpy as np
import PSPNet_v3 as psp_v3
from PIL import Image
import cv2

slim = tf.contrib.slim

checkpoint_filename = 'checkpoint-12-19-16-10/'

def cvtGray2RGB(image):
    return cv2.applyColorMap(image, cv2.COLORMAP_RAINBOW)

def main(argv=None):
    #video_capture = cv2.VideoCapture('test4.mp4')
    video_capture = cv2.VideoCapture('/home/zhoukeyang/桌面/test_video/day video 2.avi')
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(fps, size)
    fps = 20

    #video_writer = cv2.VideoWriter('other.mp4', cv2.VideoWriter_fourcc(*'MPEG'), fps, (512, 256))

    with tf.Session() as sess:
        print("initialize tensor...")
        #graph = tf.get_default_graph()
        x = tf.placeholder(tf.float32, [1, 256, 512, 3], name='x')
        is_training = tf.placeholder(tf.bool, name='is_training')
        y_out = psp_v3.PSPNet(x, is_training=is_training)

        print("restore model...")
        #saver = tf.train.import_meta_graph(checkpoint_filename + '-60.meta')
        #saver.restore(sess, tf.train.latest_checkpoint(checkpoint_filename))
        init_fn = slim.assign_from_checkpoint_fn(model_path=checkpoint_filename + '-10', var_list=tf.global_variables(), ignore_missing_vars=True)
        init_fn(sess)

        print("begin running...")
        success, frame = video_capture.read()

        for i in range(1):
            while success:
                frame = np.expand_dims(cv2.resize(frame, (512, 256)), axis=0)
                feed_dict = {x: frame, is_training: False}
                pred = sess.run(y_out, feed_dict=feed_dict)
                result = cvtGray2RGB(np.squeeze(pred[0] / np.mean(pred[0]) * 128).astype(np.uint8))
                #pred = pred[: int(0.95 * pred.shape[0]), :, :]
                result = np.squeeze((pred[0] - np.min(pred[0])) / (np.max(pred[0]) - np.min(pred[0])) * 255).astype(np.uint8)
                #result = np.squeeze(pred[0] * 2550).astype(np.uint8)
                result[result > 255] = 255
                result[result < 0] = 0
                result = cvtGray2RGB(result)
                cv2.namedWindow("pred")
                cv2.imshow("pred", np.concatenate([result, np.squeeze(frame)], axis=0))
                cv2.waitKey(1000 // fps)
                #video_writer.write(result)
                success, frame = video_capture.read()


if __name__ == '__main__':
    tf.app.run()
