from numpy import *
import numpy as np
import os
import sys
import tensorflow as tf
import scipy.io as scio
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

slim = tf.contrib.slim


def mnist_process(img, batchsize, aug=['train']):

    img = dataAug(img, batchsize, aug)

    return img


def dataAug(img, batchsize, aug=['train']):
    img = tf.reshape(img, [batchsize, 28, 28, 1])
    if 'sub_mean' in aug:
        mean = scio.loadmat('MNIST_data/mean.mat')
        mean = mean['mean']
        mean = mean[np.newaxis, :, :, :]
        # print(mean.shape)
        img = img - mean

    if 'train' in aug:
        img = tf.image.resize_image_with_crop_or_pad(img, 32, 32)
        img = tf.random_crop(img, [batchsize, 28, 28, 1])

    if 'test' in aug:
        pass

    img = tf.cast(img, tf.float32)
    return img


def cal_mean():
    img, _ = mnist.train.next_batch(60000)
    print(img.shape)
    print(img.dtype)
    print(max(img[1, :]))
    img_mean = mean(img, axis=0)
    print(img_mean.shape)
    print(max(img_mean))
    img_mean.shape = (1, 784)
    img_var=(img-img_mean)**2
    img_var = mean(img_var, axis=0)
    img_mean.shape = (28, 28, 1)
    img_var.shape = (28,28,1)
    print(img_mean.shape)
    scio.savemat('MNIST_data/mean.mat', {'mean': img_mean})
    scio.savemat('MNIST_data/var.mat', {'var': img_var})
    # print(img_mean)


def read_full_process():
    bs = 5
    _img_train = tf.placeholder(tf.float32, [None, 784])
    img_train = mnist_process(_img_train, bs, aug=['train', 'sub_mean'])
    label_train = tf.placeholder(tf.int32, [None, 10])
    _img_test = tf.placeholder(tf.float32, [None, 784])
    img_test = mnist_process(_img_test, bs, aug=['test'])
    label_test = tf.placeholder(tf.int32, [None, 10])

    img = tf.placeholder(tf.float32, [None, 28, 28, 1])
    label = tf.placeholder(tf.int32, [None, 10])

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        tf.global_variables_initializer().run()
        for i in range(2):
            print(i)
            _img, _label = mnist.train.next_batch(bs)
            _img = sess.run(img_train, feed_dict={_img_train: _img})

            print(_img.shape)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # read_full_process()
    cal_mean()
