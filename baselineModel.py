import os
import sys
import gc
import time
import logging

import scipy.io as scio
import tensorflow as tf
import numpy as np
sys.path.append('../../../data/cifar')
import cifar_creat_data
from cifar_creat_data import read_and_decode

tf.app.flags.DEFINE_string('gpu', '2', 'set gpu')

tf.app.flags.DEFINE_string(
    'rootdir', './savefile/cifar10/DenseRes/increase4_n4_bs64_imgonce',
    'set root directory to save file')

tf.app.flags.DEFINE_boolean('loadprune', False, 'if load pruned parameters')

tf.app.flags.DEFINE_boolean('restore', False, 'set whether restore from file')

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
logger = logging.getLogger(__name__)


class Cifar_ResNet:
    def __init__(self):
        # parameter
        logger.info('\n' + '*' * 100 + '\n' + '******init******\n' + '*' * 100)
        self.cifar10 = False if 'cifar100' in FLAGS.rootdir else True
        self.model_file = FLAGS.rootdir + '/modelFile'
        self.snapshot_file = self.model_file + '/snapshot'
        self.datadir = '../../../data/cifar'
        if self.cifar10:
            self.trainfile = [
                self.datadir +
                '/all-cifar/cifar-10-batches-mat/cifar_train.tfrecord'
            ]
            self.testfile = [
                self.datadir +
                '/all-cifar/cifar-10-batches-mat/cifar_test.tfrecord'
            ]
            self.img_mean = self.datadir + '/all-cifar/cifar-10-batches-mat/cifar_mean.mat'
            logger.info('cifar10_init')
        else:
            self.trainfile = [
                self.datadir + '/cifar-100-python/cifar_train.tfrecord'
            ]
            self.testfile = [
                self.datadir + '/cifar-100-python/cifar_test.tfrecord'
            ]
            self.img_mean = self.datadir + '/cifar-100-python/cifar_mean.mat'
            logger.info('cifar100_init')

        # model structure
        if self.cifar10:
            self.num_classes = 10
        else:
            self.num_classes = 100

        self.ep = {}

        self.data_provider()
        self.model()
        self.train()
        self.test()

        self.sess = tf.Session()

        logger.info('\n' + '*' * 100 + '\n' + '****init done****\n' +
                    '*' * 100)

    def data_provider(self):
        logger.info('--------------Data Provider--------------')
        with tf.variable_scope('data'):
            logger.info('train : train data')
            self.img_train, label_train = read_and_decode(
                self.trainfile,
                self.batchsize,
                aug=['train'],
                img_mean=self.img_mean)
            self.label_train = tf.one_hot(label_train, self.num_classes)
            logger.info('test : test data')
            self.img_test, label_test = read_and_decode(
                self.testfile,
                self.acc_batchsize,
                aug=['test'],
                img_mean=self.img_mean)
            self.label_test = tf.one_hot(label_test, self.num_classes)
            logger.info('test : train data')
            self.img_test_train, label_test_train = read_and_decode(
                self.trainfile,
                self.acc_batchsize,
                aug=['test'],
                img_mean=self.img_mean)
            self.label_test_train = tf.one_hot(label_test_train,
                                               self.num_classes)

            self.img = tf.placeholder(tf.float32, [None, 32, 32, 3])
            self.label = tf.placeholder(tf.int32, [None, self.num_classes])

        return


def logset():
    logger.debug('Logger set')
    logger.setLevel(level=logging.INFO)

    path = os.path.dirname(FLAGS.rootdir)
    print('dirname: '+path)
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

    handler = logging.FileHandler(FLAGS.rootdir + '_logger.txt')

    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    return


if __name__ == '__main__':
    logset()
    a = Cifar_ResNet()
    # a.cal_distribute()
    # a.cal_computation()
    # logger.info(a.allcomp)
    a.run_whole()
