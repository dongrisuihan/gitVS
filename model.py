'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import numpy as np
from collections import OrderedDict

import torchvision

from data_provider import data_provider
from utils import progress_bar

import os
import argparse
import logging
from models.lenet import MyLeNet


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training ResNet')
parser.add_argument('--savefile',
                    type=str,
                    default='./savefile/mnist/test')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume',
                    '-r',
                    default=False,
                    action='store_true',
                    help='resume from checkpoint')
# parser.add_argument(
#     '--data_path',
#     type=str,
#     default=
#     '/home/czq/文档/code/mnist')
    parser.add_argument(
    '--data_path',
    type=str,
    default=
    '/data/home/chenzhiqiang/pytorch/code/data/mnist')
parser.add_argument('-gpu', type=str, default="0")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

logger = logging.getLogger(__name__)


class Cifar_VGG:
    def __init__(self):
        logger.info('\n' + '*' * 100 + '\n' + '******init******\n' + '*' * 100)
        self.dataset = 'mnist'
        self.batchsize = 64
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info('device:' + self.device)
        self.savefile_checkpoint = args.savefile + '/checkpoint'
        self.max_epoch = 100
        self.test_every_k_epoch = 1

        self.best_acc = 0  # best test accuracy
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.train_acc = 0

        self.train_data, self.test_data = data_provider(
            self.dataset, args.data_path, self.batchsize,download=True)
       
        self.net =MyLeNet()
       
        self.criterion = nn.CrossEntropyLoss()
        self.weight_decay = 1e-4
        self.lr_weight = 10.
        self.lr = 0.1
        self.lr_drop = [30,60,80,]

        logger.info('weight decay:' + str(self.weight_decay) + ', lr drop:' +
                    str(self.lr_drop))

        self.optimizer = optim.SGD(self.net.parameters(),
                                           lr=self.lr,
                                           momentum=0.9,
                                           weight_decay=self.weight_decay,
                                           nesterov=True)

    def run(self):
        def resume():
            # Load checkpoint.
            logger.info('==> Resuming from checkpoint..')
            assert os.path.exists(self.savefile_checkpoint
                                  ), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(self.savefile_checkpoint +
                                    '/ckpt_best.pth')
            self.net.load_state_dict(checkpoint['net'])
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']

        def train(epoch):
            # logger.info('\nEpoch: %d' % epoch)

            self.net.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(self.train_data):
                inputs, targets = inputs.to(self.device), targets.to(
                    self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if batch_idx % 10 == 0 or batch_idx == len(
                        self.train_data) - 1:
                    progress_bar(
                        batch_idx, len(self.train_data),
                        'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                        (train_loss / (batch_idx + 1), 100. * correct / total,
                         correct, total))

            self.train_acc = correct / total
            self.train_loss = train_loss / len(self.train_data)
            pass

        def test(epoch):
            global best_acc
            self.net.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(self.test_data):
                    inputs, targets = inputs.to(self.device), targets.to(
                        self.device)
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    if batch_idx % 10 == 0 or batch_idx == len(
                            self.test_data) - 1:
                        progress_bar(
                            batch_idx, len(self.test_data),
                            'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                            (test_loss / (batch_idx + 1),
                             100. * correct / total, correct, total))

            # Save checkpoint.
            self.test_acc = correct / total
            logger.info(
                'epoch: %d, loss: %f; accuracy: train: %f, test: %f' %
                (epoch, self.train_loss, self.train_acc, self.test_acc))
            if self.test_acc > self.best_acc:
                logger.info('Save best model')
                self.best_acc = self.test_acc
                savemodel(epoch, 'best')
            if epoch == self.max_epoch:
                logger.info('Save final model')
                savemodel(epoch, 'final')

        def savemodel(epoch, name='final'):
            logger.info('Saving...')
            state = {
                'net': self.net.state_dict(),
                'acc': self.test_acc,
                'epoch': epoch
            }
            if not os.path.exists(self.savefile_checkpoint):
                os.mkdir(self.savefile_checkpoint)
            torch.save(state,
                       self.savefile_checkpoint + '/ckpt_' + name + '.pth')

        def init_params(net=self.net):
            logger.info('Init layer parameters.')
            self.bias = []
            self.conv_weight = []
            self.bn_weight = []
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    # print(m.weight, m.bias)
                    init.kaiming_normal(m.weight, mode='fan_out')
                    self.conv_weight += [m.weight]
                    # self.bias += [m.bias]
                    # init.constant(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant(m.weight, 1.)
                    init.constant(m.bias, 0)
                    self.bn_weight += [m.weight]
                    self.bias += [m.bias]
                elif isinstance(m, nn.Linear):
                    # init.normal(m.weight, std=1e-3)
                    self.conv_weight += [m.weight]
                    self.bias += [m.bias]
                    init.constant(m.bias, 0)

        init_params()
        if args.resume:
            resume()

        logger.info('\n' + '*' * 100 + '\n' + '******Start training******\n' +
                    '*' * 100)
        self.net = self.net.to(self.device)

        for i in range(self.max_epoch + 1):
            if i in self.lr_drop:
                self.lr /= self.lr_weight
                logger.info('learning rate:' + str(self.lr))
                # self.optimizer = optim.SGD([{
                #     'params': self.conv_weight + self.bn_weight,

                #     'weight_decay': self.weight_decay
                # }],
                #                            lr=self.lr,
                #                            momentum=0.9,
                #                            weight_decay=self.weight_decay)
                self.optimizer = optim.SGD(self.net.parameters(),
                                           lr=self.lr,
                                           momentum=0.9,
                                           weight_decay=self.weight_decay,
                                           nesterov=True)

            if i >= self.start_epoch:
                train(i)
                if i % self.test_every_k_epoch == 0 or i == self.max_epoch:
                    logger.info('test')
                    test(i)

        pass


def logset():
    logger.debug('Logger set')
    logger.setLevel(level=logging.INFO)

    path = os.path.dirname(args.savefile)
    print('dirname: ' + args.savefile)
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(args.savefile):
        os.makedirs(args.savefile)

    handler = logging.FileHandler(args.savefile + '_logger.txt')

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
    a = Cifar_VGG()
    a.run()