'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import numpy as np

import torchvision

from data_provider import data_provider
# from CIFARmodel.vgg import VGG
from CIFARmodel.preact_resnet import PreActResNet, PreActBottleneck, PreActBlock
# from CIFARmodel.mobilenetv2 import MobileNetV2
from utils import progress_bar

import os
import argparse
import logging

# parser = argparse.ArgumentParser(
#     description='PyTorch CIFAR10 Training VGG19')
# parser.add_argument('--savefile', type=str, default='./savefile/cifar100/withoutbias/vgg19')
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training ResNet')
parser.add_argument('--savefile',
                    type=str,
                    default='./savefile/cifar100/steam/ResnetWithourBottleneck/res32_gradually')
# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training MobileNetV2')
# parser.add_argument('--savefile', type=str, default='./savefile/cifar100/withourbias/MobileNet')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume',
                    '-r',
                    default=False,
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument(
    '--data_path',
    type=str,
    default=
    '/data/home/chenzhiqiang/tensorflow/KnowledgeDistill/data/cifar/all-cifar')
parser.add_argument('-gpu', type=str, default="0")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

logger = logging.getLogger(__name__)


class Cifar_VGG:
    def __init__(self):
        logger.info('\n' + '*' * 100 + '\n' + '******init******\n' + '*' * 100)
        self.dataset = 'cifar100' if 'cifar100' in args.savefile else 'cifar10'
        if self.dataset == 'cifar10':
            self.num_classes = 10
        else:
            self.num_classes = 100
        self.batchsize = 128
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info('device:'+self.device)
        self.savefile_checkpoint = args.savefile + '/checkpoint'
        self.max_epoch = 200
        self.test_every_k_epoch = 1

        self.choose_best_acc = False
        self.best_acc = 0  # best test accuracy
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.train_acc = 0
        self.test_acc = 0
        self.train_loss = 0
        arch = np.array([[16, 16, 5], [32, 32, 5], [64, 64, 5]])
        arch[:, 0:2] *= 2
        self.target_arch = np.array([[16, 16, 5], [32, 32, 5], [64, 64, 5]])
        self.target_times = 8
        self.gradual_arch = None
        # # CIFAR 10
        # # para 0.75
        # arch = np.array([[14, 26, 5], [24, 47, 5], [52, 60, 5]])
        # # para 0.5
        # arch = np.array([[12, 18, 5], [27, 36, 5], [65, 60, 5]])
        # # para 0.25
        # arch = np.array([[8, 16, 5], [21, 39, 5], [60, 87, 5]])
        # 6M
        # arch = np.array([[28, 192, 18], [78, 384, 18], [125, 322, 18]])
        # arch = np.array([[19, 181, 18], [42, 384, 18], [142, 380, 18]])
        # # 4.1M
        # arch = np.array([[41, 257, 5], [100, 415, 5], [206, 281, 5]])
        # # 2.5M
        # arch = np.array([[35, 164, 5], [70, 372, 5], [168, 224, 5]])
        # arch = np.array([[30, 116, 5], [62, 298, 5], [164, 392, 5]])
        # # CIFAR 100
        # # para 0.75
        # arch = np.array([[16, 23, 5], [24, 30, 5], [34, 106, 5]])
        # # para 0.5
        # arch = np.array([[10, 16, 5], [16, 33, 5], [44, 120, 5]])
        # # para 0.25
        # arch = np.array([[8, 8, 5], [14, 36, 5], [65, 120, 5]])

        self.train_data, self.test_data = data_provider(
            self.dataset, args.data_path, self.batchsize)
        # self.net = VGG('VGG19', self.num_classes)
        # self.block = PreActBottleneck
        self.block = PreActBlock
        self.net = PreActResNet(self.block, arch, self.num_classes)
        # self.net = MobileNetV2(num_classes=self.num_classes)
        self.para, self.flop = self.net.cost()
        logger.info('Para:' + str(self.para) + 'Flops:' + str(self.flop))
        self.criterion = nn.CrossEntropyLoss()
        self.warmup = 0
        self.weight_decay = 1e-4
        self.lr = 1.
        self.lr_drop = [0, 120, 160, 180]
        self.lr_weight = 10.
        # self.lr = 0.2
        # self.lr_drop = [0, 160, 180]
        # self.lr_weight = 2.
        logger.info('weight decay:' + str(self.weight_decay) + ', lr drop:' +
                    str(self.lr_drop))

        self.stream_epoch = 80
        self.prune_times = 96
        self.base_prune = 0
        self.dimension1 = [0, 1, 2]
        self.dimension2 = [0, 1]
        self.w_para = 0.5
        self.w_flop = 1. - self.w_para
        logger.info('stream epoch:' + str(self.stream_epoch) +
                    ', prune times:' + str(self.prune_times) +
                    ', prune base:' + str(self.base_prune) +
                    ', prune dimensions:' + str(self.dimension1) +
                    str(self.dimension2))

    def run(self):

        def resume():
            # Load checkpoint.
            logger.info('==> Resuming from checkpoint..')
            assert os.path.exists(
                self.savefile_checkpoint), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(self.savefile_checkpoint + '/ckpt_final.pth')
            self.net.load_state_dict(checkpoint['net'])
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']
            self.net.gate_set[:, :] = checkpoint['gate_set'][:, :]
            self.net.gate_set[:, :] = 0
            print(self.net.gate_set)
            self.net._set_gate()

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

                if batch_idx % 10 == 0 or batch_idx == len(self.train_data)-1:
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
                            self.test_data)-1:
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

        def gradually(epoch):
            if self.gradual_arch == None:
                to_prune = np.array(self.net.gate_set)
                to_prune[:, :] = (self.net.gate_set - self.target_arch) / self.target_times
                for l in range(self.target_times):
                    for i in self.dimension1:
                        for j in self.dimension2:
                            self.net.gate_set[i, j] -= to_prune[i, j]
                            self.net._set_gate()
                            for k in range(4):
                                train(epoch)
                            test(epoch)

        def savemodel(epoch, name='final'):
            logger.info('Saving...')
            state = {
                    'net': self.net.state_dict(),
                    'acc': self.test_acc,
                    'epoch': epoch,
                    'gate_set': self.net.gate_set
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
                    init.constant(m.weight, 0.5)
                    init.constant(m.bias, 0)
                    self.bn_weight += [m.weight]
                    self.bias += [m.bias]
                elif isinstance(m, nn.Linear):
                    init.normal(m.weight, std=1e-3)
                    self.conv_weight += [m.weight]
                    self.bias += [m.bias]
                    init.constant(m.bias, 0)

        init_params()
        if args.resume:
            resume()

        logger.info('\n' + '*' * 100 + '\n' + '******Start training******\n' +
                    '*' * 100)
        self.net = self.net.to(self.device)
        for i in range(self.warmup):
            self.optimizer = optim.SGD(self.net.parameters(),
                                       lr=0.01,
                                       momentum=0.9,
                                       weight_decay=self.weight_decay)
            train(0)

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
                                           weight_decay=self.weight_decay)

            if i >= self.start_epoch:
                train(i)
                if i % self.test_every_k_epoch == 0 or i == self.max_epoch:
                    logger.info('test')
                    test(i)
            if i == self.stream_epoch:
                gradually(i)
                # show_arch()
                # for j in range(30):
                #     while flop/self.flop > 0.5**((j+1.)/30.):
                #         stream(i)
                #         para, flop = self.net.cost()
                #         train(i)
                #         train(i)

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