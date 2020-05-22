'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np

import torchvision

from data_provider import data_provider
# from CIFARmodel.vgg import VGG
from ImageNetmodel.preact_resnet import PreActResNet, PreActBottleneck, PreActBlock
# from CIFARmodel.mobilenetv2 import MobileNetV2
from utils import progress_bar

import os
import argparse
import logging

# parser = argparse.ArgumentParser(
#     description='PyTorch CIFAR10 Training VGG19')
# parser.add_argument('--savefile', type=str, default='./savefile/cifar100/withoutbias/vgg19')
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training ResNet')
parser.add_argument('--savefile',
                    type=str,
                    default='./savefile/imagenet/stream/resnet18_bottleneck_prune_1')
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
    '/home/imagenet')
parser.add_argument('-gpu', type=str, default="0")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

logger = logging.getLogger(__name__)


class ImageNet_ResNet:
    def __init__(self):
        logger.info('\n' + '*' * 100 + '\n' + '******init******\n' + '*' * 100)
        self.dataset = 'imagenet'
        self.num_classes = 1000
        self.batchsize = 128
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info('device:'+self.device)
        self.savefile_checkpoint = args.savefile + '/checkpoint'
        self.max_epoch = 100
        self.test_every_k_epoch = 1

        self.choose_best_acc = False
        self.best_acc = 0  # best test accuracy
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.train_acc = 0
        self.test_acc = 0
        self.train_loss = 0

        self.train_data, self.test_data = data_provider(self.dataset,
                                                        args.data_path,
                                                        self.batchsize,
                                                        n_threads=16)
        # self.net = VGG('VGG19', self.num_classes)
        # arch = np.array([[64, 256, 2], [128, 512, 2], [256, 1024, 2],
        #                  [512, 2048, 2]])
        # arch = np.array([[74, 296, 2], [148, 592, 2], [296, 1184, 2],
        #                  [592, 2368, 2]])
        arch = np.array([[64, 64, 2], [128, 128, 2], [256, 256, 2], [512, 512, 2]])
        # arch = np.array([[77, 77, 2], [154, 154, 2], [307, 307, 2], [614, 614, 2]])
        # self.block = PreActBlock
        self.block = PreActBottleneck
        self.net = PreActResNet(self.block, arch, self.num_classes)
        # self.net = PreActResNet(PreActBlock, [2, 2, 2, 2], self.num_classes)
        # self.net = PreActResNet(PreActBottleneck, [3, 4, 6, 3],
        #                         self.num_classes)
        # self.net = MobileNetV2(num_classes=self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = 1.
        self.weight_decay = 1e-4
        self.lr_drop = [0, 60, 80]
        self.lr_weight = 10.
        logger.info('weight decay:' + str(self.weight_decay) + ', lr drop:' +
                    str(self.lr_drop))

        self.para, self.flop = self.net.cost()
        logger.info('Para:' + str(self.para) + ', Flops:' + str(self.flop))
        logger.info('weight decay:' + str(self.weight_decay) + ', lr drop:' +
                    str(self.lr_drop))

        self.stream_epoch = 40
        self.prune_times = 96
        self.base_prune = 0
        self.dimension1 = [0, 1, 2, 3]
        self.dimension2 = [0, 1]
        self.w_para = 0.5
        self.w_flop = 1. - self.w_para
        logger.info('stream epoch:' + str(self.stream_epoch) +
                    ', prune times:' + str(self.prune_times) +
                    ', prune base:' + str(self.base_prune) +
                    ', prune dimensions:' + str(self.dimension1) +
                    str(self.dimension2))

        self.stream_arch = {
            'arch': [arch],
            'para': [self.para],
            'flop': [self.flop],
            'cost': [0]
        }
        self.prenet = {
            'net': PreActResNet(self.block, arch, self.num_classes),
            'acc': 0,
            'cost': 0,
            'gate_set': np.array(self.net.gate_set),
            'para': self.para,
            'flop': self.flop
        }

        self.bestnet = {
            'net': PreActResNet(self.block, arch, self.num_classes),
            'acc': 0,
            'cost': 0,
            'gate_set': np.array(self.net.gate_set),
            'para': self.para,
            'flop': self.flop
        }

        self.current = {
            'net': self.net,
            'acc': 0,
            'cost': 0,
            'gate_set': self.net.gate_set,
            'para': self.para,
            'flop': self.flop
        }

    def run(self):
        def resume():
            # Load checkpoint.
            logger.info('==> Resuming from checkpoint..')
            assert os.path.exists(
                self.savefile_checkpoint), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(self.savefile_checkpoint + '/ckpt_best.pth')
            self.net.load_state_dict(checkpoint['net'])
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']
            # self.net.gate_set[:, :] = checkpoint['gate_set'][:, :]
            # self.net.gate_set[:, :] = 0
            # print(self.net.gate_set)
            # self.net._set_gate()

        def train(epoch):
            # logger.info('\nEpoch: %d' % epoch)
            self.net.train()
            train_loss = 0
            correct = 0
            correct5 = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(self.train_data):
                if epoch == self.stream_epoch and batch_idx > len(self.train_data)/5:
                    break
                # elif batch_idx > len(self.train_data)/200:
                #     break
                # inputs, targets = inputs.to(self.device), targets.to(
                #     self.device)
                inputs, targets = Variable(inputs.cuda()), Variable(
                    targets.cuda())
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                # _, predicted = outputs.max(1)
                total += targets.size(0)
                _, predicted = outputs.topk(5, 1, True, True)
                corr = predicted.eq(targets.view(-1, 1))
                correct += corr[:, :1].sum().item()
                correct5 += corr[:, :5].sum().item()
                # predicted = predicted.to('cpu')
                # corr = corr.to('cpu')
                # targets = targets.to('cpu')
                # print('predict', predicted.numpy())
                # print('corr', corr.numpy())
                # print('targets', targets.numpy())

                if batch_idx % 10 == 0 or batch_idx == len(self.train_data)-1:
                    progress_bar(
                        batch_idx, len(self.train_data),
                        'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                        (train_loss / (batch_idx + 1), 100. * correct / total,
                         correct, total))

            self.train_acc = correct / total
            self.train_acc5 = correct5 / total
            self.train_loss = train_loss / len(self.train_data)
            pass

        def test(epoch):
            global best_acc
            self.net.eval()
            test_loss = 0
            correct = 0
            correct5 = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(self.test_data):
                    # inputs, targets = inputs.to(self.device), targets.to(
                    #     self.device)
                    inputs, targets = Variable(inputs.cuda()), Variable(
                        targets.cuda())
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, targets)

                    test_loss += loss.item()
                    # _, predicted = outputs.max(1)
                    total += targets.size(0)
                    _, predicted = outputs.topk(5, 1, True, True)
                    corr = predicted.eq(targets.view(-1, 1))
                    correct += corr[:, :1].sum().item()
                    correct5 += corr[:, :5].sum().item()

                    if batch_idx % 10 == 0 or batch_idx == len(
                            self.test_data)-1:
                        progress_bar(
                            batch_idx, len(self.test_data),
                            'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                            (test_loss / (batch_idx + 1),
                             100. * correct / total, correct, total))

            # Save checkpoint.
            self.test_acc = correct / total
            self.test_acc5 = correct5 / total
            logger.info(
                'epoch: %d, loss: %f; accuracy: train: %f, %f, test: %f, %f' %
                (epoch, self.train_loss, self.train_acc, self.train_acc5,
                 self.test_acc, self.test_acc5))
            if self.test_acc > self.best_acc:
                logger.info('Save best model')
                self.best_acc = self.test_acc
                savemodel(epoch, 'best')
            if epoch == self.max_epoch:
                logger.info('Save final model')
                savemodel(epoch, 'final')

        def stream(epoch):
            def saveatob(a, b):
                b['acc'] = a['acc']
                b['cost'] = a['cost']
                b['gate_set'][:, :] = a['gate_set'][:, :]
                b['net'].load_state_dict(a['net'].state_dict())
                b['para'] = a['para']
                b['flop'] = a['flop']

            def densityofa(a, b):
                # return (b['acc'] - a['acc']) / (
                #     ((b['para'] - a['para'])**self.w_para) *
                #     ((b['flop'] - a['flop'])**self.w_flop))
                return (b['acc'] - a['acc']) / (((b['para']**self.w_para) *
                                                 (b['flop']**self.w_flop)) -
                                                ((a['para']**self.w_para) *
                                                 (a['flop']**self.w_flop)))

            def calstream(current, prenet):
                def cost(p0, f0, p1, f1):
                    return ((p0**self.w_para) *
                            (f0**self.w_flop)) - ((p1**self.w_para) *
                                                  (f1**self.w_flop))

                st = np.array(current['gate_set'])
                st[:] = -1
                for i in self.dimension1:
                    for j in self.dimension2:
                        st[i, j] = current['gate_set'][i, j]
                self.base_prune = np.min(st[st > 0]) * 0.2 + 1
                st[:] = 0
                s1, s2 = st.shape
                p0, f0 = current['net'].cost()
                for i in self.dimension1:
                    for j in self.dimension2:
                        current['gate_set'][i, j] -= self.base_prune
                        if current['gate_set'][i, j] < 0:
                            current['gate_set'][i, j] = 0
                        p1, f1 = current['net'].cost()
                        current['gate_set'][:, :] = prenet['gate_set'][:, :]
                        st[i, j] = cost(p0, f0, p1, f1)

                m = np.max(st)

                cur, pre = 0, 0
                for i in self.dimension1:
                    for j in self.dimension2:
                        k = 1
                        while True:
                            current['gate_set'][i, j] -= k
                            t = current['gate_set'][i, j] == 0
                            p1, f1 = current['net'].cost()
                            current['gate_set'][:, :] = prenet[
                                'gate_set'][:, :]
                            if t:
                                st[i, j] = k
                                break

                            cur = cost(p0, f0, p1, f1)
                            if cur == m:
                                st[i, j] = k
                                break
                            if cur > m:
                                if cur - m < m - pre:
                                    st[i, j] = k
                                else:
                                    st[i, j] = k - 1
                                break
                            if cur < m:
                                pre = cur
                            k += 1
                logger.info('prune of each part:' + str(st))
                return st

            prenet = self.prenet
            bestnet = self.bestnet
            current = self.current
            train(epoch)
            train(epoch)
            # test(epoch)
            current['para'], current['flop'] = self.net.cost()
            current['acc'] = self.train_acc
            saveatob(current, prenet)
            s1, s2 = current['gate_set'].shape
            st = calstream(current, prenet)
            print(s1, s2)
            for i in self.dimension1:
                for j in self.dimension2:
                    saveatob(prenet, current)
                    current['gate_set'][i, j] -= st[i, j]
                    if current['gate_set'][i, j] == 0:
                        continue
                    logger.info(self.net.gate_set)
                    self.net._set_gate()
                    for k in range(2):
                        train(epoch)
                    para, flop = self.net.cost()
                    current['para'], current['flop'] = para, flop
                    current['acc'] = self.train_acc
                    current['cost'] = densityofa(current, prenet)
                    logger.info(current['cost'])
                    # test(epoch)
                    # current['acc'] = self.test_acc
                    if bestnet[
                            'cost'] == 0 or current['cost'] < bestnet['cost']:
                        current['gate_set'][i, j] = int(
                            prenet['gate_set'][i, j] * 0.95 - 1)
                        current['para'], current['flop'] = current['net'].cost(
                        )
                        saveatob(current, bestnet)
                        logger.info('para:' + str(para) + ', flops:' +
                                    str(flop) + ', current acc:' +
                                    str(bestnet['acc']) + ', prenet acc:' +
                                    str(prenet['acc']))

            saveatob(bestnet, current)
            savemodel(epoch, 'stream')
            self.net._set_gate()
            self.stream_arch['arch'].append(np.array(bestnet['gate_set']))
            self.stream_arch['para'].append(bestnet['para'])
            self.stream_arch['flop'].append(bestnet['flop'])
            self.stream_arch['cost'].append(bestnet['cost'])
            bestnet['cost'] = 0
            logger.info(current['gate_set'])

        def show_arch():
            arch = np.array(self.stream_arch['arch'])
            s0, s1, s2 = arch.shape
            logger.info('arch:')
            for i in range(s1):
                for j in range(s2):
                    logger.info(arch[:, i, j])
            logger.info('parameter and flop:')
            logger.info(self.stream_arch['para'])
            logger.info(self.stream_arch['flop'])
            logger.info(self.stream_arch['cost'])

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
                    init.normal(m.weight, std=1e-3)
                    self.conv_weight += [m.weight]
                    self.bias += [m.bias]
                    init.constant(m.bias, 0)

        init_params()
        if args.resume:
            resume()

        logger.info('\n' + '*' * 100 + '\n' + '******Start training******\n' +
                    '*' * 100)
        # self.net = self.net.to(self.device)
        # self.net = nn.DataParallel(self.net)
        self.net = self.net.cuda()

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
                para, flop = self.net.cost()
                self.current['acc'] = self.train_acc
                self.current['cost'] = para
                while para**self.w_para * flop**self.w_flop > (
                        18100992.0**self.w_para) * (3557507072.0**self.w_flop):
                    # while para/self.para > 0.36/1.8:
                    # for k in range(self.prune_times):
                    logger.info('para rate:' + str(para / self.para) +
                                ',flop rate: ' + str(flop / self.flop))
                    stream(i)
                    para, flop = self.net.cost()
                show_arch()

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
    a = ImageNet_ResNet()
    a.run()