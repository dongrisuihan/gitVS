'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch.nn.init as init


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, arch, stride=1, gate=[], dp=0.05):
        super(PreActBlock, self).__init__()
        self.gate = gate
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, arch[0], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(arch[0])
        self.conv2 = nn.Conv2d(arch[0], arch[1], kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != arch[1]:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, arch[1], kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = out * self.gate[0]
        out = self.conv2(F.relu(self.bn2(out)))
        out = out * self.gate[2]
        out += shortcut
        out = out * self.gate[1]
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, arch, stride=1, gate=[], dp=0.05):
        super(PreActBottleneck, self).__init__()
        self.gate = gate
        # self.dp1 = nn.Dropout(dp)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, arch[0], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(arch[0])
        self.conv2 = nn.Conv2d(arch[0], arch[0], kernel_size=7, stride=stride, padding=3, groups=int(arch[0]), bias=False)
        self.bn3 = nn.BatchNorm2d(arch[0])
        self.conv3 = nn.Conv2d(arch[0], arch[1], kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(arch[1])

        if stride != 1 or in_planes != arch[1]:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, arch[1], kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(arch[1])
            )

    def forward(self, x):
        out = F.relu(x)
        # out = self.dp1(out)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        # out = self.dp1(out)
        out = out * self.gate[0]
        # out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv2(out)
        out = out * self.gate[0]
        out = self.conv3(F.relu(self.bn3(out)))
        out = self.bn4(out)
        out *= self.gate[2]
        out += shortcut
        out = out * self.gate[1]
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, archs, num_classes=10, dp=0.05):
        super(PreActResNet, self).__init__()
        self.num_classes = num_classes
        # planes, shortcut_width, depth
        self.arch_set = np.array(archs)
        self.gate_set = np.array(archs)
        self.block = block
        self.dp = dp
        # self.arch_set = np.array([[16, 16 * block.expansion, num_blocks[0]],
        #                           [32, 32 * block.expansion, num_blocks[1]],
        #                           [64, 64 * block.expansion, num_blocks[2]]])
        # self.gate_set = np.array([[16, 16 * block.expansion, num_blocks[0]],
        #              [32, 32 * block.expansion, num_blocks[1]],
        #              [64, 64 * block.expansion, num_blocks[2]]])
        print(self.gate_set)
        self._make_gate()
        self.in_planes = self.arch_set[0, 1]

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.arch_set[0], stride=1, gate=self.gate[0])
        self.layer2 = self._make_layer(block, self.arch_set[1], stride=2, gate=self.gate[1])
        self.layer3 = self._make_layer(block, self.arch_set[2], stride=2, gate=self.gate[2])

        self.bn = nn.BatchNorm2d(self.arch_set[2, 1])
        self.linear = nn.Linear(self.arch_set[2, 1], num_classes)

    def _make_layer(self, block, arch, stride, gate):
        strides = [stride] + [1]*(arch[2]-1)
        layers = []
        for i, stride in enumerate(strides):
            g = [gate[0], gate[1], gate[2][0, i, 0, 0]]
            # print(g)
            b = block(self.in_planes, arch, stride, g)
            layers.append(b)
            self.in_planes = arch[1]
        return nn.Sequential(*layers)

    def _make_gate(self, device='cuda'):
        self.gate = []
        for gs in self.arch_set:
            gate = []
            for g in gs:
                gate.append(torch.ones(1, g, 1, 1).to(device))
            self.gate.append(gate)

    def _set_gate(self):
        for gs in self.gate:
            for g in gs:
                init.constant(g, 0.)
                # print(g)
        for i, gs in enumerate(self.gate):
            for j, g in enumerate(gs):
                g[0, 0:self.gate_set[i, j], 0, 0] = 1
                # print(g)

    def cost(self):
        if self.block == PreActBottleneck:
            para = 0
            flop = 0
            a = 32 * 32
            gate = self.gate_set
            for g in gate:
                p = g[1] * g[0] * 2 + g[0] * 7 * 7
                p = p * g[2] - g[1] * g[0]
                f = p * a
                para += p
                flop += f
                a /= 4
            p00 = gate[0][1] * gate[0][1] + gate[0][1] * 3 * 3 * 3
            f00 = 32 * 32 * p00
            p01 = gate[0][1] * gate[1][1] * 2
            f01 = (32 * 32 + 16 * 16) * gate[0][1] * gate[1][1]
            p12 = gate[1][1] * gate[2][1] * 2
            f12 = (16 * 16 + 8 * 8) * gate[1][1] * gate[2][1]
            p22 = gate[2][1] * self.num_classes
            f22 = p22
            para += p00 + p01 + p12 + p22
            flop += f00 + f01 + f12 + f22
            return para, flop
        elif self.block == PreActBlock:
            para = 0
            flop = 0
            a = 32 * 32
            gate = self.gate_set
            for g in gate:
                p = g[0] * g[1] * 2 * 3 * 3
                p = p * g[2] - g[0] * g[1] * 3 * 3
                f = p * a
                para += p
                flop += f
                a /= 4
            p00 = gate[0][1] * (gate[0][1] + 3) * 3 * 3
            f00 = 32 * 32 * p00
            p01 = gate[0][1] * gate[1][1] * (3 * 3 + 1)
            f01 = 16 * 16 * p01
            p12 = gate[1][1] * gate[2][1] * (3 * 3 + 1)
            f12 = 8 * 8 * p12
            p22 = gate[2][1] * self.num_classes
            f22 = p22
            para += p00 + p01 + p12 + p22
            flop += f00 + f01 + f12 + f22
            return para, flop

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out *= self.gate[0][1]
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet32():
    return PreActResNet(PreActBlock, [5, 5, 5])


def PreActResNet44():
    return PreActResNet(PreActBlock, [7, 7, 7])


def PreActResNet50():
    return PreActResNet(PreActBottleneck, [9, 9, 9])


def PreActResNet101():
    return PreActResNet(PreActBottleneck, [18, 18, 18])


def test():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    net = PreActResNet101()
    net.to('cuda')
    net._make_gate()
    net._set_gate()
    print(net.cost())
    # print(net.gate[0][2])
    # for i, m in enumerate(net.modules()):
    #     if isinstance(m, nn.Conv2d):
    #     # if isinstance(m, nn.Sequential):
    #         if hasattr(m, 'bias'):
    #             print(i, m)
    #             print(m.bias)
    y = net((torch.ones(64, 3, 32, 32).to('cuda')))
    # y = y * torch.ones(1)
    # print(y[0, 1])

# test()
if __name__ == "__main__":
    test()
