import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re

class Gate(nn.Module):
    def __init__(self, num_channels):
        super(Gate, self).__init__()
        self.num_channels = num_channels
        self.gate = nn.Parameter(torch.Tensor(1, num_channels, 1, 1), requires_grad=False)

    def forward(self, input_tensor):
        output = input_tensor * self.gate
        return output

    def extra_repr(self):
        s = ('{num_channels}')
        return s.format(**self.__dict__)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, planes, stride):
        super(Block, self).__init__()
        self.stride = stride

        # planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.g1 = Gate(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.g2 = Gate(planes)

        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.g3 = Gate(out_planes)

        self.shortcut = stride == 1 and in_planes == out_planes
    

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.g1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.g2(out)
        out = self.bn3(self.conv3(out))
        if self.shortcut:
            out = out + x
        return self.g3(out)


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    # cfg = [(1,  16, 1, 1),
    #        (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
    #        (6,  32, 3, 2),
    #        (6,  64, 4, 2),
    #        (6,  96, 3, 1),
    #        (6, 160, 3, 2),
    #        (6, 320, 1, 1)]

    def __init__(self, arch, num_classes=10, small_input=True):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        # assert arch.shape[0] == 7
        self.gate_set = arch
    
        self.num_classes = num_classes
        self.fix_cfg = np.array([[1, 1],
                        [2, 1],
                        [3, 2],
                        [4, 2],
                        [3, 1],
                        [3, 2],
                        [1, 1]])
        
        self.arch_P = [[2,3,4], [2,3], [1,2]]
        
        self.avgpool_size = 4
        first_stride = 1
        self.fmap_size = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 16, 16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

        if not small_input:
            self.fix_cfg[0][1] = 2
            self.avgpool_size = 7
            first_stride = 2
            self.fmap_size = [112, 112, 56, 56, 56, 56, 56, 56, 56, 56, 56, 28, 28, 28, 28, 28, 28, 28, 28, 28, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]

        arch = self.__real_arch_form()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=first_stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(arch, in_planes=32)
        self.conv2 = nn.Conv2d(arch[-1][1], 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, Gate):
                nn.init.ones_(m.gate)

    def _make_layers(self, arch, in_planes):
        layers = nn.Sequential()
        for i, cfg in enumerate(zip(arch, self.fix_cfg)):
            arch_set, fc = cfg
            bottle_size = arch_set[0]
            out_planes = arch_set[1]
            num_blocks = fc[0]
            stride = fc[1]
            strides = [stride] + [1]*(num_blocks-1)
            for j, stride in enumerate(strides):
                layers.add_module('Block{}-{}'.format(i, j), Block(in_planes, out_planes, bottle_size, stride))
                in_planes = out_planes
        return layers

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, self.avgpool_size)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    
    def set_gate(self):
        for name, param in self.named_parameters():
            if 'gate' in name:
                param.data.fill_(0)

        gate_set = self.__real_arch_form()
        for i, gs in enumerate(gate_set):
            for name, param in self.named_parameters():
                if 'Block{}'.format(i) in name:
                    if 'g3' in name:
                        param.data[:, :gs[1], :, :] = 1
                    elif 'g1' in name or 'g2' in name:
                        param.data[:, :gs[0], :, :] = 1


    def cost(self, gate_set=None):

        gs = self.__real_arch_form(arch_set=gate_set)

        total_ops = 0
        total_param, conv_param, bn_param = self.__get_param(gs)
    
        for fs, conv, bn in zip(self.fmap_size, conv_param, bn_param):
            total_ops += fs ** 2 * conv
            total_ops += fs ** 2 * bn

        total_ops += self.num_classes * 1280

        return total_param, total_ops


    def __real_arch_form(self, arch_set=None):
    
        if arch_set is None:
            arch = self.gate_set
        else:
            arch = arch_set

        assert arch.shape == (3,2)

        real_arch = []
        for i, ps in enumerate(self.arch_P):
            for p in ps:
                real_arch.append([arch[i][0]*p, arch[i][1]*p])

        real_arch = np.array(real_arch)
        real_arch[0][0] = real_arch[0][1]

        return real_arch


    def __get_param(self, gs):

        total_param = 0
        conv_param = []
        bn_param = []

        for name, m in self.named_modules():
            if 'Block' in name:
                info = re.findall(r"\d+-\d+", name)[0].split('-')
                block_idx = int(info[0])
                layer_idx = int(info[1])

                if 'conv1' in name:
                    if layer_idx == 0:
                        if block_idx == 0:
                            param = 32 * gs[block_idx][0]
                        else:
                            param = gs[block_idx-1][1] * gs[block_idx][0]
                    else:
                        param = gs[block_idx][1] * gs[block_idx][0]
                    conv_param.append(param)
                    total_param += param

                elif 'conv2' in name:
                    conv_param.append(gs[block_idx][0] * 9)
                    total_param += gs[block_idx][0] * 9

                elif 'conv3' in name:
                    param = gs[block_idx][0] * gs[block_idx][1]
                    conv_param.append(param)
                    total_param += param
                
                elif 'bn1' in name or 'bn2' in name:
                    total_param += gs[block_idx][0] * 2
                    bn_param.append(gs[block_idx][0] * 2)
                
                elif 'bn3' in name:
                    total_param += gs[block_idx][1] * 2
                    bn_param.append(gs[block_idx][1] * 2)
            
            elif isinstance(m, nn.Conv2d):
                param = m.kernel_size[0] * m.kernel_size[1] * m.in_channels * m.out_channels
                conv_param.append(param)
                total_param += param
            
            elif isinstance(m, nn.BatchNorm2d):
                total_param += m.num_features * 2
                bn_param.append(m.num_features * 2)
            
            elif isinstance(m, nn.Linear):
                total_param += m.out_features * m.in_features
        
        return total_param, conv_param, bn_param

# if __name__ == "__main__":
#     def cf(num, format="%.2f"):
#         if num > 1e12:
#             return format % (num / 1e12) + "T"
#         if num > 1e9:
#             return format % (num / 1e9) + "G"
#         if num > 1e6:
#             return format % (num / 1e6) + "M"
#         if num > 1e3:
#             return format % (num / 1e3) + "K"
    
#     arch = np.array(
#        [[48, 8], #2:3:4
#         [192, 32], #2:3
#         [960, 160]] #1:2
#     )
   
#     # arch = np.array(
#     #     [[16, 16],
#     #     [6*24,  24],
#     #     [6*32,  32],
#     #     [6*64,  64],
#     #     [6*96,  96],
#     #     [6*160, 160],
#     #     [6*320, 320]]
#     # )
#     net = MobileNetV2(arch)
#     # net = MobileNetV2(arch, num_classes=1000, small_input=False)
#     x = torch.randn(2,3,32,32)
#     y = net(x)
#     print(y.size())
#     # cp=[]
#     # for m in net.modules():
#     #     if isinstance(m, nn.Conv2d):
#     #         cp.append(m.weight.data.numel())
#     # print(cp)
#     param, ops = net.cost()
#     print(param,ops)
