from __future__ import print_function

import torch
import torch.nn as nn



class alexnet(nn.Module):
    def __init__(self, in_channel=1, feat_dim=128):
        super(alexnet, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channel, 96//2, 11, 4, 2, bias=False),
            nn.BatchNorm2d(96//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(96//2, 256//2, 5, 1, 2, bias=False),
            nn.BatchNorm2d(256//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(256//2, 384//2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384//2),
            nn.ReLU(inplace=True),
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(384//2, 384//2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384//2),
            nn.ReLU(inplace=True),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(384//2, 256//2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256 * 6 * 6 // 2, 4096 // 2),
            nn.BatchNorm1d(4096 // 2),
            nn.ReLU(inplace=True),
        )
        self.fc7 = nn.Sequential(
            nn.Linear(4096 // 2, 4096 // 2),
            nn.BatchNorm1d(4096 // 2),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(4096 // 2, feat_dim)
        #self.l2norm = Normalize(2)

    def forward(self, x, layer):
        if layer <= 0:
            return x
        x = self.conv_block_1(x)
        if layer == 1:
            return x
        x = self.conv_block_2(x)
        if layer == 2:
            return x
        x = self.conv_block_3(x)
        if layer == 3:
            return x
        x = self.conv_block_4(x)
        if layer == 4:
            return x
        x = self.conv_block_5(x)
        x = x.view(x.shape[0], -1)
        x = self.fc6(x)
        if layer == 5:
            return x
        x = self.fc7(x)
        if layer == 6:
            return x
        x = self.fc(x)
        #x = self.l2norm(x)
        return x


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class alexnet_cifar(nn.Module):  # for the input size 3*32*32
    def __init__(self, in_channel=3, feat_dim=128):
        super(alexnet_cifar, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channel, 96 // 2, 5, 1, 2, bias=False),  # kernel_size, stride, padding
            nn.BatchNorm2d(96 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # kernel_size, stride
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(96 // 2, 256 // 2, 3, 1, 2, bias=False),
            nn.BatchNorm2d(256 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )  # [128, 192, 8, 8]
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(256 // 2, 384 // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384 // 2),
            nn.ReLU(inplace=True),
        )  # [128, 192, 8, 8]
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(384 // 2, 384 // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384 // 2),
            nn.ReLU(inplace=True),
        )  # [128, 192, 8, 8]
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(384 // 2, 256 // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )  # [128, 128, 3, 3]
        self.fc6 = nn.Sequential(
            nn.Linear(128 * 3 * 3, 4096 // 2),
            nn.BatchNorm1d(4096 // 2),
            nn.ReLU(inplace=True),
        )
        self.fc7 = nn.Sequential(
            nn.Linear(4096 // 2, 1024 // 2),
            nn.BatchNorm1d(1024 // 2),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(1024 // 2, feat_dim)
        #self.l2norm = Normalize(2)

    def forward(self, x, layer=8):
        if layer <= 0:
            return x
        x = self.conv_block_1(x)
        if layer == 1:
            return x
        x = self.conv_block_2(x)
        if layer == 2:
            return x
        x = self.conv_block_3(x)
        if layer == 3:
            return x
        x = self.conv_block_4(x)
        if layer == 4:
            return x
        x = self.conv_block_5(x)
        x = x.view(x.shape[0], -1)
        x = self.fc6(x)
        if layer == 5:
            return x
        x = self.fc7(x)
        if layer == 6:
            return x
        x = self.fc(x)
        #x = self.l2norm(x)
        return x


if __name__ == '__main__':

    import torch
    model = alexnet().cuda()
    data = torch.rand(10, 3, 224, 224).cuda()
    out = model.compute_feat(data, 5)

    for i in range(10):
        out = model.compute_feat(data, i)
        print(i, out.shape)