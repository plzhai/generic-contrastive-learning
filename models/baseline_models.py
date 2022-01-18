import torch
import torch.nn as nn
from models.alexnet import Normalize
import torch.nn.functional as F
import torchvision.models as models
import math
import numpy as np
from models import alexnet, resnet, networks






class MyCOCNets(nn.Module):
    def __init__(self, arch='resnet18', feat_dim=2048):#['resnet18','resnet18']
        super(MyCOCNets, self).__init__()
        #self.encoder = MixNet(name)
        if arch == 'resnet18':
            self.f1 = resnet.__getattribute__(arch)(in_channel=3, width=1)
            self.f2 = resnet.__getattribute__(arch)(in_channel=3, width=1)
        elif arch == 'alexnet_cifar':
            self.f1 = alexnet.__getattribute__(arch)(in_channel=3)
            self.f2 = alexnet.__getattribute__(arch)(in_channel=3)
        else:
            raise KeyError('no given arch {}'.format(arch))
            #self.f2 = resnet.__getattribute__(name[1])(in_channel=3,width=1)

        prev_dim = self.f1.fc.weight.shape[1]
        self.f1.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                   nn.BatchNorm1d(prev_dim),
                                   nn.ReLU(inplace=True),  # first layer
                                   nn.Linear(prev_dim, feat_dim, bias=True),
                                   # nn.BatchNorm1d(prev_dim),
                                   # nn.ReLU(inplace=True),  # second layer
                                   # self.f1.fc,
                                   nn.BatchNorm1d(feat_dim, affine=False))  # output layer
        self.f1.fc[3].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        self.f2.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                   nn.BatchNorm1d(prev_dim),
                                   nn.ReLU(inplace=True),  # first layer
                                   nn.Linear(prev_dim, feat_dim, bias=True),
                                   # nn.BatchNorm1d(prev_dim),
                                   # nn.ReLU(inplace=True),  # second layer
                                   # self.f1.fc,
                                   nn.BatchNorm1d(feat_dim, affine=False))  # output layer
        self.f2.fc[3].bias.requires_grad = False  # hack: not use bias as it is followed by BN


        #self.encoder = nn.DataParallel(self.encoder)
        self.predictor = nn.Sequential(nn.Linear(feat_dim, 512, bias=False),
                                       nn.BatchNorm1d(512),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(512, feat_dim))  # output layer

    def forward(self, x):
        z1 = self.f1(x)
        z2 = self.f2(x)
        #z1, z2 = self.encoder(x, layers) # NxC
        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC
        return z1,z2,p1,p2


class SimSiam(nn.Module):
    def __init__(self, arch= 'resnet18', feat_dim=2048):
        super(SimSiam, self).__init__()
        #self.f1 = models.__dict__['resnet18'](num_classes=feat_dim, zero_init_residual=True)
        if arch == 'resnet18':
            self.f1 = resnet.__getattribute__(arch)(in_channel=3, width=1)
        elif arch == 'alexnet_cifar':
            self.f1 = alexnet.__getattribute__(arch)(in_channel=3)
        else:
            raise KeyError('no given arch {}'.format(arch))
        prev_dim = self.f1.fc.weight.shape[1]
        self.f1.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                   nn.BatchNorm1d(prev_dim),
                                   nn.ReLU(inplace=True),  # first layer
                                   nn.Linear(prev_dim, feat_dim, bias=True),
                                   # nn.BatchNorm1d(prev_dim),
                                   # nn.ReLU(inplace=True),  # second layer
                                   # self.f1.fc,
                                   nn.BatchNorm1d(feat_dim, affine=False))  # output layer
        self.f1.fc[3].bias.requires_grad = False  # hack: not use bias as it is followed by BN
        self.predictor = nn.Sequential(nn.Linear(feat_dim, 512, bias=False),
                                       nn.BatchNorm1d(512),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(512, feat_dim))  # output layer
    def forward(self,x1,x2):
        z1 = self.f1(x1)  # NxC # for linear evaluation
        z2 = self.f1(x2)  # NxC # # for linear evaluation

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC
        return z1.detach(), z2.detach(), p1, p2


class CMC(nn.Module):
    def __init__(self, arch='resnet18', feat_dim=128):
        super(CMC, self).__init__()
        if arch == 'resnet18':
            self.f1 = resnet.__getattribute__(arch)(low_dim=feat_dim, in_channel=1, width=1)
            self.f2 = resnet.__getattribute__(arch)(low_dim=feat_dim, in_channel=2, width=1)
        elif arch == 'alexnet_cifar':
            self.f1 = alexnet.__getattribute__(arch)(in_channel=1)
            self.f2 = alexnet.__getattribute__(arch)(in_channel=2)
        else:
            raise NotImplementedError('arch {} is not implemented'.format(arch))
        self.f1.fc = nn.Sequential(self.f1.fc,
                                   Normalize(2))
        self.f2.fc = nn.Sequential(self.f2.fc,
                                   Normalize(2))

    def forward(self, l, ab):
        #l, ab = torch.split(x, [1, 2], dim=1)
        z1 = self.f1(l, layer=6)
        z2 = self.f2(ab, layer=6)
        p1 = self.f1.fc(z1)
        p2 = self.f2.fc(z2)
        return z1, z2, p1, p2

class COC(nn.Module):
    def __init__(self, arch=['resnet18', 'alexnet'], feat_dim=2048):
        super(COC, self).__init__()
        assert arch[0] in ['resnet18', 'alexnet', 'simple_conv', 'simple_mlp'] and arch[1] in \
               ['resnet18', 'alexnet', 'simple_conv', 'simple_mlp'], "no given arch name {} or {}".format(arch[0],
                                                                                                          arch[1])
        self.f1 = networks.__getattribute__(arch[0])(feat_dim=feat_dim, in_channel=3)
        self.f2 = networks.__getattribute__(arch[1])(feat_dim=feat_dim, in_channel=3)

        dim_mlp = self.f1.fc.in_features
        self.f1.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp, bias=False),
                                   nn.BatchNorm1d(dim_mlp),
                                   nn.ReLU(),
                                   nn.Linear(dim_mlp, feat_dim, bias=True),
                                   nn.BatchNorm1d(feat_dim))
        self.f1.fc[3].bias.requires_grad = False  # hack: not use bias as it is followed by BN
        self.f2.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp, bias=False),
                                   nn.BatchNorm1d(dim_mlp),
                                   nn.ReLU(),
                                   nn.Linear(dim_mlp, feat_dim, bias=True),
                                   nn.BatchNorm1d(feat_dim))
        self.f2.fc[3].bias.requires_grad = False  # hack: not use bias as it is followed by BN
        self.predictor1 = nn.Sequential(nn.Linear(feat_dim, 512, bias=False),
                                       nn.BatchNorm1d(512),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(512, feat_dim))  # output layer
        self.predictor2 = nn.Sequential(nn.Linear(feat_dim, 512, bias=False),
                                       nn.BatchNorm1d(512),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(512, feat_dim))  # output layer
    def forward(self, l, ab):
        #l, ab = torch.split(x, [1, 2], dim=1)
        z1 = self.f1(l, layer=6)
        z2 = self.f2(ab, layer=6)
        d1 = self.f1.fc(z1)
        d2 = self.f2.fc(z2)
        p1 = self.predictor1(d1)
        p2 = self.predictor2(d2)
        return z1, z2, [d1.detach(), d2.detach()], [p1, p2]

class AlignedCOC(nn.Module):
    def __init__(self, arch=['resnet18','alexnet'], feat_dim=128):
        super(AlignedCOC, self).__init__()
        assert arch[0] in ['resnet18', 'alexnet', 'simple_conv', 'simple_mlp'] and arch[1] in \
               ['resnet18', 'alexnet', 'simple_conv', 'simple_mlp'], "no given arch name {} or {}".format(arch[0], arch[1])
        self.f1 = networks.__getattribute__(arch[0])(feat_dim=feat_dim, in_channel=3)
        self.f2 = networks.__getattribute__(arch[1])(feat_dim=feat_dim, in_channel=3)
        dim_mlp = self.f1.fc.in_features
        self.f1.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp, bias=False),
                                   nn.BatchNorm1d(dim_mlp),
                                   nn.ReLU(),
                                   nn.Linear(dim_mlp, feat_dim, bias=False),
                                   nn.BatchNorm1d(feat_dim))

        self.f2.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp, bias=False),
                                   nn.BatchNorm1d(dim_mlp),
                                   nn.ReLU(),
                                   nn.Linear(dim_mlp, feat_dim, bias=False),
                                   nn.BatchNorm1d(feat_dim))

    def forward(self, l, ab):
        #l, ab = torch.split(x, [1, 2], dim=1)
        z1 = self.f1(l, layer=6)
        z2 = self.f2(ab, layer=6)
        p1 = self.f1.fc(z1)
        p2 = self.f2.fc(z2)
        return z1, z2, p1, p2

class LocalCOC(nn.Module):
    def __init__(self, arch=['resnet18','alexnet'], feat_dim=128):
        super(LocalCOC, self).__init__()
        assert arch[0] in ['resnet18', 'alexnet', 'simple_conv', 'simple_mlp'] and arch[1] in \
               ['resnet18', 'alexnet', 'simple_conv', 'simple_mlp'], "no given arch name {} or {}".format(arch[0], arch[1])

        self.f1 = networks.__getattribute__(arch[0])(feat_dim=feat_dim, in_channel=3)
        self.f2 = networks.__getattribute__(arch[1])(feat_dim=feat_dim, in_channel=3)

        dim_mlp = self.f1.fc.in_features
        self.f1.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp, bias=False),
                                   nn.BatchNorm1d(dim_mlp),
                                   nn.ReLU(),
                                   nn.Linear(dim_mlp, feat_dim, bias=False),
                                   nn.BatchNorm1d(feat_dim))

        self.f2.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp, bias=False),
                                   nn.BatchNorm1d(dim_mlp),
                                   nn.ReLU(),
                                   nn.Linear(dim_mlp, feat_dim, bias=False),
                                   nn.BatchNorm1d(feat_dim))
        self.predictor1 = nn.Sequential(nn.AvgPool2d(9, stride=1),
                                       nn.Flatten(),
                                       nn.Linear(256, dim_mlp, bias=False),
                                       nn.BatchNorm1d(dim_mlp),
                                       nn.ReLU(),  # hidden layer
                                       nn.Linear(dim_mlp, feat_dim, bias=False),
                                       nn.BatchNorm1d(feat_dim))  # output layer
        self.predictor2 = nn.Sequential(nn.AvgPool2d(9, stride=1),
                                       nn.Flatten(),
                                       nn.Linear(256, dim_mlp, bias=False),
                                       nn.BatchNorm1d(dim_mlp),
                                       nn.ReLU(),  # hidden layer
                                       nn.Linear(dim_mlp, feat_dim, bias=False),
                                       nn.BatchNorm1d(feat_dim))  # output layer

    def forward(self, l, ab):
        #l, ab = torch.split(x, [1, 2], dim=1)
        t1 = self.f1(l, layer=4)
        t2 = self.f2(ab, layer=4)
        p1_ = self.predictor1(t1)
        p2_ = self.predictor2(t2)

        z1 = self.f1.layer4(t1)
        z1 = self.f1.avgpool(z1) if z1.shape[-1] > 1 else z1
        z1 =  z1.view(z1.size(0), -1)

        z2 = self.f2.layer4(t2)
        z2 = self.f2.avgpool(z2) if z2.shape[-1] > 1 else z2
        z2 =  z2.view(z2.size(0), -1)

        p1 = self.f1.fc(z1)
        p2 = self.f2.fc(z2)
        return z1, z2, [p1_, p2_], [p1, p2]
class MomentumCOC(nn.Module):
    def __init__(self, arch=['resnet18','resnet18'], feat_dim=128):
        super(MomentumCOC, self).__init__()
        assert arch[0] in ['resnet18', 'alexnet', 'simple_conv', 'simple_mlp'] and arch[1] in \
               ['resnet18', 'alexnet', 'simple_conv', 'simple_mlp'], "no given arch name {} or {}".format(arch[0], arch[1])
        self.f1 = networks.__getattribute__(arch[0])(feat_dim=feat_dim, in_channel=3)
        self.f2 = networks.__getattribute__(arch[1])(feat_dim=feat_dim, in_channel=3)
        dim_mlp = self.f1.fc.in_features
        self.projector = nn.Sequential(nn.Linear(dim_mlp, dim_mlp, bias=False),
                                   nn.BatchNorm1d(dim_mlp),
                                   nn.ReLU(),
                                   nn.Linear(dim_mlp, feat_dim, bias=False),
                                   nn.BatchNorm1d(feat_dim))
    def update_moving_average(self, momentum=0.99):
        for current_params, ma_params in zip(self.f1.parameters(), self.f2.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = old_weight * momentum + (1 - momentum) * up_weight
    def forward(self, l, ab):
        #l, ab = torch.split(x, [1, 2], dim=1)
        z1 = self.f1(l, layer=6)
        p1 = self.projector(z1)
        with torch.no_grad():
            z2 = self.f2(ab, layer=6)
            p2 = self.projector(z2)
        return z1, z2.detach(), p1, p2.detach()


class SimCLR(nn.Module):
    def __init__(self, arch='resnet18', feat_dim=128):
        super(SimCLR, self).__init__()
        self.f1 = networks.__getattribute__(arch)(feat_dim=feat_dim, in_channel=3)
        #self.f1 = networks.__getattribute__(arch)(feat_dim=feat_dim, in_channel=3)

        dim_mlp = self.f1.fc.in_features
        self.f1.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp, bias=False),
                                   nn.BatchNorm1d(dim_mlp),
                                   nn.ReLU(),
                                   nn.Linear(dim_mlp, feat_dim, bias=False),
                                   nn.BatchNorm1d(feat_dim))# projection layer
    def forward(self, x1, x2):
        z1 = self.f1(x1,layer=6)# for linear evaluation
        z2 = self.f1(x2,layer=6) # for linear evaluation
        p1 = self.f1.fc(z1) # for loss computation
        p2 = self.f1.fc(z2) # for loss computation
        return z1, z2, p1, p2

# this is for 3 views
class MyDeepCoNets(nn.Module):
    def __init__(self, arch='resnet18', feat_dim=128):
        super(MyDeepCoNets, self).__init__()
        self.f1 = networks.__getattribute__(arch)(feat_dim=feat_dim, in_channel=3)
        dim_mlp = self.f1.fc.in_features
        self.f1.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp, bias=False),
                                   nn.BatchNorm1d(dim_mlp),
                                   nn.ReLU(),
                                   nn.Linear(dim_mlp, feat_dim, bias=False),
                                   nn.BatchNorm1d(feat_dim))  # projection layer
        self.predictor = nn.Sequential(nn.AvgPool2d(9, stride=1),
                                       nn.Flatten(),
                                       nn.Linear(256, dim_mlp, bias=False),
                                       nn.BatchNorm1d(dim_mlp),
                                       nn.ReLU(),  # hidden layer
                                       nn.Linear(dim_mlp, feat_dim, bias=False),
                                       nn.BatchNorm1d(feat_dim))  # output layer

    def forward(self, x1, x2, x3):
        p1_ = self.f1(x1, layer=4)
        p2_ = self.f1(x2, layer=4)
        z1 = p1_ # first layer of contrastive learninng framework
        # for the first contrastive loss computation : info_nce(p1,p2)
        p1 = self.predictor(p1_)
        p2 = self.predictor(p2_)

        z = self.f1.layer4(z1)
        z = self.f1.avgpool(z) if z.shape[-1] > 1 else z

        p3_ = self.f1(x3, layer=6)
        p4_ =  z.view(z.size(0), -1)

        z2 = p4_ # second layer of contrastive learninng framework
        # for the second contrastive loss computation : info_nce(p1,p2)
        p3 = self.f1.fc(p3_)
        p4 = self.f1.fc(z2)
        #z3 = self.f1(x3, layer=6)
        return z1, z2, [p1, p2], [p3, p4]


class SimCSE(nn.Module):
    def __init__(self, feat_dim):
        super(SimCSE, self).__init__()

    def forward(self):
        pass


class NIBlayer(nn.Module):
    def __init__(self, logvar_t=-1.0):
        super(NIBlayer, self).__init__()
        self.logvar_t = torch.nn.Parameter(torch.Tensor([logvar_t]))
    def forward(self, hidden):
        return hidden + torch.exp(0.5 * self.logvar_t) * torch.randn_like((hidden))


class MyIBNets(nn.Module):
    def __init__(self, feat_dim):
        super(MyIBNets, self).__init__()
        self.f1 = resnet.__getattribute__('resnet18')(in_channel=3, width=1)

        dim_mlp = self.f1.fc.in_features
        self.f1.fc = nn.Sequential(NIBlayer(),
                                   nn.Linear(dim_mlp, dim_mlp),
                                   nn.ReLU(),
                                   nn.Linear(dim_mlp, feat_dim, bias=True),)# projection l

    def forward(self):
        pass
