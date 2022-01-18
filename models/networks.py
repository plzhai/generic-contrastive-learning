from models.resnet import ResNet, BasicBlock
from models.alexnet import alexnet_cifar
from models.transformers import ViT
import torch.utils.model_zoo as model_zoo
import torch
import math
import torch.nn as nn
from models.alexnet import Normalize
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def resnet18(in_channel=3, feat_dim=128):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], low_dim=feat_dim, in_channel=in_channel, width=1)
    return model

def alexnet(in_channel=3, feat_dim=128):
    model = alexnet_cifar(in_channel=in_channel, feat_dim=feat_dim)
    return model


class simple_conv(nn.Module):# for cifar10
    def __init__(self, in_channel=3, feat_dim=128):
        super(simple_conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 18, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(18, 36, 3)
        self.fc1 = nn.Linear(36 * 6 * 6, 512)
        self.fc = nn.Linear(512, feat_dim)
        #self.fc3 = nn.Linear(256, feat_dim)
        #self.l2norm = Normalize(2)

    def forward(self, x, layer=7):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        if layer == 6:
            return x
        x = self.fc(x)
        #x = self.fc3(x)
        #x = self.l2norm(x)
        return x


class simple_mlp(nn.Module):# for cifar10
    def __init__(self, in_channel=3, feat_dim=128):
        super(simple_mlp, self).__init__()
        self.fc1 = nn.Linear(in_channel * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc = nn.Linear(512, feat_dim)
        #self.fc4 = nn.Linear(256, feat_dim)
        #self.l2norm = Normalize(2)

    def forward(self, x, layer=7):

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if layer == 6:
            return x
        x = self.fc(x)
        #x = self.l2norm(x)
        return x


class simple_mlp_infobio(nn.Module):# for cifar10
    def __init__(self, feat_dim=10):
        super(simple_mlp_infobio, self).__init__()
        self.fc1 = nn.Linear(2000, 64)

        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc = nn.Linear(16, feat_dim)
        #self.l2norm = Normalize(2)

    def forward(self, x, layer=7):
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        if layer == 3:
            return x
        x = F.relu(self.fc2(x))
        if layer == 4:
            return x
        x = F.relu(self.fc3(x))
        if layer == 5:
            return x
        x = F.relu(self.fc4(x))
        if layer == 6:
            return x
        x = self.fc(x)
        #x = self.l2norm(x)
        return x


class simple_mlp_dropout_infobio(nn.Module):  # for cifar10
    def __init__(self, feat_dim=128):
        super(simple_mlp_dropout_infobio, self).__init__()
        self.fc1 = nn.Linear(2000, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(512, 256)
        self.fc = nn.Linear(256, feat_dim)
        # self.l2norm = Normalize(2)

    def forward(self, x, layer=7):
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.dropout1(F.relu(self.fc1(x)))
        if layer == 3:
            return x
        x = self.dropout2(F.relu(self.fc2(x)))
        if layer == 4:
            return x
        x = self.dropout3(F.relu(self.fc3(x)))
        if layer == 5:
            return x
        x = F.relu(self.fc4(x))
        if layer == 6:
            return x
        x = self.fc(x)
        # x = self.l2norm(x)
        return x

class HardReLU(nn.Hardtanh):
    def __init__(self, inplace: bool = False):
        super(HardReLU, self).__init__(0., 1., inplace)
    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class transform_layer(nn.Module):
    def __init__(self, c, h, w, epsilon):
        super(transform_layer, self).__init__()

        self.weight = nn.parameter.Parameter(torch.randn((c, h, w)))
        #self.bias = nn.parameter.Parameter(torch.randn((c, h, w)))
        self.sigmoid = nn.Hardsigmoid()
        #self.tanh = nn.Tanh()
        #self.relu = HardReLU()
        # self.c = c
        # self.h = h
        # self.w = w
        # self.epsilon = epsilon
        # self.transform = nn.Sequential(
        #     nn.Flatten(-2),
        #     nn.Linear(1024, 1024, bias=True),
        #     HardReLU()
        # )

    def forward(self, x):
        weight = self.sigmoid(self.weight)
        #bias = self.tanh(self.bias) * self.epsilon
        x = x * weight
        # x = self.relu(x)
        #bsz = x.size(0)
        #x = self.transform(x)
        #x = x.reshape(bsz, self.c, self.h, self.w)
        return x #[0, 1]

class transform_cnn_layer(nn.Module):
    def __init__(self, c, h, w, epsilon):
        super(transform_cnn_layer, self).__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding='same')
        self.conv2 = nn.Conv2d(c, c, 3, padding='same')
        self.conv3 = nn.Conv2d(c, c, 3, padding='same')
        self.sigmoid = nn.Hardsigmoid()
    def forward(self, x):
        orig = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = orig * self.sigmoid(self.conv3(x))
        return x


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view((-1,)+self.shape)
def alexnet_encoder():
    model = alexnet_cifar(in_channel=3, feat_dim=3*32*32)
    model.fc = nn.Sequential(
        model.fc,
        Reshape((3,32,32))
    )
    return model

class view_learner(nn.Module):
    def __init__(self, noise_type, encoder_type='alexnet'):
        super(view_learner, self).__init__()
        assert noise_type in ['additive', 'multiplicative', 'variational', 'bernoulli'], 'wrong type : {}'.format(noise_type)
        if encoder_type == 'alexnet':
            self.encoder = alexnet_encoder()
        else:
            self.encoder = nn.Sequential(
                Reshape((32*32*3,)),
                nn.Linear(32*32*3, 512),
                nn.ReLU(),
                nn.Linear(512, 32*32*3),
                Reshape((3,32,32,))
            )
        self.type = noise_type
        self.in_features = 3*32*32
        if self.type == 'additive':
            logvar_t = -1
            self.logvar_t = torch.nn.Parameter(torch.Tensor([logvar_t]))
        if self.type == 'multiplicative':
            self.max_alpha = 0.7
    def compute_distances(self, x):
        '''
        Computes the distance matrix for the KDE Entropy estimation:
        - x (Tensor) : array of functions to compute the distances matrix from
        '''
        x_norm = (x ** 2).sum(1).view(-1, 1)
        x_t = torch.transpose(x, 0, 1)
        x_t_norm = x_norm.view(1, -1)
        dist = x_norm + x_t_norm - 2.0 * torch.mm(x, x_t)
        dist = torch.clamp(dist, 0, np.inf)
        return dist
    def forward(self, x):
        if self.type == 'additive':
            x_t = x + torch.exp(0.5*self.logvar_t) * torch.randn_like(x)
            n_batch = x.size(0)
            var = torch.exp(self.logvar_t) + 1e-10  # to avoid 0's in the log
            # calculation of the constant
            normalization_constant = math.log(n_batch)
            # calculation of the elements contribution
            dist = self.compute_distances(x.view(-1, self.in_features))
            distance_contribution = - torch.mean(torch.logsumexp(input=- 0.5 * dist / var, dim=1))
            # mutual information calculation (natts)
            kl_xt = (normalization_constant + distance_contribution).div(math.log(2))
        elif self.type == 'multiplicative':
            alpha = torch.sigmoid(self.encoder(x)) * self.max_alpha + 1e-3
            kl = -torch.log(alpha)
            epsilon = torch.exp(torch.zeros_like(x) + alpha*torch.randn_like(x))
            x_t = x * epsilon
            kl_xt = kl.mean()#kl.sum(axis=[1,2,3]).mean()  # KL(p(z|x)|p(z))
        elif self.type == 'bernoulli':
            w_e = self.encoder(x)
            temperature = 1.
            bias = 0.0 + 0.0001
            delta = (bias - (1 - bias)) * torch.rand(x.size()) + (1 - bias)
            delta = delta.cuda()
            p_e = torch.sigmoid((torch.log(delta)-torch.log(1.-delta) + w_e)/temperature)
            #edge_drop_out_prob = 1. - p_e # minimize
            x_t = x * p_e
            kl_xt = p_e.sum([1,2,3]).div(32*32*3).mean() # minimize
        else:
            logvar = self.encoder(x)
            std = logvar.mul(0.5).exp_()
            #logvar = self.encoder(x)
            #std = F.softplus(logvar-5, beta=1.)
            epsilon = Variable(logvar.data.new(logvar.size()).normal_())
            x_t = x + epsilon * std

            #kl_xt = 0.5 * torch.mean(torch.sum(std**2 - torch.log(std**2) - 1,dim=[1,2,3])) # minimize
            kl_xt = ((logvar.exp() - 1 - logvar) / 2).mean()
        return x_t, kl_xt

def f(x,t):

    logits = np.log()
    return 1/(1+np.exp(-x))