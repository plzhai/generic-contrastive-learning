from __future__ import print_function

import torch
import pickle
import numpy as np
import torch.nn as nn
import sys
import math
import torchvision
import models.resnet as resnet
from torchvision import transforms
import dataset

EETA_DEFAULT = 0.001



def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def adjust_learning_rate_2(epoch, args, optimizer):
    """Decay the learning rate based on schedule"""
    cur_lr = args.learning_rate * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = args.learning_rate
        else:
            param_group['lr'] = cur_lr

def searize(data, path):
    with open(path, 'wb')as f:
        pickle.dump(data,f)
def unsearize(path):
    with open(path, 'rb')as f:
        data = pickle.load(f)
    return data

def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# cosine decay learning rate scheduler with linear warmup
class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch,
                 constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                    1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


from models.LinearModel import Flatten
# set the classifier for COC and CMC
class LinearClassifier(nn.Module):
    def __init__(self, feat_dim, n_label=10):
        super(LinearClassifier, self).__init__()

        self.classifier = nn.Sequential()
        self.classifier.add_module('Flatten', Flatten())
        print('classifier input: {}'.format(feat_dim))
        self.classifier.add_module('LiniearClassifier', nn.Linear(feat_dim, n_label))
        #self.classifier.add_module("Softmax", nn.Softmax(-1))
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.classifier(x)

from PIL import Image
class IndexdCifar10(torchvision.datasets.CIFAR10):
    def __init__(self, **kwargs):
        super(IndexdCifar10, self).__init__(**kwargs)
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

def prepare_data(transforms, batch_size=128, data_type='cifar10',is_train=True):
    if data_type == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root='../data', train=is_train, download=True, transform=transforms)
    elif data_type == 'STL10':
        trainset = torchvision.datasets.STL10(
            root='../data', split="train" if is_train else "test", download=True, transform=transforms)
    else:
        raise KeyError('no given data type {}'.format(data_type))
    # train_loader = torch.utils.data.DataLoader(
    #    trainset, batch_size=128, shuffle=True, num_workers=2)
    data_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, sampler=None)
    return data_loader

def linear_evaluate_model(model, epoch=20, data_type='cifar10'):
    model.eval()
    best_acc_1 = 0
    best_acc_5 = 0
    patience = 2
    cur = 0
    if data_type == 'cifar10':
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        raise KeyError('no given data {}'.format(data_type))
    train_loader = prepare_data(transform_train,128,data_type, is_train=True)
    test_loader = prepare_data(transform_test,128,data_type,is_train=False)

    classifier = LinearClassifier(512, num_classes)
    criterion = nn.CrossEntropyLoss().cuda()
    if torch.cuda.is_available():
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4, weight_decay=0.0008)
    # train the classifier
    for e in range(epoch):
        classifier.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for idx, (imgs, tgt) in enumerate(train_loader):
            if torch.cuda.is_available():
                imgs = imgs.cuda(non_blocking=True)
                tgt = tgt.cuda(non_blocking=True)
            with torch.no_grad():
                hidden = model(imgs,layer=6)
                hidden = hidden.detach()
            output = classifier(hidden)
            loss = criterion(output, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc1, acc5 = accuracy(output, tgt, topk=(1, 5))
            losses.update(loss.item(), imgs.size(0))
            top1.update(acc1[0], imgs.size(0))
            top5.update(acc5[0], imgs.size(0))

        print('Train: [{0}/{1}]\t'
              'Acc@1 {top1.avg:.3f}\t'
              'Acc@5 {top5.avg:.3f}\t'.format(e, epoch, top1=top1, top5=top5))

        # test the classifier
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        classifier.eval()
        with torch.no_grad():
            for idx, (images,target) in enumerate(test_loader):
                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                hidden = model(images,layer=6)
                output = classifier(hidden)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
            print('Test: [{0}/{1}]\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}\t'.format(e, epoch, top1=top1, top5=top5))

        tst_acc_1 = top1.avg
        tst_acc_5 = top5.avg
        if tst_acc_1 <= best_acc_1:
            if cur >= patience:
                break
            else:
                cur += 1
        else:
            best_acc_1 = tst_acc_1
            best_acc_5 = tst_acc_5

    return best_acc_1, best_acc_5


def linear_evaluate_model_v2(model, epoch=20, data_type='cifar10', view='Lab'):
    model.eval()
    best_acc_1 = 0
    best_acc_5 = 0
    patience = 2
    cur = 0
    if data_type == 'cifar10':
        num_classes = 10
        if view == 'Lab':
            mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
            std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
            transform_train = transforms.Compose([
				# transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.0)),
				transforms.RandomCrop(32, padding=4), # maybe not necessary
				transforms.RandomHorizontalFlip(),
				dataset.RGB2Lab(),
				transforms.ToTensor(),
				transforms.Normalize(mean, std),
			])
            transform_test = transforms.Compose([
                dataset.RGB2Lab(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    else:
        raise KeyError('no given data {}'.format(data_type))
    train_loader = prepare_data(transform_train,256,data_type, is_train=True)
    test_loader = prepare_data(transform_test,256,data_type,is_train=False)

    classifier = LinearClassifier(512*2, num_classes)
    criterion = nn.CrossEntropyLoss().cuda()
    if torch.cuda.is_available():
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4, weight_decay=0.0008)
    # train the classifier
    for e in range(epoch):
        classifier.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for idx, (imgs, tgt) in enumerate(train_loader):
            if torch.cuda.is_available():
                imgs = imgs.cuda(non_blocking=True)
                tgt = tgt.cuda(non_blocking=True)
            with torch.no_grad():
                if view == 'Lab':
                    l, ab = torch.split(imgs, [1, 2], dim=1)
                else:
                    l, ab = imgs, imgs
                hidden1, hidden2, _, _ = model(l, ab)
                hidden = torch.cat([hidden1, hidden2], dim=-1)
                hidden = hidden.detach()
            output = classifier(hidden)
            loss = criterion(output, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc1, acc5 = accuracy(output, tgt, topk=(1, 5))
            losses.update(loss.item(), imgs.size(0))
            top1.update(acc1[0], imgs.size(0))
            top5.update(acc5[0], imgs.size(0))

        print('Train: [{0}/{1}]\t'
              'Acc@1 {top1.avg:.3f}\t'
              'Acc@5 {top5.avg:.3f}\t'.format(e, epoch, top1=top1, top5=top5))

        # test the classifier
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        classifier.eval()
        with torch.no_grad():
            for idx, (images,target) in enumerate(test_loader):
                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                if view == 'Lab':
                    l, ab = torch.split(images, [1, 2], dim=1)
                else:
                    l, ab = images, images
                hidden1, hidden2, _, _ = model(l, ab)
                hidden = torch.cat([hidden1, hidden2], dim=-1)
                output = classifier(hidden)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
            print('Test: [{0}/{1}]\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}\t'.format(e, epoch, top1=top1, top5=top5))

        tst_acc_1 = top1.avg
        tst_acc_5 = top5.avg
        if tst_acc_1 <= best_acc_1:
            if cur >= patience:
                break
            else:
                cur += 1
        else:
            best_acc_1 = tst_acc_1
            best_acc_5 = tst_acc_5

    return best_acc_1, best_acc_5

def linear_evaluate(model_path, epoch=20, arch='resnet18', width=1, data_type='cifar10'):
    best_acc_1 = 0
    best_acc_5 = 0
    patience = 3
    cur = 0
    if data_type == 'cifar10':
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        raise KeyError('no given data {}'.format(data_type))
    train_loader = prepare_data(transform_train,128,data_type, is_train=True)
    test_loader = prepare_data(transform_test,128,data_type,is_train=False)

    model = resnet.__getattribute__(arch)(low_dim=num_classes, in_channel=3, width=width)
    ckpt = torch.load(model_path)
    msg = model.load_state_dict(ckpt['model'], strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    criterion = nn.CrossEntropyLoss().cuda()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.Adam(parameters, lr=3e-4, weight_decay=0.0008)
    # train the classifier
    for e in range(epoch):
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for idx, (imgs, tgt) in enumerate(train_loader):
            if torch.cuda.is_available():
                imgs = imgs.cuda(non_blocking=True)
                tgt = tgt.cuda(non_blocking=True)
            output = model(imgs)
            loss = criterion(output, tgt)

            acc1, acc5 = accuracy(output, tgt, topk=(1, 5))
            losses.update(loss.item(), imgs.size(0))
            top1.update(acc1[0], imgs.size(0))
            top5.update(acc5[0], imgs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train: [{0}/{1}]\t'
              'Acc@1 {top1.avg:.3f}\t'
              'Acc@5 {top5.avg:.3f}\t'.format(e, epoch, top1=top1, top5=top5))

        # test the classifier
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            for idx, (images,target) in enumerate(test_loader):
                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                output = model(images)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
            print('Test: [{0}/{1}]\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}\t'.format(e, epoch, top1=top1, top5=top5))

        tst_acc_1 = top1.avg
        tst_acc_5 = top5.avg
        if tst_acc_1 < best_acc_1:
            if cur >= patience:
                break
            else:
                cur += 1
        else:
            best_acc_1 = tst_acc_1
            best_acc_5 = tst_acc_5

    return best_acc_1, best_acc_5

#  for CMC nets and COC nets with different encoders
def linear_evaluate_v2(model_path, epoch=20, arch='resnet18', width=1, data_type='cifar10'):
    best_acc_1 = 0
    best_acc_5 = 0
    patience = 3
    cur = 0
    if data_type == 'cifar10':
        num_classes = 10
        classifier = LinearClassifier(1024, num_classes)
        classifier.train()
        if torch.cuda.is_available():
            classifier = classifier.cuda()
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        raise KeyError('no given data {}'.format(data_type))
    train_loader = prepare_data(transform_train, 128, data_type, is_train=True)
    test_loader = prepare_data(transform_test, 128, data_type, is_train=False)
    if "CMC_" in model_path:
        model_1 = resnet.__getattribute__(arch)(low_dim=num_classes, in_channel=1, width=width)
        model_2 = resnet.__getattribute__(arch)(low_dim=num_classes, in_channel=2, width=width)
    elif "COC" in  model_path:
        model_1 = resnet.__getattribute__(arch)(low_dim=num_classes, in_channel=3, width=width)
        model_2 = resnet.__getattribute__(arch)(low_dim=num_classes, in_channel=3, width=width)
    ckpt = torch.load(model_path)
    msg1 = model_1.load_state_dict(ckpt['model.f1'], strict=False)
    msg2 = model_2.load_state_dict(ckpt['model.f2'], strict=False)
    assert set(msg1.missing_keys) == {"fc.weight", "fc.bias"} and set(msg2.missing_keys) == {"fc.weight", "fc.bias"}
    # freeze all layers but the last fc

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model_1 = model_1.cuda()
        model_2 = model_2.cuda()
        criterion = criterion.cuda()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4, weight_decay=0.0008)
    # train the classifier
    for e in range(epoch):
        classifier.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for idx, (imgs, tgt) in enumerate(train_loader):
            if torch.cuda.is_available():
                imgs = imgs.cuda(non_blocking=True)
                tgt = tgt.cuda(non_blocking=True)
            if "CMC_" in  model_path:
                x1, x2 = torch.split(imgs, [1, 2], dim=1)
            else:
                x1, x2 = imgs, imgs
            with torch.no_grad():
                hidden = torch.cat([model_1(x1,layer=6),model_2(x2,layer=6)],dim=1)
                hidden = hidden.detach()
            output = classifier(hidden)
            loss = criterion(output, tgt)

            acc1, acc5 = accuracy(output, tgt, topk=(1, 5))
            losses.update(loss.item(), imgs.size(0))
            top1.update(acc1[0], imgs.size(0))
            top5.update(acc5[0], imgs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train: [{0}/{1}]\t'
              'Acc@1 {top1.avg:.3f}\t'
              'Acc@5 {top5.avg:.3f}\t'.format(e, epoch, top1=top1, top5=top5))

        # test the classifier
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        classifier.eval()
        with torch.no_grad():
            for idx, (images, target) in enumerate(test_loader):
                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                if "CMC_" in model_path:
                    x1, x2 = torch.split(images, [1, 2], dim=1)
                else:
                    x1, x2 = images, images
                hidden = torch.cat([model_1(x1, layer=6), model_2(x2, layer=6)], dim=1)
                output = classifier(hidden)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
            print('Test: [{0}/{1}]\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}\t'.format(e, epoch, top1=top1, top5=top5))

        tst_acc_1 = top1.avg
        tst_acc_5 = top5.avg
        if tst_acc_1 < best_acc_1:
            if cur >= patience:
                break
            else:
                cur += 1
        else:
            best_acc_1 = tst_acc_1
            best_acc_5 = tst_acc_5

    return best_acc_1, best_acc_5
import tqdm
def test(net, memory_data_loader, test_data_loader, epoch, args):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description(
                'Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

    return total_top1 / total_num * 100


# test using a knn monitor
import torch.nn.functional as F
from dataset import BioInfoDataset
def knn_evaluate_model(net, knn_k=200, knn_t=0.1, data_type='cifar10'):
    net.eval()
    if data_type == 'cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root='../data', train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR10(
            root='../data', train=False, download=True, transform=transform_test)
        memory_data_loader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=False,
            num_workers=0, pin_memory=True, sampler=None)
        test_data_loader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False,
            num_workers=0, pin_memory=True, sampler=None)
        classes = len(memory_data_loader.dataset.classes)
    else:
        dataset = BioInfoDataset(data_type)
        nb_data = len(dataset)
        trainset, testset = torch.utils.data.random_split(dataset,
                                                          [int(np.ceil(nb_data * 0.8)), int(np.floor(nb_data * 0.2))])

        memory_data_loader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True, sampler=None)
        test_data_loader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True, sampler=None)
        classes = memory_data_loader.dataset[0:][1].max() +1

    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in memory_data_loader:
            feature = net(data.cuda(non_blocking=True), layer=6)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        #feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        feature_labels = torch.tensor(memory_data_loader.dataset[0:][1], device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        #test_bar = tqdm(test_data_loader)
        for data, target in test_data_loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data, layer=6)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            #test_bar.set_description(
            #    'Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

    return total_top1 / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


import torchvision
class CIFAR10Instance(torchvision.datasets.CIFAR10):
    """CIFAR10Instance Dataset.
    """
    def __getitem__(self, index):
        #if self.train:
        #    img, target = self.train_data[index], self.train_labels[index]
        #else:
        #    img, target = self.test_data[index], self.test_labels[index]
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index





if __name__ == '__main__':
    meter = AverageMeter()
