import torch
import time
import sys
import os
import torch.nn as nn
from models import networks
from models.attack_libs import RepresentationAdv
import torch.optim as optim
import utils
from dataset import IndexedBioInfoDataset
from argument import args
from torchvision import transforms
from utils import AverageMeter, accuracy, LinearClassifier
import torch.backends.cudnn as cudnn
from sync_batchnorm import convert_model
from NCE.NCEAverage import InfoNCE, MemoryInsDis, MemoryMoCo, moment_update, MemoryAdv
from NCE.NCECriterion import NCESoftmaxLoss, NCECriterion
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'


def prepare_data(batch_size, data_name='Lake_2018'):
    # Data
    if data_name == 'cifar10':
        print('==> Preparing data..')
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
        train_loader = utils.prepare_data(transform_train, batch_size=batch_size, data_type=data_name, is_train=True)
        test_loader = utils.prepare_data(transform_test, batch_size=batch_size, data_type=data_name, is_train=False)
    elif data_name in ['Lake_2018', 'Campbell', 'Chen', 'Tasic18', 'Tosches_lizard', 'Tosches_turtle']:
        dataset = IndexedBioInfoDataset(data_name)

        nb_data = len(dataset)
        trainset, testset = torch.utils.data.random_split(dataset,
                                                          [int(np.ceil(nb_data * 0.8)), int(np.floor(nb_data * 0.2))])

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, sampler=None)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, sampler=None)
    return train_loader, test_loader

def set_model(args):
    model = networks.__getattribute__(args.arch)()
    contrast = MemoryInsDis(10, args.n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax).cuda()
    criterion = NCESoftmaxLoss()

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = convert_model(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    learning_rate = args.learning_rate * args.batch_size / 256
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    return model, criterion, optimizer, contrast


def train(epoch, model, criterion, contrast, optimizer, scheduler, train_loader, args):
    model.train()
    criterion.train()
    #criterion.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    for idx, (inputs, _, index) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # len(inputs) ==2 here
        bsz = inputs.size(0)
        if torch.cuda.is_available():
            inputs, index = inputs.cuda(non_blocking=True), index.cuda(non_blocking=True)
        proj = model(inputs)
        out = contrast(proj, index)
        loss = criterion(out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        # ===================meters=====================
        losses.update(loss.item(), bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            # print(out_l.shape)
            sys.stdout.flush()
            logging['loss'].append(losses.avg)


def main(model, criterion, optimizer, contrast, args):
    # set early stopping
    patience = 3
    cur = 0
    best_acc1 = 0

    scheduler = utils.LR_Scheduler(optimizer, args.warmup, 0, args.epochs,
                                   args.learning_rate*args.batch_size/256, 0, len(train_loader))

    args.start_epoch = 1
    for epoch in range(args.start_epoch, args.epochs+1):
        #utils.adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")
        time1 = time.time()
        train(epoch, model, criterion,contrast, optimizer, scheduler, train_loader, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        # save model

        # eval model
        if epoch % args.eval_freq == 0:
            print("==> evaluating")
            acc1 = utils.knn_evaluate_model(model, knn_k=200, knn_t=0.1, data_type=args.data_name)
            acc5 = acc1
            #acc1, acc5 = utils.linear_evaluate_model(model, epoch=20, data_type="cifar10")
            logging['epoch'].append(epoch)
            logging['acc@1'].append(acc1)
            logging['acc@5'].append(acc5)
            # early stopping
            if acc1 > best_acc1:
                cur = 0
                best_acc1 = acc1
                best_acc5 = acc5
                logging['bst@1'] = best_acc1
                logging['bst@5'] = best_acc5
            else:
                if cur >= patience:
                    #logging_file = args.model_name + "_best@1_{:.3f}_epoch_{}".format(best_acc1, epoch - args.eval_freq)
                    #utils.searize(logging, os.path.join(args.model_path, logging_file))
                    print('No improvement since epoch {} at {:.3f}'.format(epoch - args.eval_freq, best_acc1))
                    return best_acc1
                else:
                    cur += 1
        torch.cuda.empty_cache()
    return best_acc1

from train_Supervised import visualize

if __name__ == "__main__":
    # prepare the datasets
    for data_name in ['Campbell', 'Chen', 'Tasic18', 'Tosches_lizard', 'Tosches_turtle']:
        args.data_name = data_name

        print('==> Preparing data...')
        args.method = 'memory'
        args.arch = 'simple_mlp_infobio'
        args.batch_size = 256
        train_loader, test_loader = prepare_data(args.batch_size, data_name=args.data_name)
        #for adversarial_training in [False]
        args.epochs = 200
        args.eval_freq = 10
        args.nce_t = 0.1
        model, criterion, optimizer, contrast = set_model(args)
        #args.model_name = "{}_dir_{}_eps_{}_bsz_{}_t_{}_lr_{}".format(args.method,  args.epsilon, args.batch_size, args.nce_t, args.learning_rate)

        logging = {'epoch': [], 'acc@1': [], 'acc@5': [], 'loss':[]}
        acc = main(model, criterion, optimizer, contrast , args)
        visualize(model, data_name=args.data_name, task='contrast', acc=acc)
