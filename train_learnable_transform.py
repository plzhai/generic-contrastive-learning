import torch
import time
import sys
import os
import torch.nn as nn
from models import networks
from models.attack_libs import RepresentationAdv
import torch.optim as optim
import utils
from dataset import TwoCropsTransform, NoAugTransform
from argument import args
from torchvision import transforms
from utils import AverageMeter, accuracy, LinearClassifier
import torch.backends.cudnn as cudnn
from sync_batchnorm import convert_model
from NCE.NCEAverage import InfoNCE, MemoryInsDis, MemoryMoCo, moment_update, MemoryAdv
from NCE.NCECriterion import NCESoftmaxLoss, NCECriterion

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

def set_model(args):
    model = networks.__getattribute__(args.arch)()
    transform = networks.__getattribute__('transform_layer')(3, 32, 32, args.epsilon)
    if args.head == '2fc':
        model.fc = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), model.fc)
    elif args.head == '2bn':
        model.fc = nn.Sequential(nn.Linear(512, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(),
                               nn.Linear(512, args.feat_dim, bias=False),
                               nn.BatchNorm1d(args.feat_dim))# projection layer
    else:
        pass
    criterion = InfoNCE(args.nce_t)

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = convert_model(model)
        model = model.cuda()
        transform = transform.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    learning_rate = args.learning_rate * args.batch_size / 256
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer_t = optim.SGD(transform.parameters(), lr=learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    return model, transform, criterion, optimizer, optimizer_t


def train(epoch, model, transform, criterion, optimizer, optimizer_t, scheduler, scheduler_t, train_loader, args):
    model.train()
    transform.train()
    criterion.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_t = AverageMeter()
    end = time.time()
    for idx, (inputs, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # len(inputs) ==2 here
        bsz = inputs[0].size(0)
        if torch.cuda.is_available():
            inputs = inputs[0].cuda(non_blocking=True), inputs[1].cuda(non_blocking=True)
        # -----------------------
        #  Train transform layer
        # -----------------------
        # advinputs = transform(inputs)
        # proj_l, proj_ab = model(inputs), model(advinputs)
        # loss_t = -criterion(proj_l, proj_ab)
        # optimizer_t.zero_grad()
        # loss_t.backward(retain_graph=True)
        # optimizer_t.step()
        #
        # scheduler_t.step()
        # losses_t.update(loss_t.item(), bsz)

        # -----------------------
        #  Train model
        # -----------------------
        proj_l, proj = model(inputs[0]), model(inputs[1])
        #proj = model(transform(inputs).detach())
        loss = criterion(proj_l, proj)
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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                'loss t {loss_t.val:.3f} ({loss_t.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, loss_t=losses_t))
            # print(out_l.shape)
            sys.stdout.flush()
            logging['loss'].append(losses.avg)


def main(args):
    # set early stopping
    patience = 3
    cur = 0
    best_acc1 = 0

    model, transform, criterion, optimizer, optimizer_t = set_model(args)
    scheduler = utils.LR_Scheduler(optimizer, args.warmup, 0, args.epochs,
                                   args.learning_rate*args.batch_size/256, 0, len(train_loader))
    scheduler_t = utils.LR_Scheduler(optimizer_t, args.warmup, 0, args.epochs,
                                   args.learning_rate * args.batch_size / 256, 0, len(train_loader))
    args.start_epoch = 1
    for epoch in range(args.start_epoch, args.epochs+1):
        #utils.adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")
        time1 = time.time()
        train(epoch, model, transform, criterion, optimizer, optimizer_t, scheduler, scheduler_t, train_loader, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        # save model

        # eval model
        if epoch % args.eval_freq == 0:
            print("==> evaluating")
            if args.classifier == 'logistic':
                acc1, acc5 = utils.linear_evaluate_model(model, epoch=20, data_type="cifar10")
            else:
                acc1 = utils.knn_evaluate_model(model, knn_k=args.knn_k, knn_t=args.knn_t)
            logging['epoch'].append(epoch)
            logging['acc@1'].append(acc1)
            print('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, acc1))

            # early stopping
            if acc1 > best_acc1:
                cur = 0
                best_acc1 = acc1
                logging['bst@1'] = best_acc1
            else:
                if cur >= patience:
                    logging_file = args.model_name + "_best@1_{:.3f}_epoch_{}".format(best_acc1, epoch - args.eval_freq)
                    utils.searize(logging, os.path.join(args.model_path, logging_file))
                    print('No improvement since epoch {} at {:.3f}'.format(epoch - args.eval_freq, best_acc1))
                    break
                else:
                    cur += 1
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # prepare the datasets
    print('==> Preparing data...')
    args.batch_size = 1024
    args.epochs = 200
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    augment_train = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomApply([
            transforms.ColorJitter(0.8 * 0.5, 0.8 * 0.5, 0.8 * 0.5, 0.2 * 0.5)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_loader = utils.prepare_data(NoAugTransform(transform_train, augment_train), args.batch_size, 'cifar10')

    args.eval_freq = 10
    args.nce_t = 0.1
    args.arch = 'alexnet_cifar'
    args.head  = '2fc'
    args.model_name = "{}_{}_{}_eps_{}_bsz_{}_t_{}_lr_{}".format(args.arch, args.head, args.method, args.epsilon, args.batch_size, args.nce_t, args.learning_rate)
    logging = {'epoch': [], 'acc@1': [], 'acc@5': [], 'loss':[]}
    main(args)
