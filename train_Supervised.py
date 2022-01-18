import torch
import torch.nn as nn
import torchvision.transforms as transforms
import utils
import os
import hypertools as hyp
import matplotlib.pyplot as plt
import sys
from utils import adjust_learning_rate,adjust_learning_rate_2, AverageMeter, accuracy, LinearClassifier
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
from models import networks
import numpy as np
from dataset import BioInfoDataset
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

batch_size = 256
print_freq = 10
save_freq = 20
learning_rate = 0.001
weight_decay = 5e-4
arch = 'ViT'
# arch = 'ViT'
num_classes = 10
arch = 'simple_mlp_infobio'

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
        dataset = BioInfoDataset(data_name)

        nb_data = len(dataset)
        trainset, testset = torch.utils.data.random_split(dataset,
                                                          [int(np.ceil(nb_data * 0.8)), int(np.floor(nb_data * 0.2))])

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=256, shuffle=True, num_workers=0, pin_memory=True, sampler=None)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True, sampler=None)
    return train_loader, test_loader


def set_model(num_classes):
    print("==> Building model...")
    model = networks.__getattribute__(arch)(num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True
    return model, criterion, optimizer, scheduler


# Training one epoch
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    end = time.time()
    for idx, (inputs, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)
        inputs, targets = inputs.cuda(), targets.cuda()
        bsz = inputs.size(0)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), bsz)

        _, predicted = outputs.max(1)
        acc = predicted.eq(targets).sum().item() / bsz
        accuracy.update(acc, bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % 50 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  "acc {acc.val:.3f} ({acc.avg:.3f})".format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, acc=accuracy))
            sys.stdout.flush()

# test the dataset
def test(epoch):
    global best_acc
    model.eval()
    losses = AverageMeter()
    accuracy = AverageMeter()

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            bsz = inputs.size(0)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
            acc = predicted.eq(targets).sum().item() / bsz

            losses.update(loss.item(), bsz)
            accuracy.update(acc, bsz)
            # if (idx + 1) % print_freq == 0:
            #     print('Test: [{0}][{1}/{2}]\t'
            #           'loss {loss.val:.3f} ({loss.avg:.3f})\t'
            #           "acc {acc.val:.3f} ({acc.avg:.3f})".format(
            #         epoch, idx + 1, len(test_loader), loss=losses, acc=accuracy))
            #     sys.stdout.flush()
    return losses, accuracy

def main(epoch):
    for e in range(epoch+1):
        start = time.time()
        train(e)
        print("epoch {}, total time {:.2f}".format(e, time.time()-start))
        loss, accuracy = test(e)
        scheduler.step()
        if e % print_freq ==0 :
            print('Test: [{0}/{1}]\t'
                   'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      "acc {acc.val:.3f} ({acc.avg:.3f})".format(
                    e + 1, epoch+1, loss=loss, acc=accuracy))
            sys.stdout.flush()

        if e % save_freq == 0:
            print('==> Saving...')
            model_without_ddp = model.module
            state = {
                'model': model_without_ddp.state_dict(),
                'criterion': criterion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': e,
            }
            save_file = os.path.join(model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=e))
            torch.save(state, save_file)
            # help release GPU memory
            del state
        torch.cuda.empty_cache()
    return loss.avg, accuracy.avg



def visualize(model, data_name='Lake_2018', task='supervised', acc = 1.0):
    fig_path = os.path.join('figures', task+'-'+data_name)
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    dataset = BioInfoDataset(data_name)
    x, y = dataset[0:]
    num_classes = len(np.unique(y))
    with torch.no_grad():
        hiddens = model(torch.from_numpy(x).cuda(), layer=6)
    hiddens = hiddens.cpu().numpy()
    plt.figure()
    # pca
    hyp.plot(hiddens, '.', reduce='PCA', ndims=2, hue=y, title='{}-PCA-{:.3f}'.format(data_name, acc))
    plt.savefig(os.path.join(fig_path,'pca.png'))
    ##1. tsne

    hyp.plot(hiddens, '.', reduce='TSNE', ndims=2, hue=y, title='{}-TSNE-{:.3f}'.format(data_name, acc))
    plt.savefig(os.path.join(fig_path, 'tsne.png'))
    ### UMAP
    hyp.plot(hiddens, '.', reduce='UMAP', ndims=2, hue=y, title='{}-UMAP-{:.3f}'.format(data_name, acc))
    plt.savefig(os.path.join(fig_path,'umap.png'))





if __name__ == "__main__":
    for data_name in ['Campbell', 'Chen', 'Tasic18', 'Tosches_lizard', 'Tosches_turtle']:

        train_loader, test_loader = prepare_data(batch_size, data_name)
        num_classes = train_loader.dataset[0:][1].max() + 1
        model, criterion, optimizer, scheduler = set_model(num_classes)
        model_name = 'Supervised_{}_lr_{}_decay_{}_bsz_{}'.format(arch, learning_rate,
                                                                      weight_decay, batch_size)
        model_folder = os.path.join("checkpoints", model_name)
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        loss, acc = main(100)
        print('testing acc :{:.3f}'.format(acc))
        visualize(model, data_name, acc)


