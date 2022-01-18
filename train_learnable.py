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
    transform = networks.__getattribute__('view_learner')('bernoulli', 'alexnet')
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
    regulariztion = nn.MSELoss() if args.reg == 'l2' else nn.L1Loss()

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = convert_model(model)
        model = model.cuda()
        transform = transform.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    learning_rate = args.learning_rate * args.batch_size / 256
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer_t = optim.SGD(transform.parameters(), lr=1e-3, momentum=0.999, weight_decay=args.weight_decay)
    return model, transform, criterion, regulariztion, optimizer, optimizer_t

# adversarial training
from models.attack_libs import project
def perturb(model, inputs, criterion, regularization, random_start=True):
    if random_start:
        rand_perturb = torch.FloatTensor(inputs.shape).uniform_(
            -args.epsilon, args.epsilon)
        rand_perturb = rand_perturb.float().cuda()
        x = inputs.float().clone() + rand_perturb
        x = torch.clamp(x, args.min_val, args.max_val)
    else:
        x = inputs.clone()
    x.requires_grad = True
    model.eval()

    with torch.enable_grad():
        out_target = model(inputs)
        for _iter in range(args.max_iters):
            model.zero_grad()
            # inputs = torch.cat((x, target), axis=0)
            outputs = model(x)
            loss = criterion(outputs, out_target) - args.gamma * regularization(outputs, out_target)
            grads = torch.autograd.grad(loss, x, grad_outputs=None, only_inputs=True, retain_graph=False)[0]
            scaled_g = grads.data
            #scaled_g = torch.sign(grads.data)
            x.data += args.alpha * scaled_g
            x = torch.clamp(x, args.min_val, args.max_val)
            x = project(x, inputs, args.epsilon)
    model.train()
    return x.detach()
class transform_(nn.Module):
    def __init__(self, temperature, _type='conv'):
        super(transform_, self).__init__()
        if _type == 'conv':
            self.layer = nn.Sequential(
                nn.Conv2d(3, 3, 3, padding='same'),
                nn.ReLU(),
                nn.Conv2d(3, 3, 3, padding='same'),
            )
        else:
            self.layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3*32*32, 1024),
            )
        self.temp = temperature
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        w_e = self.layer(x)
        weight = self.sigmoid(w_e/self.temp)
        x_t = x * weight
        kl = weight.mean()# to maximize
        return x_t, kl


def train(epoch, model, transform, criterion, regularization, optimizer, optimizer_t, scheduler, scheduler_t, train_loader, args):
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
        bsz = inputs.size(0)
        if torch.cuda.is_available():
            inputs = inputs.cuda(non_blocking=True)
        # -----------------------
        #  Train transform layer
        # -----------------------
        #advinputs = perturb(model, inputs, criterion, regularization, random_start=True)
        model.eval()
        transform.train()
        advinputs, kl_loss = transform(inputs)
        proj_l, proj_ab = model(inputs), model(advinputs)
        loss_t = criterion(proj_l, proj_ab) + args.gamma * kl_loss
        optimizer_t.zero_grad()
        (-loss_t).backward(retain_graph=True)
        optimizer_t.step()
        scheduler_t.step()
        losses_t.update(kl_loss.item(), bsz)
        # -----------------------
        #  Train model
        # -----------------------
        model.train()
        transform.eval()
        proj_l, proj = model(inputs), model(transform(inputs)[0].detach())
        #proj = model(transform(inputs)[0].detach())
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
            logging['loss_t'].append(losses_t.avg)


def main(args):
    # set early stopping
    patience = 3
    cur = 0
    best_acc1 = 0

    model, transform, criterion, regularization, optimizer, optimizer_t = set_model(args)
    scheduler = utils.LR_Scheduler(optimizer, args.warmup, 0, args.epochs,
                                   args.learning_rate*args.batch_size/256, 0, len(train_loader))
    scheduler_t = utils.LR_Scheduler(optimizer_t, args.warmup, 0, args.epochs,
                                   args.learning_rate * args.batch_size / 256, 0, len(train_loader))
    args.start_epoch = 1
    for epoch in range(args.start_epoch, args.epochs+1):
        #utils.adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")
        time1 = time.time()
        train(epoch, model, transform, criterion, regularization, optimizer, optimizer_t, scheduler, scheduler_t, train_loader, args)
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
    args.batch_size = 512
    args.epochs = 200
    transform_train = transforms.Compose([
        #transforms.RandomResizedCrop(32),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    train_loader = utils.prepare_data(transform_train, args.batch_size, 'cifar10')
    args.eval_freq = 10
    args.nce_t = 0.1
    args.gamma = 10.
    args.reg = 'l2'
    args.method = 'transform'
    #for head in ['1fc', '2fc', '2bn']:
    args.head = '2fc'
    args.arch = 'alexnet_cifar'
    for gamma in [0.01, 0.1, 1., 10.,0.0]:
        args.gamma = gamma
        args.model_name = "{}_{}_eps_{}_bsz_{}_t_{}_lr_{}".format(args.gamma, args.method, args.epsilon, args.batch_size, args.nce_t, args.learning_rate)
        logging = {'epoch': [], 'acc@1': [], 'acc@5': [], 'loss':[], 'loss_t':[]}
        main(args)
