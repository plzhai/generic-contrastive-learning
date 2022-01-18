import os
class options():
    def __init__(self):
        pass

args = options()
args.epsilon = 0.0314 # maximum perturbation of adversaries (8 / 255 for cifar10)
args.alpha = 0.007 #(2/255)
args.max_iters = 10
args.random_start = True # True for PGD
args.dataset = 'cifar10'
if args.dataset == 'cifar10':
    args.n_data = 50000
args.min_val = 0.
args.max_val = 1.

args.eval_freq = 10
args.save_freq = 50
args.print_freq = 10
args.warmup = 10

args.batch_size = 256

args.arch = 'resnet18'
args.feat_dim = 128


args.nce_t = 0.1
args.nce_k = 16384
args.nce_m = 0.5
args.softmax = True
# optimizer
args.weight_decay = 1.e-4
# lr: 0.06 for batch 512 (or 0.03 for batch 256)
args.learning_rate = 0.1
args.momentum = 0.9
args.epochs = 800

# scheduler
args.lr_decay_rate = 0.1
args.lr_decay_epochs = [120,160,200]

args.model_path = 'checkpoints'
if not os.path.exists(args.model_path):
    os.mkdir(args.model_path)


args.method = 'moco'# 'memory', 'adv'

args.direction = 'random'# [None, 'random', 'fgsm']


# evaluate model

args.knn_k = 200
args.knn_t = 0.1
args.classifier = 'knn'# or 'logistic'