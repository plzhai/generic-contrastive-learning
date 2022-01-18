"""
this code is modified from
https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks
https://github.com/louis2889184/pytorch-adversarial-training
https://github.com/MadryLab/robustness
https://github.com/yaodongyu/TRADES
"""

import torch
import torch.nn.functional as F
from NCE.NCEAverage import InfoNCE


def project(x, original_x, epsilon, _type='linf'):
    if _type == 'linf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon

        x = torch.max(torch.min(x, max_x), min_x)
    else:
        raise NotImplementedError

    return x


class FastGradientSignUntargeted():
    """
        Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
        with iterative grad sign updates
    """

    def __init__(self, model, linear, epsilon, alpha, min_val, max_val, max_iters, _type='linf'):

        # Model
        self.model = model
        self.linear = linear
        # Maximum perturbation
        self.epsilon = epsilon
        # Movement multiplier per iteration
        self.alpha = alpha
        # Minimum value of the pixels
        self.min_val = min_val
        # Maximum value of the pixels
        self.max_val = max_val
        # Maximum numbers of iteration to generated adversaries
        self.max_iters = max_iters
        # The perturbation of epsilon
        self._type = _type

    def perturb(self, original_images, labels, reduction4loss='mean', random_start=True):
        # original_images: values are within self.min_val and self.max_val
        # The adversaries created from random close points to the original data
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = rand_perturb.cuda()
            x = original_images.clone() + rand_perturb
            x = torch.clamp(x, self.min_val, self.max_val)
        else:
            x = original_images.clone()
        x.requires_grad = True

        self.model.eval()
        if not self.linear == 'None':
            self.linear.eval()

        with torch.enable_grad():
            for _iter in range(self.max_iters):
                self.model.zero_grad()
                if not self.linear == 'None':
                    self.linear.zero_grad()
                if self.linear == 'None':
                    outputs = self.model(x)
                else:
                    outputs = self.linear(self.model(x))
                loss = F.cross_entropy(outputs, labels, reduction=reduction4loss)
                grad_outputs = None
                grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs, only_inputs=True, retain_graph=False)[0]
                if self._type == 'linf':
                    scaled_g = torch.sign(grads.data)
                x.data += self.alpha * scaled_g

                x = torch.clamp(x, self.min_val, self.max_val)
                x = project(x, original_images, self.epsilon, self._type)

        return x.detach()

from NCE.NCEAverage import InfoNCE
class RepresentationAdv():
    def __init__(self, epsilon, alpha, min_val, max_val, max_iters, _type='linf'):
        # Model
        # Maximum perturbation
        self.epsilon = epsilon
        # Movement multiplier per iteration
        self.alpha = alpha
        # Minimum value of the pixels
        self.min_val = min_val
        # Maximum value of the pixels
        self.max_val = max_val
        # Maximum numbers of iteration to generated adversaries
        self.max_iters = max_iters
        # The perturbation of epsilon
        self._type = _type


    def get_adv(self, model, criterion, original_images, target, random_start=True):
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = rand_perturb.float().cuda()
            x = original_images.float().clone() + rand_perturb
            x = torch.clamp(x, self.min_val, self.max_val)
        else:
            x = original_images.clone()
        x.requires_grad = True
        model.eval()
        criterion.eval()

        with torch.enable_grad():
            out_target = model(target)
            for _iter in range(self.max_iters):
                model.zero_grad()
                #inputs = torch.cat((x, target), axis=0)
                outputs = model(x)
                loss = criterion(outputs, out_target)
                grads = torch.autograd.grad(loss, x, grad_outputs=None, only_inputs=True, retain_graph=False)[0]
                if self._type == 'linf':
                    scaled_g = torch.sign(grads.data)
                x.data += self.alpha * scaled_g
                x = torch.clamp(x, self.min_val, self.max_val)
                x = project(x, original_images, self.epsilon, self._type)
        model.train()
        criterion.train()
        return x.detach()
