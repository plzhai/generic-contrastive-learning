import torch
from torch import nn
from .alias_multinomial import AliasMethod
import math
import torch.nn.functional as F

class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_l', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, l, ab, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = l.size(0)
        outputSize = self.memory_l.size(0)
        inputSize = self.memory_l.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
        weight_l = weight_l.view(batchSize, K + 1, inputSize)
        out_ab = torch.bmm(weight_l, ab.view(batchSize, inputSize, 1))
        # sample
        weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
        weight_ab = weight_ab.view(batchSize, K + 1, inputSize)
        out_l = torch.bmm(weight_ab, l.view(batchSize, inputSize, 1))

        if self.use_softmax:
            out_ab = torch.div(out_ab, T)
            out_l = torch.div(out_l, T)
            out_l = out_l.contiguous()
            out_ab = out_ab.contiguous()
        else:
            out_ab = torch.exp(torch.div(out_ab, T))
            out_l = torch.exp(torch.div(out_l, T))
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_l < 0:
                self.params[2] = out_l.mean() * outputSize
                Z_l = self.params[2].clone().detach().item()
                print("normalization constant Z_l is set to {:.1f}".format(Z_l))
            if Z_ab < 0:
                self.params[3] = out_ab.mean() * outputSize
                Z_ab = self.params[3].clone().detach().item()
                print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))
            # compute out_l, out_ab
            out_l = torch.div(out_l, Z_l).contiguous()
            out_ab = torch.div(out_ab, Z_ab).contiguous()

        # # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)

        return out_l, out_ab

# =========================
# loss from Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss by Tengyu Ma
# =========================

class Population_spectral():
    def __init__(self):
        super(Population_spectral, self).__init__()
    def forward(self):
        return

# =========================
# loss from SimCLR
# =========================
import torch.distributed as dist

class NT_Xent(nn.Module):
    def __init__(self, temperature):
        super(NT_Xent, self).__init__()
        self.temperature = temperature
        #self.world_size = world_size


        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        self.batch_size = z_i.shape[0]
        self.mask = self.mask_correlated_samples(self.batch_size)
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)
        #if self.world_size > 1:
        #    z = torch.cat(GatherLayer.apply(z), dim=0)
        #z = torch.cat(z, dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

class FlatNCE(nn.Module):
    def __init__(self, temperature):
        super(FlatNCE, self).__init__()
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, z_i, z_j):
        batch_size = z_i.size()[0]
        features = torch.cat([z_i,z_j], dim=0)
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        #labels = labels.to(self.args.device)
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)

        labels = torch.zeros(positives.shape[0], dtype=torch.long).to(positives.device) # -

        #        logits = logits / self.args.temperature #-
        logits = (negatives - positives) / self.temperature  # (512,510) #+
        v = torch.logsumexp(logits, dim=1, keepdim=True)  # (512,1)
        loss_vec = torch.exp(v - v.detach())

        assert loss_vec.shape == (len(logits), 1)
        dummy_logits = torch.cat([torch.zeros(logits.size(0), 1).to(positives.device), logits], 1)
        loss = loss_vec.mean() - 1 + self.criterion(dummy_logits, labels).detach()  # +
        return loss


class InfoNCE(nn.Module):
    def __init__(self, temperature):
        super(InfoNCE, self).__init__()
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, z_i, z_j):
        batch_size = z_i.size()[0]
        features = torch.cat([z_i, z_j], dim=0)
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        #labels = labels.to(self.args.device)
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        logits = logits / self.temperature
        loss = self.criterion(logits, labels)
        return loss


from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss

class Memory(nn.Module):
    def __init__(self, args):
        super(Memory, self).__init__()
        self.contrast = NCEAverage(args.feat_dim, args.n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)
        self.criterion_l = NCESoftmaxLoss() if args.softmax else NCECriterion(args.n_data)
        self.criterion_ab = NCESoftmaxLoss() if args.softmax else NCECriterion(args.n_data)
        # if torch.cuda.is_available():
        #
        #     self.contrast = self.contrast.cuda(args.gpu)
        #     self.criterion_ab = self.criterion_ab.cuda(args.gpu)
        #     self.criterion_l = self.criterion_l.cuda(args.gpu)

    def forward(self, feat_l, feat_ab, index):
        out_l, out_ab = self.contrast(feat_l, feat_ab, index)
        l_loss = self.criterion_l(out_l)
        ab_loss = self.criterion_ab(out_ab)
        l_prob = out_l[:, 0].mean()
        ab_prob = out_ab[:, 0].mean()
        loss = l_loss + ab_loss
        return loss




# =========================
# InsDis and MoCo
# =========================

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class MemoryInsDis(nn.Module):
    """Memory bank with instance discrimination"""
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(MemoryInsDis, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax
        self.normalize = Normalize(2)

        self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y, idx=None):
        # normalize
        x = self.normalize(x)

        K = int(self.params[0].item())
        T = self.params[1].item()
        Z = self.params[2].item()
        momentum = self.params[3].item()

        batchSize = x.size(0)
        outputSize = self.memory.size(0)
        inputSize = self.memory.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        # sample
        weight = torch.index_select(self.memory, 0, idx.view(-1))
        weight = weight.view(batchSize, K + 1, inputSize)
        out = torch.bmm(weight, x.view(batchSize, inputSize, 1))

        if self.use_softmax:
            out = torch.div(out, T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, T))
            if Z < 0:
                self.params[2] = out.mean() * outputSize
                Z = self.params[2].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            weight_pos = torch.index_select(self.memory, 0, y.view(-1))
            weight_pos.mul_(momentum)
            weight_pos.add_(torch.mul(x, 1 - momentum))
            weight_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(weight_norm)
            self.memory.index_copy_(0, y, updated_weight)

        return out


class MemoryAdv(nn.Module):
    """Memory bank with instance discrimination"""
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(MemoryAdv, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax
        self.normalize = Normalize(2)

        self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, orig, x, y, idx=None):
        # normalize
        x = self.normalize(x)
        orig = self.normalize(orig)

        K = int(self.params[0].item())
        T = self.params[1].item()
        Z = self.params[2].item()
        momentum = self.params[3].item()

        batchSize = x.size(0)
        outputSize = self.memory.size(0)
        inputSize = self.memory.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        # sample
        weight = torch.index_select(self.memory, 0, idx.view(-1))
        weight = weight.view(batchSize, K + 1, inputSize)
        out = torch.bmm(weight, orig.view(batchSize, inputSize, 1))

        if self.use_softmax:
            out = torch.div(out, T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, T))
            if Z < 0:
                self.params[2] = out.mean() * outputSize
                Z = self.params[2].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            weight_pos = torch.index_select(self.memory, 0, y.view(-1))
            weight_pos.mul_(momentum)
            weight_pos.add_(torch.mul(x, 1 - momentum))
            weight_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(weight_norm)
            self.memory.index_copy_(0, y, updated_weight)

        return out

class MemoryBank(nn.Module):
    """Memory bank with instance discrimination"""
    def __init__(self, inputSize, outputSize, K, momentum=0.5, use_softmax=False):
        super(MemoryBank, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.momentum = momentum
        self.use_softmax = use_softmax
        self.normalize = Normalize(2)

        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y, idx=None):
        # normalize
        x = self.normalize(x)

        K = self.K
        momentum = self.momentum

        batchSize = x.size(0)
        inputSize = self.memory.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        # sample
        weight = torch.index_select(self.memory, 0, idx.view(-1))
        weight = weight.view(batchSize, K + 1, inputSize)
        # # update memory
        with torch.no_grad():
            weight_pos = torch.index_select(self.memory, 0, y.view(-1))
            weight_pos.mul_(momentum)
            weight_pos.add_(torch.mul(x, 1 - momentum))
            weight_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(weight_norm)
            self.memory.index_copy_(0, y, updated_weight)
        return weight

class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, inputSize, outputSize, K, T=0.07, use_softmax=False):
        super(MemoryMoCo, self).__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax
        self.normalize = Normalize(2)
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, inputSize))

    def forward(self, q, k):
        q = self.normalize(q)
        k = self.normalize(k)

        batchSize = q.shape[0]
        k = k.detach()

        Z = self.params[0].item()
        # pos logit
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        # neg logit
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)

        out = torch.cat((l_pos, l_neg), dim=1)

        if self.use_softmax:
            out = torch.div(out, self.T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, self.T))
            if Z < 0:
                self.params[0] = out.mean() * self.outputSize
                Z = self.params[0].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + batchSize) % self.queueSize

        return out

def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    #for p1, p2 in zip(model.parameters(), model_ema.parameters()):
    #    p2.data.mul_(m).add_(1-m, p1.detach().data)
    for param_q, param_k in zip(model.parameters(), model_ema.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)
        #param_k.requires_grad = False  # not update by gradient
