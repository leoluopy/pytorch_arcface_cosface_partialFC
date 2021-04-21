import logging
import os

import torch
import torch.distributed as dist
from torch.nn import Module
from torch.nn.functional import normalize, linear
from torch.nn.parameter import Parameter


class PartialFC(Module):
    """
    Author: {Xiang An, Yang Xiao, XuHan Zhu} in DeepGlint,
    Partial FC: Training 10 Million Identities on a Single Machine
    See the original paper:
    https://arxiv.org/abs/2010.05222
    """

    @torch.no_grad()
    def __init__(self, rank, local_rank, world_size, batch_size, resume,
                 margin_softmax, num_classes, sample_rate=1.0, embedding_size=512, prefix="./"):
        super(PartialFC, self).__init__()
        #
        self.num_classes: int = num_classes
        self.rank: int = rank
        self.local_rank: int = local_rank
        self.device: torch.device = torch.device("cuda:{}".format(self.local_rank))
        self.world_size: int = world_size
        self.batch_size: int = batch_size
        self.margin_softmax: callable = margin_softmax
        self.sample_rate: float = sample_rate
        self.embedding_size: int = embedding_size
        self.prefix: str = prefix
        self.num_local: int = num_classes // world_size + int(rank < num_classes % world_size)
        self.class_start: int = num_classes // world_size * rank + min(rank, num_classes % world_size)
        self.num_sample: int = int(self.sample_rate * self.num_local)

        self.weight_name = os.path.join(self.prefix, "rank:{}_softmax_weight.pt".format(self.rank))
        self.weight_mom_name = os.path.join(self.prefix, "rank:{}_softmax_weight_mom.pt".format(self.rank))

        if resume:
            try:
                self.weight: torch.Tensor = torch.load(self.weight_name)
                logging.info("softmax weight resume successfully!")
            except (FileNotFoundError, KeyError, IndexError):
                self.weight = torch.normal(0, 0.01, (self.num_local, self.embedding_size), device=self.device)
                logging.info("softmax weight resume fail!")

            try:
                self.weight_mom: torch.Tensor = torch.load(self.weight_mom_name)
                logging.info("softmax weight mom resume successfully!")
            except (FileNotFoundError, KeyError, IndexError):
                self.weight_mom: torch.Tensor = torch.zeros_like(self.weight)
                logging.info("softmax weight mom resume fail!")
        else:
            self.weight = torch.normal(0, 0.01, (self.num_local, self.embedding_size), device=self.device)
            self.weight_mom: torch.Tensor = torch.zeros_like(self.weight)
            logging.info("softmax weight init successfully!")
            logging.info("softmax weight mom init successfully!")
        self.stream: torch.cuda.Stream = torch.cuda.Stream(local_rank)

        self.index = None
        if int(self.sample_rate) == 1:
            self.update = lambda: 0
            self.sub_weight = Parameter(self.weight)
            self.sub_weight_mom = self.weight_mom
        else:
            self.sub_weight = Parameter(torch.empty((0, 0)).cuda(local_rank))

    def save_params(self):
        torch.save(self.weight.data, self.weight_name)
        torch.save(self.weight_mom, self.weight_mom_name)

    @torch.no_grad()
    def sample(self, total_label):
        index_positive = (self.class_start <= total_label) & (total_label < self.class_start + self.num_local)
        total_label[~index_positive] = -1
        total_label[index_positive] -= self.class_start
        if int(self.sample_rate) != 1:
            positive = torch.unique(total_label[index_positive], sorted=True)
            if self.num_sample - positive.size(0) >= 0:
                perm = torch.rand(size=[self.num_local], device=self.device)
                perm[positive] = 2.0
                index = torch.topk(perm, k=self.num_sample)[1]
                index = index.sort()[0]
            else:
                index = positive
            self.index = index
            total_label[index_positive] = torch.searchsorted(index, total_label[index_positive])
            self.sub_weight = Parameter(self.weight[index])
            self.sub_weight_mom = self.weight_mom[index]

    def forward(self, total_features, norm_weight):
        torch.cuda.current_stream().wait_stream(self.stream)
        logits = linear(total_features, norm_weight)
        return logits

    @torch.no_grad()
    def update(self):
        self.weight_mom[self.index] = self.sub_weight_mom
        self.weight[self.index] = self.sub_weight

    def prepare(self, label, optimizer):
        with torch.cuda.stream(self.stream):
            total_label = torch.zeros(
                size=[self.batch_size * self.world_size], device=self.device, dtype=torch.long)
            # gather label 目标是获取label判断是不是本张卡的fc 处理,选择fc .
            dist.all_gather(list(total_label.chunk(self.world_size, dim=0)), label)
            self.sample(total_label)
            # 跟新优化器的被优化参数,以及参数的动量.
            optimizer.state.pop(optimizer.param_groups[-1]['params'][0], None)
            optimizer.param_groups[-1]['params'][0] = self.sub_weight
            optimizer.state[self.sub_weight]['momentum_buffer'] = self.sub_weight_mom
            norm_weight = normalize(self.sub_weight)
            return total_label, norm_weight

    def forward_backward(self, label, features, optimizer, backbone):
        # norm_weight 本张卡的 中心权重, total_label 本个batch中所有数据标签[中心不在本卡的被标记为-1]
        total_label, norm_weight = self.prepare(label, optimizer)
        total_features = torch.zeros(
            size=[self.batch_size * self.world_size, self.embedding_size], device=self.device)
        dist.all_gather(list(total_features.chunk(self.world_size, dim=0)), features.data)
        total_features.requires_grad = True

        # 矩阵相乘得到cosine 相似度,含有负数项[乘-1的原因]
        logits = self.forward(total_features, norm_weight)
        # 加magin 并乘 scale
        logits = self.margin_softmax(logits, total_label)
        with torch.no_grad():
            # 取响应最大的一个中心的权重 1/C, 最后维度为 B
            max_fc = torch.max(logits, dim=1, keepdim=True)[0]
            # 取多张卡中响应最大的中心权重 , 非选中样本, 因为乘以-1, 以及中心采用[0,0.0.1]正态分布初始化的原因,能被过滤掉
            dist.all_reduce(max_fc, dist.ReduceOp.MAX)

            # for numerical stability , this is a exp normalised implementation
            logits_exp = torch.exp(logits - max_fc)
            logits_sum_exp = logits_exp.sum(dim=1, keepdims=True)
            dist.all_reduce(logits_sum_exp, dist.ReduceOp.SUM)

            # calculate prob
            logits_exp.div_(logits_sum_exp)

            # get one-hot
            grad = logits_exp
            index = torch.where(total_label != -1)[0]
            one_hot = torch.zeros(size=[index.size()[0], grad.size()[1]], device=grad.device)
            one_hot.scatter_(1, total_label[index, None], 1)

            # calculate loss , 公式为: softmax loss = -y * log(softmax(x))
            loss = torch.zeros(grad.size()[0], 1, device=grad.device)
            # 取标签位置的预计loss项
            loss[index] = grad[index].gather(1, total_label[index, None])
            # 多卡求和，loss项，合并在一起．　
            dist.all_reduce(loss, dist.ReduceOp.SUM)
            loss_v = loss.clamp_min_(1e-30).log_().mean() * (-1)

            # calculate grad
            grad[index] -= one_hot
            grad.div_(self.batch_size * self.world_size)

        # 分段求导的写法，先求softmax 的导数存在grad中，然后传入logits的求导调用 ,
        # 断点观察backbone.module.fc.weight.grad和total_features.grad，导数只到total_features.grad ,因为all_gather把计算图打断了，退出函数后，外面再接棒feature.backward()
        # NOTE: optimizer.param_groups[0]['params'][0].grad 计算图没受到影响，会计算完毕．
        logits.backward(grad)
        if total_features.grad is not None:
            # toal_feature的梯度已经计算过了，调用detach防止本个batch 中被再次计算．
            total_features.grad.detach_()
        x_grad: torch.Tensor = torch.zeros_like(features, requires_grad=True)
        # feature gradient all-reduce
        dist.reduce_scatter(x_grad, list(total_features.grad.chunk(self.world_size, dim=0)))
        x_grad = x_grad * self.world_size
        # backward backbone
        return x_grad, loss_v
