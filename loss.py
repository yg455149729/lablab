# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# import torch.distributed as dist
# from torchvision.transforms import Resize
import torchvision
import random
from functools import partial
from torch.nn.modules.loss import _Loss
# import os
# os.environ["RANK"]='0'
def _reduce(loss, reduction, **kwargs):
    if reduction == 'none':
        ret = loss
    elif reduction == 'mean':
        normalizer = loss.numel()
        if kwargs.get('normalizer', None):
            normalizer = kwargs['normalizer']
        ret = loss.sum() / normalizer
    elif reduction == 'sum':
        ret = loss.sum()
    else:
        raise ValueError(reduction + ' is not valid')
    return ret
class BaseLoss(_Loss):
    # do not use syntax like `super(xxx, self).__init__,
    # which will cause infinited recursion while using class decorator`
    def __init__(self,
                 name='base',
                 reduction='none',
                 loss_weight=1.0):
        r"""
        Arguments:
            - name (:obj:`str`): name of the loss function
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
        """
        _Loss.__init__(self, reduction=reduction)
        self.loss_weight = loss_weight
        self.name = name

    def __call__(self, input, target, reduction_override=None, normalizer_override=None, **kwargs):
        r"""
        Arguments:
            - input (:obj:`Tensor`)
            - reduction (:obj:`Tensor`)
            - reduction_override (:obj:`str`): choice of 'none', 'mean', 'sum', override the reduction type
            defined in __init__ function
            - normalizer_override (:obj:`float`): override the normalizer when reduction is 'mean'
        """
        reduction = reduction_override if reduction_override else self.reduction
        assert (normalizer_override is None or reduction == 'mean'), \
            f'normalizer is not allowed when reduction is {reduction}'
        loss = _Loss.__call__(self, input, target, reduction, normalizer=normalizer_override, **kwargs)
        return loss * self.loss_weight

    def forward(self, input, target, reduction, normalizer=None, **kwargs):
        raise
def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps
class GeneralizedCrossEntropyLoss(BaseLoss):
    def __init__(self,
                 name='generalized_cross_entropy_loss',
                 reduction='none',
                 loss_weight=1.0,
                 activation_type='softmax',
                 ignore_index=-1,):
        BaseLoss.__init__(self,
                          name=name,
                          reduction=reduction,
                          loss_weight=loss_weight)
        self.activation_type = activation_type
        self.ignore_index = ignore_index
class EqualizedFocalLoss(GeneralizedCrossEntropyLoss):
    def __init__(self,
                 name='equalized_focal_loss',
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=-1,
                 num_classes=31,
                 focal_gamma=1.0,
                 focal_alpha=0.5,
                 scale_factor=8.0,
                 fpn_levels=5):
        activation_type = 'sigmoid'
        GeneralizedCrossEntropyLoss.__init__(self,
                                             name=name,
                                             reduction=reduction,
                                             loss_weight=loss_weight,
                                             activation_type=activation_type,
                                             ignore_index=ignore_index)

        # cfg for focal loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # ignore bg class and ignore idx
        self.num_classes = num_classes

        # cfg for efl loss
        self.scale_factor = scale_factor
        # initial variables
        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))
        self.register_buffer('pos_neg', torch.ones(self.num_classes))

        # grad collect
        self.grad_buffer = []
        self.fpn_levels = fpn_levels

        # logger.info(f"build EqualizedFocalLoss, focal_alpha: {focal_alpha}, focal_gamma: {focal_gamma}, \
        #             scale_factor: {scale_factor}")

    def forward(self, input, target, reduction, normalizer=None):
        self.n_c = input.shape[-1]
        self.input = input.reshape(-1, self.n_c)
        self.target = target.reshape(-1)
        self.n_i, _ = self.input.size()

        # def expand_label(pred, gt_classes):
        #     target = pred.new_zeros(self.n_i, self.n_c + 1)
        #     target[torch.arange(self.n_i), gt_classes] = 1
        #     return target[:, 1:]

        # expand_target = expand_label(self.input, self.target)
        expand_target = target

        inputs = self.input
        targets = expand_target
        # self.cache_mask = sample_mask
        # self.cache_target = expand_target

        pred = torch.sigmoid(inputs)
        pred_t = pred * targets + (1 - pred) * (1 - targets)
        assert not torch.any(torch.isnan(pred_t))

        map_val = 1 - self.pos_neg.detach()
        dy_gamma = self.focal_gamma + self.scale_factor * map_val
        # focusing factor
        ff = dy_gamma.view(1, -1).expand(self.n_i, self.n_c)

        # weighting factor
        wf = ff / self.focal_gamma

        # ce_loss
        ce_loss = -torch.log(pred_t+1e-6)
        assert not torch.any(torch.isinf(ce_loss))
        assert not torch.any(torch.isnan(ce_loss))
        # print(ce_loss)
        # print(pred_t)
        # print(ff)
        # print(wf)
        cls_loss = ce_loss * torch.pow((1 - pred_t), ff.cuda()) * wf.cuda()
        assert not torch.any(torch.isinf(cls_loss))
        assert not torch.any(torch.isnan(cls_loss))
        # to avoid an OOM error
        # torch.cuda.empty_cache()

        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
            cls_loss = alpha_t * cls_loss

        if normalizer is None:
            normalizer = self.n_i
        return _reduce(cls_loss, reduction, normalizer=normalizer)

    def collect_grad(self, grad_in):
        bs = grad_in.shape[0]
        self.grad_buffer.append(grad_in.detach().permute(0, 2, 3, 1).reshape(bs, -1, self.num_classes))
        if len(self.grad_buffer) == self.fpn_levels:
            target = self.cache_target[self.cache_mask]
            grad = torch.cat(self.grad_buffer[::-1], dim=1).reshape(-1, self.num_classes)

            grad = torch.abs(grad)[self.cache_mask]
            pos_grad = torch.sum(grad * target, dim=0)
            neg_grad = torch.sum(grad * (1 - target), dim=0)

            # allreduce(pos_grad)
            # allreduce(neg_grad)

            self.pos_grad += pos_grad
            self.neg_grad += neg_grad
            self.pos_neg = torch.clamp(self.pos_grad / (self.neg_grad + 1e-10), min=0, max=1)

            self.grad_buffer = []

def useful_ratio(box1, box2, xywh=True,eps=1e-7):
    '''

    :param bboxes_a: gtbox.[num,4]
    :param bboxes_b: 感受野面积[num,4]
    :param xyxy:
    :return:
    '''

    # def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)
        # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # print("Box1",torch.stack((b1_x1,b1_y1,b1_x2,b1_y2),dim=1))
    # print("Box2", torch.stack((b2_x1, b2_y1, b2_x2, b2_y2), dim=1))

    return inter/(w2*h2)

class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()

class EQLv2(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=4,  # 1203 for lvis v1.0, 1230 for lvis v0.5
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 vis_grad=False,
                 test_with_obj=True):
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True

        # cfg for eqlv2
        self.vis_grad = vis_grad
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        # initial variables
        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))
        # At the beginning of training, we set a high value (eg. 100)
        # for the initial gradient ratio so that the weight for pos gradients and neg gradients are 1.
        self.register_buffer('pos_neg', torch.ones(self.num_classes) * 100)

        self.test_with_obj = test_with_obj

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        self.n_i, self.n_c = cls_score.size()

        self.gt_classes = label
        # print(label)
        self.pred_class_logits = cls_score

        # def expand_label(pred, gt_classes):
        #     target = pred.new_zeros(self.n_i, self.n_c)
        #     target[torch.arange(self.n_i), gt_classes] = 1
        #     return target

        # target = expand_label(cls_score, label)
        target = label

        pos_w, neg_w = self.get_weight(cls_score)

        # print(pos_w.shape)
        weight = pos_w * target + neg_w * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(cls_score, target,
                                                      reduction='none')
        cls_loss = torch.sum(cls_loss * weight) / self.n_i

        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())

        return self.loss_weight * cls_loss

    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1
        return num_channel

    def get_activation(self, cls_score):
        cls_score = torch.sigmoid(cls_score)
        n_i, n_c = cls_score.size()
        bg_score = cls_score[:, -1].view(n_i, 1)
        if self.test_with_obj:
            cls_score[:, :-1] *= (1 - bg_score)
        return cls_score

    def collect_grad(self, cls_score, target, weight):
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)

        # do not collect grad for objectiveness branch [:-1]
        pos_grad = torch.sum(grad * target * weight, dim=0)
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)

        # dist.all_reduce(pos_grad)
        # dist.all_reduce(neg_grad)
        # print(self.pos_grad)
        # print(pos_grad)
        self.pos_grad = self.pos_grad.cuda()
        self.neg_grad = self.neg_grad.cuda()
        self.pos_grad += pos_grad
        self.neg_grad += neg_grad

        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)

    def get_weight(self, cls_score):
        # neg_w = torch.cat([self.map_func(self.pos_neg).cuda(), cls_score[...,:-1].new_ones(1)])

        neg_w = self.map_func(self.pos_neg).cuda()
        pos_w = 1 + self.alpha * (1 - neg_w)
        neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
        pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.5):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class InfoNceLoss(nn.Module):
    def __init__(self, reduction="none", info_strides=[8,16,32]):
        super(InfoNceLoss, self).__init__()
        self.reduction = reduction
        self.strides = info_strides
        # self.max_neg_dic={info_strides[0]: 35, info_strides[1]: 35, info_strides[2]: 30}
        # self.temperature = {info_strides[0]: 0.1, info_strides[1]:0.1, info_strides[2]: 1}
        self.CrossEntropy = torch.nn.CrossEntropyLoss()
        self.max_neg_dic = {info_strides[0]: 20, info_strides[1]: 20, info_strides[2]: 20}
        self.temperature = {info_strides[0]:0.2, info_strides[1]:0.2, info_strides[2]:0.2}
    def forward(self, xin, labels, stride):
        # tail_label_index = 3
        tail_loc = torch.zeros(0,2, dtype=torch.long).cuda()
        # tail_label_index_list = [6]
        # tail_label_index_list = [2,3]
        # positive_up = [0]
        tail_label_index_list = [5, 6, 8]
        new_loss = 0.0

        # print("YGYG",labels.shape)
        temperature = self.temperature[stride]
        max_neg = self.max_neg_dic[stride]
        result = torch.tensor([], requires_grad=True).cuda().reshape(0, 21)
        for tail_label_index in tail_label_index_list:
            labels_cp = labels.clone()
            labels_cp[:,2:] = labels_cp[:, 2:] / stride
            tail_labels = labels_cp[labels[:,1]==tail_label_index]
            if tail_labels.numel():
                feature = []
                neg_labels = labels_cp[(labels_cp[:,1]!=tail_label_index_list[0])
                                       & (labels_cp[:,1]!=tail_label_index_list[1])]
                neg_pair = torch.tensor([], requires_grad=True).cuda().reshape(0, 20)
                for index in range(len(tail_labels)):
                    tail_label = tail_labels[index]
                    i_neg_pair=[]
                    #YGYGneed to be modified
                    x1 = tail_label[2] - tail_label[4] * 0.5
                    y1 = tail_label[3] - tail_label[5] * 0.5

                    x2 = tail_label[2] + tail_label[4] * 0.5
                    y2 = tail_label[3] + tail_label[5] * 0.5
                    temp = torchvision.ops.roi_align(xin, torch.tensor([[tail_label[0], x1, y1, x2, y2]]).cuda(),
                                                     [7, 7])

                    temp = temp.reshape((1, -1)).squeeze()
                    temp = F.normalize(temp, dim=0)

                    feature.append(temp)

                    same_pic_neg_label = neg_labels[neg_labels[:,0] == tail_label[0]]
                    rest_num = max_neg - same_pic_neg_label.shape[0]
                    if rest_num>0:

                        diff_pic_neg_label = neg_labels[neg_labels[:,0] != tail_label[0]]
                        # rest_index = random.sample(range(diff_pic_neg_label.shape[0]), rest_num)
                        rest_index = []
                        if  diff_pic_neg_label.shape[0]!=0:
                            if(rest_num < diff_pic_neg_label.shape[0]):
                                rest_index = random.sample(range(diff_pic_neg_label.shape[0]), rest_num)

                            else:
                                rest_index = random.sample(range(diff_pic_neg_label.shape[0]), diff_pic_neg_label.shape[0])
                                size = rest_num - diff_pic_neg_label.shape[0]
                                if diff_pic_neg_label.shape[0] == 0:
                                    rest_index.extend([rest_index[-1]]*size)
                                else:
                                    heihei = np.random.randint(0, diff_pic_neg_label.shape[0], size=size)
                                    rest_index.extend(heihei)
                            # if(rest_num > diff_pic_neg_label.shape[0]):
                            #     size = rest_num - diff_pic_neg_label.shape[0]
                            #     heihei = np.random.randint(0, diff_pic_neg_label.shape[0], size=size)
                            #     rest_index.extend(heihei)
                            # else:

                            for i in rest_index:
                                same_pic_neg_label = torch.cat([same_pic_neg_label,diff_pic_neg_label[i].unsqueeze(0)],dim=0)
                        else:
                            same_pic_neg_label = torch.cat([same_pic_neg_label,same_pic_neg_label[0:20-len(same_pic_neg_label)]],dim=0)

                    for neg_label in same_pic_neg_label[0:20]:
                        # YGYGneed to be modified
                        x1 = neg_label[2] - neg_label[4] * 0.5
                        y1 = neg_label[3] - neg_label[5] * 0.5

                        x2 = neg_label[2] + neg_label[4] * 0.5
                        y2 = neg_label[3] + neg_label[5] * 0.5
                        neg_temp = torchvision.ops.roi_align(xin, torch.tensor([[neg_label[0], x1, y1, x2, y2]]).cuda(),
                                                         [7, 7])

                        neg_temp = neg_temp.reshape((1, -1)).squeeze()
                        neg_temp = F.normalize(neg_temp, dim=0)
                        similarity = (torch.cosine_similarity(temp,neg_temp,dim=0)+1)/2 / temperature
                        i_neg_pair.append(similarity)
                    i_neg_pair = torch.stack(i_neg_pair,dim=0)
                    neg_pair = torch.cat([neg_pair,i_neg_pair.unsqueeze(0)],dim=0)


                pos_similarity = torch.tensor([],requires_grad=True).cuda().reshape(0,1)
                for i in range(len(feature)):
                    random_int = len(feature)-i-1
                    similarity = (torch.cosine_similarity(feature[i],feature[random_int],dim=0)+1)/2 / temperature
                    similarity = similarity.unsqueeze(0)
                    pos_similarity =torch.cat([pos_similarity,similarity.unsqueeze(0)],dim=0)
                logits = torch.cat([pos_similarity,neg_pair],dim=1)
                result = torch.cat([result,logits],dim=0)


        if result.numel():
            contra_labels = torch.zeros(result.shape[0], dtype=torch.long).cuda()
            new_loss = self.CrossEntropy(result, contra_labels)
            return new_loss
        else:
            return 0
class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        # BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        # BCEcls = EQLv2(class_weight=torch.tensor([h['cls_pw']]))
        BCEcls = EqualizedFocalLoss()

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.infols = InfoNceLoss()
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        self.h = h

    def __call__(self, p, targets, feature=None):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        sample = []
        result = []
        # print("targets:",targets)
        # print("build_targets complete :")
        # print(tbox[0][0],tbox[1][0],tbox[2][0])
        stride = [8,16,32]
        InfoLoss = []
        if feature is not None:
            for index in range(len(feature)):
                InfoLoss.append(self.infols(feature[index], targets, stride[index]))
        else:
            InfoLoss = [0,0 ,0]
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            # print(pi.shape)
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                if i==0:
                    stride = 8
                elif i==1:
                    stride = 16
                else:
                    stride = 32
                W = anchors[i][:, 0] * stride
                H = anchors[i][:, 1] * stride
                origin_j = gj*stride + stride/2
                origin_i = gi*stride + stride/2
                stride_list = (torch.ones(len(origin_j))*stride).cuda()
                sense_of_filed = torch.stack((origin_i,origin_j,stride_list,stride_list),dim=1)

                # print("YGYG")
                # print(anchors[i])
                rec = torch.stack((origin_i,origin_j,W,H),dim=1)
                sample.append(rec)
                result.append(pbox)

                tbb = tbox[i].clone()
                tbb[:, 0] += gi
                tbb[:, 1] += gj
                tbb *= stride
                # tbb[:, 1] *= stride

                # print("YGYGYG",sense_of_filed)
                # print("YGYG",tbb)

                # great_weight = useful_ratio(tbb, sense_of_filed).squeeze()
                # print(great_weight)
                # great_weight = 1.0/(1+torch.exp(-10*(great_weight-0.5)))/2+0.5

                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False ,CIoU=True).squeeze()  # iou(prediction, target)

                # lbox += ((1.0 - iou)*great_weight).mean()  # iou loss
                lbox += ((1.0 - iou)).mean()
                # print("IOU",iou)
                # print("great",great_weight)
                # print("final:",(1.0-iou)*great_weight)

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    # lcls += self.BCEcls(pcls, t)
                    # hyperparameters
                    # print(pcls.shape)
                    # print(t.shape)
                    # print(great_weight.shape)
                    # lcls+=F.binary_cross_entropy_with_logits(pcls, t,weight=great_weight.unsqueeze(1),pos_weight=torch.tensor([self.h['cls_pw']]).cuda())
                    lcls += self.BCEcls(pcls, t)
                    # lcls+=equalized_focal_loss(pcls,t)
                    # BCE
                    # lcls += self.BCEcls(pcls, t)
                # Append targets to text file/
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lcontrals = (InfoLoss[0] + InfoLoss[1] + InfoLoss[2])
        bs = tobj.shape[0]  # batch size
        if(lcontrals != 0):
            lcontrals = ((InfoLoss[0] + InfoLoss[1]+ InfoLoss[2]) / 64).unsqueeze(0)
            return (lbox + lobj + lcls) * bs,  torch.cat((lbox, lobj, lcls)).detach()
        else:
            return (lbox + lobj + lcls) * bs,  torch.cat((lbox, lobj, lcls)).detach()
        # print("LOSS",lbox* bs,lobj* bs,lcls* bs,lcontrals)
        # print(sample)

        # return (lbox + lobj + lcls) * bs+lcontrals, torch.cat((lbox, lobj, lcls,lcontrals)).detach()

    def build_targets(self, p, targets):
        '''
        zhaochu yu gtbox zuipipei de xianyankuang
        '''
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        ''' lei bie  bianjiehe suoyin maokuang  '''
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        '''ai.shape = (na,nt)'''
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        '''targets.shape = (na,nt,7) gei meige mubiao jia suoyin'''
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        '''
        nl daibiao jiancetou de shuliang FPN sanceng 
        '''
        # print("build_targets begin")
        for i in range(self.nl):
            # anchors = self.anchors[i] #sange zhiyou chang kuan de xianyankuang
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            # indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class


        return tcls, tbox, indices, anch

