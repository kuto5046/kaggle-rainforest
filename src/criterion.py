import torch 
import torch.nn as nn
import src.configuration as C


class ImprovedPANNsLoss(nn.Module):
    def __init__(self, output_key="logit", weights=[1, 1]):
        super().__init__()

        self.output_key = output_key
        if output_key == "logit":
            self.normal_loss = nn.BCEWithLogitsLoss()
        else:
            self.normal_loss = nn.BCELoss()

        self.bce = nn.BCELoss()
        self.weights = weights

    def forward(self, inputs, target, phase="train"):
        input = inputs[self.output_key]
        framewise_output = inputs["framewise_output"]  # framewiseからclipwiseを求める
        target = target.float()

        # validの場合view, maxで分割したデータを１つのデータとして集約する必要がある
        if phase == 'valid':
            input = C.split2one(input, target)
            framewise_output = framewise_output.view(target.shape[0], -1, framewise_output.shape[1], target.shape[1])  # [1200, 400, 24] -> [20, 6, 400, 24]
            framewise_output = torch.max(framewise_output, dim=1)[0]  # [20, 6, 400, 24] -> [20, 400, 24]
             
        clipwise_output_with_max, _ = framewise_output.max(dim=1)
        normal_loss = self.normal_loss(input, target)
        auxiliary_loss = self.bce(clipwise_output_with_max, target)

        return self.weights[0] * normal_loss + self.weights[1] * auxiliary_loss


# based https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class FocalLoss(nn.Module):
    def __init__(self, output_key="logit", gamma=2.0, alpha=1.0):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.output_key = output_key
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, target, phase='train'):
        input = inputs[self.output_key]
        target = target.float()
        
        # validの場合view, maxで分割したデータを１つのデータとして集約する必要がある
        if phase == 'valid':
            input = C.split2one(input, target)

        bce_loss = self.loss(input, target)
        probas = torch.sigmoid(input)
        loss = torch.where(target >= 0.5, self.alpha * (1. - probas)**self.gamma * bce_loss, probas**self.gamma * bce_loss)
        loss = loss.mean()
        return loss

# sigmoidを内包しているのでlogitを入力とする
class BCEWithLogitsLoss(nn.Module):
    """
    Loss内部でoutput_keyを適用するためにcustom lossを作成
    
    """
    def __init__(self, output_key="logit"):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.output_key = output_key

    def forward(self, inputs, target, phase="train"):
        input = inputs[self.output_key]
        target = target.float()

        # validの場合view, maxで分割したデータを１つのデータとして集約する必要がある
        if phase == 'valid':
            input = C.split2one(input, target)

        loss = self.loss(input, target)
        return loss


# based https://github.com/ex4sperans/freesound-classification/blob/71b9920ce0ae376aa7f1a3a2943f0f92f4820813/networks/losses.py
class LSEPLoss(nn.Module):
    def __init__(self, output_key='logit', average=True):
        super().__init__()

        self.average = average
        self.output_key = output_key
    
    def forward(self, inputs, target, phase='train'):
        input = inputs[self.output_key]
        target = target.float()
    
        # validの場合view, maxで分割したデータを１つのデータとして集約する必要がある
        if phase == 'valid':
            input = C.split2one(input, target)
    
        differences = input.unsqueeze(1) - input.unsqueeze(2)
        where_different = (target.unsqueeze(1) < target.unsqueeze(2)).float()

        exps = differences.exp() * where_different
        lsep = torch.log(1 + exps.sum(2).sum(1))

        if self.average:
            return lsep.mean()
        else:
            return lsep

# based https://github.com/ex4sperans/freesound-classification/blob/71b9920ce0ae376aa7f1a3a2943f0f92f4820813/networks/losses.py
class LSEPStableLoss(nn.Module):
    def __init__(self, output_key="logit", average=True):
        super(LSEPStableLoss, self).__init__()
        self.average = average
        self.output_key = output_key
    
    def forward(self, inputs, target, phase="train"):
        input = inputs[self.output_key]
        if 'framewise' in self.output_key:
            input, _ = input.max(dim=1)
        target = target.float()
        # validの場合view, maxで分割したデータを１つのデータとして集約する必要がある
        if phase == 'valid':
            input = C.split2one(input, target)

        n = input.size(0)
        differences = input.unsqueeze(1) - input.unsqueeze(2)
        where_lower = (target.unsqueeze(1) < target.unsqueeze(2)).float()

        differences = differences.view(n, -1)
        where_lower = where_lower.view(n, -1)

        max_difference, index = torch.max(differences, dim=1, keepdim=True)
        differences = differences - max_difference
        exps = differences.exp() * where_lower

        lsep = max_difference + torch.log(torch.exp(-max_difference) + exps.sum(-1))

        if self.average:
            return lsep.mean()
        else:
            return lsep

# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Custom loss functions"""

import torch
from torch.nn import functional as F
from torch.autograd import Variable


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes
