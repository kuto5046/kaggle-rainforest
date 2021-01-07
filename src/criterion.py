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

    def forward(self, input, target, phase="train"):
        input_ = input[self.output_key]
        framewise_output = input["framewise_output"]  # framewiseからclipwiseを求める
        target = target.float()

        # validの場合view, maxで分割したデータを１つのデータとして集約する必要がある
        if phase == 'valid':
            input_ = C.split2one(input_, target)
            framewise_output = framewise_output.view(target.shape[0], -1, framewise_output.shape[1], target.shape[1])  # [1200, 400, 24] -> [20, 6, 400, 24]
            framewise_output = torch.max(framewise_output, dim=1)[0]  # [20, 6, 400, 24] -> [20, 400, 24]
             
        clipwise_output_with_max, _ = framewise_output.max(dim=1)
        normal_loss = self.normal_loss(input_, target)
        auxiliary_loss = self.bce(clipwise_output_with_max, target)

        return self.weights[0] * normal_loss + self.weights[1] * auxiliary_loss


# refered following repo
# https://github.com/ex4sperans/freesound-classification/blob/71b9920ce0ae376aa7f1a3a2943f0f92f4820813/networks/losses.py
class LSEPLoss(nn.Module):
    def __init__(self, average=True):
        self.average = average
    
    def forward(self, input, target, phase='train'):
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


class LSEPStableLoss(nn.Module):
    def __init__(self, average=True):
        self.average = average
    
    def forward(self, input, target, phase="train"):
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

