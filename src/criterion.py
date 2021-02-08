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
    def __init__(self, output_key="logit", gamma=2.0, alpha=1.0, posi_weight=1.0, zero_weight=0.2, zero_smoothing_label=0.2):
        super().__init__()
        self.posi_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.nega_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.zero_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.output_key = output_key
        self.gamma = gamma
        self.alpha = alpha
        self.posi_weight = posi_weight
        self.zero_weight = zero_weight
        self.zero_smoothing_label = zero_smoothing_label

    def forward(self, inputs, target, phase='train'):
        input = inputs[self.output_key]
        target = target.float()
    
        posi_mask = (target == 1).float()
        nega_mask = (target == -1).float()  # (20, 24)
        zero_mask = (target == 0).float()  # 明確にラベルがついていないちょっとあやふやな箇所
        
        # validの場合view, maxで分割したデータを１つのデータとして集約する必要がある
        if phase == 'valid':
            input = C.split2one(input, target)
        
        posi_y = torch.ones(input.shape).to('cuda')
        nega_y = torch.zeros(input.shape).to('cuda')  # dummy
        zero_y = torch.full(input.shape, self.zero_smoothing_label).to('cuda')   # zero labelにsmoothingをかける
        # zero_y = torch.zeros(input.shape).to('cuda')  # dummy

        posi_loss = self.posi_loss(input, posi_y)
        nega_loss = self.nega_loss(input, nega_y)  # 全て負例と見做してloss計算
        zero_loss = self.zero_loss(input, zero_y)

        probas = input.sigmoid()
        focal_pw = (1. - probas)**self.gamma
        posi_loss = (posi_loss * posi_mask * focal_pw).sum()
        nega_loss = (nega_loss * nega_mask).sum()  # ラベルのついているクラスのみlossを残す
        zero_loss = (zero_loss * zero_mask).sum()
        # pos_w = 0 if posi_mask.sum() == 0 else 1/posi_mask.sum()
        # nega_w = 0 if nega_mask.sum() == 0 else 1/nega_mask.sum()
        # zero_w = 0 if zero_mask.sum() == 0 else 1/zero_mask.sum()
        return posi_loss*self.posi_weight, nega_loss, zero_loss*self.zero_weight

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


