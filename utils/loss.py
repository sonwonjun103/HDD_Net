import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        "predict & target batch size don't match"
        assert predict.shape[0] == target.shape[0]
        
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        return loss.mean()

    
# BCEDiceLoss
class BCEDiceLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss() 
        self.beta = beta
        self.dice = DiceLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, target):
        bceloss = self.alpha * self.bce(input, target)
        diceloss = self.beta * self.dice(input, target)
        return bceloss, diceloss