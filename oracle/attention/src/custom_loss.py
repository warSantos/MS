import torch
from torch import nn

class DiffMatrixLoss(nn.Module):

    def __init__(self):
        super(DiffMatrixLoss, self).__init__()

    def forward(self, output, target):

        return torch.sum(torch.abs(output - target))


class AttnLogLoss(nn.Module):

    def __init__(self):
        super(AttnLogLoss, self).__init__()

    def forward(self, output, target):

        inner_mult = torch.mul(output, target)
        diff = - torch.log(0.00001 + torch.sum(inner_mult, dim=2))
        sum_target = torch.sum(target, dim=2)

        return torch.sum(1 / target.shape[0] * (torch.mul(diff, sum_target)))

class ExpandedLoss(nn.Module):

    def __init__(self):
        super(ExpandedLoss, self).__init__()
    
    def forward(self, output, target):

        # Comparing the attention weights with the matrix of good combinations.
        diff = torch.abs(target - output)
        # Lines with only zeros are hits.
        contrast = diff == torch.zeros(target[0][0].shape[1])
        # For document verify if is there any with only zeros.
        result = torch.all(contrast, dim=2)
        # Getting the documents with no zero lines.
        result = ~torch.any(result, dim=1)
        # Return the number of documents that the attention weigts
        # found no equivalent combination.
        return torch.sum(result)