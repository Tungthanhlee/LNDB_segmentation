#dont use
import torch
from torch.autograd import Function
from itertools import repeat
import numpy as np
import torch.nn
# Intersection = dot(A, B)
# Union = dot(A, A) + dot(B, B)
# The Dice loss function is defined as
# 1/2 * intersection / union
#
# The derivative is 2[(union * target - 2 * intersect * input) / union^2]

class DiceLoss(Function):
    def __init__(self, *args, **kwargs):
        super(DiceLoss, self).__init__()
        pass
    
    
    def forward(self, input, target, save=True):
        # print("input: ", input.shape)
        # print("target: ", target.shape)
        # target = target.view(-1)
        # try:
            
        if save:
            self.save_for_backward(input, target)
        eps = 0.000001
        _, result_ = input.max(1)
        result_ = torch.squeeze(result_)
        if input.is_cuda:
            result = torch.cuda.FloatTensor(result_.size())
            self.target_ = torch.cuda.FloatTensor(target.size())
        else:
            result = torch.FloatTensor(result_.size())
            self.target_ = torch.FloatTensor(target.size())
        result.copy_(result_)
        self.target_.copy_(target)
        target = self.target_
#       print(input)
        intersect = torch.sum(result*target)
        # binary values so sum the same as sum of squares
        result_sum = torch.sum(result)
        target_sum = torch.sum(target)
        union = result_sum + target_sum + (2*eps)

        # the target volume can be empty - so we still want to
        # end up with a score of 1 if the result is 0/0
        IoU = intersect / union
        # print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
        #     union, intersect, target_sum, result_sum, 2*IoU))
        out = torch.FloatTensor(1).fill_(2*IoU)
        self.intersect, self.union = intersect, union
        return out
        # except Exception:
        #     print("input: ", input.shape)
        #     print("target: ", target.shape)
        #     exit()

    
    def backward(self, grad_output):
        input, _ = self.saved_tensors
        intersect, union = self.intersect, self.union
        target = self.target_
        gt = torch.div(target, union)
        IoU2 = intersect/(union*union)
        pred = torch.mul(input[:, 1,...], IoU2)
        dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
        # grad_input = torch.stack((torch.mul(dDice, -grad_output[0]),
        #                         torch.mul(dDice, grad_output[0])), 1)
        grad_input = torch.cat((torch.mul(torch.unsqueeze(dDice,1), grad_output[0]),
                                torch.mul(torch.unsqueeze(dDice,1), -grad_output[0])), dim = 1)
        return grad_input , None
    
class BinaryDiceLoss(torch.nn.Module):
    """
        Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            true: a tensor of shape [B, 1, D, H, W].
            logits: a tensor of shape [B, C, D, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            dice_loss: the Sørensen–Dice loss.
    """
    def __init__(self, smooth=1e-6, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        # predict = predict.contiguous().view(predict.shape[0], -1)
        # target = target.contiguous().view(target.shape[0], -1)

        # num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        # den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        # loss = 1 - num / den

        num_classes = predict.shape[1]

        true_1_hot = torch.eye(num_classes)[target.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 4, 1, 2, 3).float()
        
        probas = F.softmax(predict, dim=1)
        true_1_hot = true_1_hot.type(predict.type())
        
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = ((2. * intersection + self.smooth) / (cardinality + self.smooth))
        loss = (1 - dice_loss)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

def binary_dice_loss(pred, target):
    # loss = BinaryDiceLoss()
    # return loss.forward(pred, target)
    return BinaryDiceLoss()(pred, target)

def dice_loss(input, target):
    # print("input: ", input.shape)
    # print("target: ", target.shape)
    # loss = DiceLoss()
    # loss
    return DiceLoss()(input, target)

def dice_error(input, target):
    eps = 0.000001
    _, result_ = input.max(1)
    result_ = torch.squeeze(result_)
    if input.is_cuda:
        result = torch.cuda.FloatTensor(result_.size())
        target_ = torch.cuda.FloatTensor(target.size())
    else:
        result = torch.FloatTensor(result_.size())
        target_ = torch.FloatTensor(target.size())
    result.copy_(result_.data)
    target_.copy_(target.data)
    target = target_
    intersect = torch.sum(result* target)

    result_sum = torch.sum(result)
    target_sum = torch.sum(target)
    union = result_sum + target_sum + 2*eps
    intersect = np.max([eps, intersect])
    # the target volume can be empty - so we still want to
    # end up with a score of 1 if the result is 0/0
    IoU = intersect / union
#    print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
#        union, intersect, target_sum, result_sum, 2*IoU))
    return 2*IoU

# SMOOTH = 1e-6

# def iou(outputs: torch.Tensor, labels: torch.Tensor, SMOOTH = 1e-6):
#     # You can comment out this line if you are passing tensors of equal shape
#     # But if you are passing output from UNet or something it will most probably
#     # be with the BATCH x 1 x H x W shape
#     outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
#     intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
#     union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
#     iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
#     thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
#     return thresholded  # Or thresholded.mean() if you are interested in average across the batch

# def iou_score(output, target, SMOOTH=1e-6):
#     # output = output.squeeze(1)
#     output = output.view(output.size(0), -1)
#     target = target.view(target.size(0), -1)

#     intersect = (output*target).sum(2).sum(1)
#     union = (output+target).sum(2).sum(1)
#     iou = (intersect+SMOOTH)/(union+SMOOTH)

#     return iou.mean()

    

if __name__ == "__main__":
    out = torch.rand((2,2,80,80,80), dtype=torch.float)
    target = torch.zeros((2,80,80,80), dtype=torch.bool)
    # iou = iou_2(out, target)
    # print(iou)
    # loss = binary_dice_loss(out, target)
    # print(loss)
    dice = dice_error(out, target)
    print(dice)
