
import random
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
# from utils import apply_sigmoid


class DiceScoreStorer(object):
    """
    store dice score of each patch, 
    seperate pos and neg patches,
    """
    def __init__(self, sigmoid=False, thresh=0.5, eps=1e-6):
        # self.pos = AverageMeter(cache=False)
        # self.neg = AverageMeter(cache=False)
        self.array = []
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.eps = eps
        self.sigmoid = sigmoid
        self.thresh = thresh
    def __len__(self):
        return len(self.array)

    def update(self, pred_mask, gt_mask):
        N = pred_mask.size(0)
        # is_pos = gt_mask.view(N,-1).sum(1).type(torch.bool)
        dice_scores = self._dice_score(pred_mask, gt_mask)
        dice_scores = list(dice_scores.detach().cpu().numpy())
        self.array = self.array + dice_scores
        self.count += N
        self.sum += sum(dice_scores)
        self.avg = self.sum/self.count

    @property
    def val(self):
        return self.array[-1]

    def _dice_score(self, preds, gt):
        if self.sigmoid:
            preds = (torch.sigmoid(preds) > self.thresh).type(gt.type())
        else:
            preds = torch.softmax(preds, dim=1)
            preds = preds.max(axis=1)[1].unsqueeze(1)
        
        preds = preds.view(preds.size(0), -1)
        gt = gt.view(gt.size(0), -1)
        # try:
        intersect = gt*preds
        # except:
            # print("####Check sigmoid or softmax mode in config####")

        return (2.0*intersect.sum(1) + self.eps)/(preds.sum(1)+gt.sum(1) + self.eps)
    

class IoUStorer(object):
    """
    store dice score of each patch, 
    seperate pos and neg patches,
    """
    def __init__(self, sigmoid=False, thresh=0.5, eps=1e-6):
        # self.pos = AverageMeter(cache=False)
        # self.neg = AverageMeter(cache=False)
        self.array = []
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.sigmoid = sigmoid
        self.thresh = thresh
        self.eps = eps
        
    def __len__(self):
        return len(self.array)

    def update(self, pred_mask, gt_mask):
        N = pred_mask.size(0)
        # is_pos = gt_mask.view(N,-1).sum(1).type(torch.bool)
        iou = self._iou(pred_mask, gt_mask)
        iou = list(iou.detach().cpu().numpy())
        self.array = self.array + iou
        self.count += N
        self.sum += sum(iou)
        self.avg = self.sum/self.count

    @property
    def val(self):
        return self.array[-1]

    def _iou(self, preds, gt):
        if self.sigmoid:
            preds = (torch.sigmoid(preds) > self.thresh).type(gt.type())
            
        else:
            preds = torch.softmax(preds, dim=1)
            preds = preds.max(axis=1)[1].unsqueeze(1)
            # print(preds)
        # preds = preds.max(axis=1)[1]
        preds = preds.view(preds.size(0), -1)
        gt = gt.view(gt.size(0), -1)
        # print("@@@@@@@:", preds.size(), gt.size())
        # try:
        intersect = gt*preds
        # except:
        #     print("####Check sigmoid or softmax mode in config####")
        # print(intersect)
        union = ((gt + preds) > 0).type(intersect.type())
        return (intersect.sum(1) + self.eps)/(union.sum(1) + self.eps)

if __name__ == "__main__":
    
    out = torch.rand((5,2,80,80,80), dtype=torch.float)
    
    # F.sigmoid(out)
    target = torch.zeros((5,80,80,80), dtype=torch.bool)
    # print(out)
    top_iou = IoUStorer(sigmoid=False, thresh=0.5)
    top_dice = DiceScoreStorer(sigmoid=False, thresh=0.5)


    print(top_iou._iou(out, target).size())
    print(top_dice._dice_score(out, target).size())