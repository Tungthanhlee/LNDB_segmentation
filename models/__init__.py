
import apex
from apex import amp
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pandas as pd
from pytorch_toolbelt.inference import tta as pytta
import os

# from .resnet import ResNet
from .vnet_1 import VNet
from .unet3d import unet_3D
from .unet_CT_multi_att_dsv_3D import unet_CT_multi_att_dsv_3D
from .unet_CT_single_att_dsv_3D import unet_CT_single_att_dsv_3D
from .unet_CT_dsv_3D import unet_CT_dsv_3D
from .unet_grid_attention_3D import unet_grid_attention_3D
from tensorboardX import SummaryWriter

from .utils import AverageMeter, DICE, IOU, apply_sigmoid, apply_softmax, save_checkpoint
from .dice_loss_ import *
from .metrics import DiceScoreStorer, IoUStorer
import warnings
warnings.filterwarnings("ignore")



def get_model(cfg):
    
    if cfg.TRAIN.MODEL in 'vnet':
        model = VNet()
    elif cfg.TRAIN.MODEL in 'unet_3D':
        model = unet_3D(n_classes=cfg.DATA.SEG_CLASSES)
    elif cfg.TRAIN.MODEL in 'unet_CT_multi_att_dsv_3D':
        model = unet_CT_multi_att_dsv_3D(n_classes=cfg.DATA.SEG_CLASSES)
    elif cfg.TRAIN.MODEL in 'unet_CT_single_att_dsv_3D':
        model = unet_CT_single_att_dsv_3D(n_classes=cfg.DATA.SEG_CLASSES)
    elif cfg.TRAIN.MODEL in 'unet_CT_dsv_3D':
        model = unet_CT_dsv_3D(n_classes=cfg.DATA.SEG_CLASSES) 
    elif cfg.TRAIN.MODEL in 'unet_grid_attention_3D':
        model = unet_grid_attention_3D(n_classes=cfg.DATA.SEG_CLASSES) 
    elif cfg.TRAIN.MODEL in 'unet_nonlocal_3D':
        model = unet_grid_attention_3D(n_classes=cfg.DATA.SEG_CLASSES) 
    else:
        print("Model not found")

    return model

def test_model(_print, cfg, model, test_loader, weight="", tta=False):

    if tta:
        model = pytta.TTAWrapper(model, pytta.fliplr_image2label)
    # print("@@@@@")
    # model = model()
    # print(model)
    # model = nn.DataParallel(model)
    model.load_state_dict(torch.load(weight)["state_dict"])
    

    model.eval()
    tbar = tqdm(test_loader)
    # if os.path.exists()
    with torch.no_grad():
        for batch in tbar:
            _id, finding, image = batch
            # print("#####",_id)
            # move data to cuda
            image = image.cuda()
            
            output = model(image)

            output = torch.softmax(output, dim=1)
            output = output.max(axis=1)[1]
            # print("@@@@out shape: ", output.size())
            output = output.type(torch.bool).cpu().numpy()
            for i in range(output.shape[0]):
                np.save(os.path.join(cfg.DIRS.TEST, 'LNDb-{:04d}_finding{}.npy'.format(_id[i].item(), finding[i].item())), output[i])
                # print("Exported to numpy")
    
def valid_model(_print, cfg, model, valid_criterion, valid_loader, tta=False):
    losses = AverageMeter()
    top_iou = AverageMeter()
    top_dice = AverageMeter()
    # top_iou = IoUStorer()
    top_iou = IoUStorer(sigmoid=cfg.METRIC.SIGMOID, thresh=cfg.METRIC.THRESHOLD)
    top_dice = DiceScoreStorer(sigmoid=cfg.METRIC.SIGMOID, thresh=cfg.METRIC.THRESHOLD)

    if tta:
        model = pytta.TTAWrapper(model, pytta.fliplr_image2label)

    model.eval()
    tbar = tqdm(valid_loader)

    with torch.no_grad():
        # for i, (image, target) in enumerate(tbar):
        for i, batch in enumerate(tbar):
            _id, finding, image, target = batch
            #move data to cuda
            image = image.cuda()
            target = target.cuda()
            output = model(image)
            
            #loss
            loss = valid_criterion(output, target)
            
            #metric
            # dice_score = DICE(output, target)
            # iou_score = IOU(output, target)

            
            #record metrics
            top_dice.update(output, target)
            top_iou.update(output, target)

            #record
            losses.update(loss.item(), image.size(0))
            # top_iou.update(iou_score, image.size(0))
            # top_dice.update(dice_score, image.size(0))          

    _print("Valid iou: %.3f, dice: %.3f loss: %.3f" % (top_iou.avg, top_dice.avg, losses.avg))
    
    # return top_dice.avg.data.cpu().numpy(), top_iou.avg.data.cpu().numpy()
    return top_dice.avg, top_iou.avg
    


def train_loop(_print, cfg, model, train_loader, valid_loader, criterion, valid_criterion, optimizer, scheduler, start_epoch, best_metric):
    if cfg.DEBUG == False:
        tb = SummaryWriter(f"runs/{cfg.EXP}/{cfg.TRAIN.MODEL}", comment=f"{cfg.COMMENT}") #for visualization
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        _print(f"Epoch {epoch + 1}")
        
        #define some meters
        losses = AverageMeter()
        # top_iou = AverageMeter()
        # top_dice = AverageMeter()

        top_iou = IoUStorer(sigmoid=cfg.METRIC.SIGMOID, thresh=cfg.METRIC.THRESHOLD)
        top_dice = DiceScoreStorer(sigmoid=cfg.METRIC.SIGMOID, thresh=cfg.METRIC.THRESHOLD)

        """
        TRAINING
        """
        #switch model to training mode
        model.train()
        
        tbar = tqdm(train_loader)

        for i, batch in enumerate(tbar):
            _id, finding, image, target = batch
            #move data to cuda
            image = image.cuda()
            target = target.cuda()

            
            #data through model
            output = model(image)

            
            # if cfg.METRIC.SIGMOID:
            #     output = apply_sigmoid(output)
            # else:
            #     output = apply_softmax(output)
            #calculate loss
            #if using BCEloss, squeeze output to have same shape with target, BxHxWxD
            loss = criterion(output, target)
            # print("out:#####", output.size())
            # print("target:####", target.size())
            #metrics
            # dice_score = DICE(output, target)
            # iou_score = IOU(output, target)
            #record dice score and iou
            top_dice.update(output, target)
            top_iou.update(output, target)


            # gradient accumulation
            loss = loss / cfg.OPT.GD_STEPS
            
            if cfg.SYSTEM.FP16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (i + 1) % cfg.OPT.GD_STEPS == 0:
                scheduler(optimizer, i, epoch, None) # Cosine LR Scheduler
                optimizer.step()
                optimizer.zero_grad()

            # record loss
            losses.update(loss.item() * cfg.OPT.GD_STEPS, image.size(0))
            # top_iou.update(iou_score, image.size(0))
            # top_dice.update(dice_score, image.size(0))


            tbar.set_description("Train iou: %.3f, dice: %.3f loss: %.3f, learning rate: %.6f" % (top_iou.avg, top_dice.avg, losses.avg, optimizer.param_groups[-1]['lr']))
            if cfg.DEBUG == False:
                #tensorboard
                tb.add_scalars('Loss', {'loss':losses.avg}, epoch)
                # tb.add_scalars('Dice_score', {'top_dice':top_dice.avg}, epoch)
                # tb.add_scalars('Iou_score', {'top_iou':top_iou.avg}, epoch)
                tb.add_scalars('Train',
                            {'top_dice':top_dice.avg,
                            'top_iou':top_iou.avg}, epoch)
                tb.add_scalars('Lr', {'Lr':optimizer.param_groups[-1]['lr']}, epoch)

            

        _print("Train iou: %.3f, dice: %.3f, loss: %.3f, learning rate: %.6f" % (top_iou.avg, top_dice.avg, losses.avg, optimizer.param_groups[-1]['lr']))

        """
        VALIDATION
        """

        top_dice_valid, top_iou_valid = valid_model(_print, cfg, model, valid_criterion, valid_loader)

        #Take dice_score as main_metric to save checkpoint
        is_best = top_dice_valid > best_metric
        best_metric = max(top_dice_valid, best_metric)
        
        
        
        #tensorboard
        if cfg.DEBUG == False:
            tb.add_scalars('Valid',
                            {'top_dice':top_dice_valid,
                            'top_iou':top_iou_valid}, epoch)
            
            save_checkpoint({
                "epoch": epoch + 1,
                "arch": cfg.EXP,
                "state_dict": model.state_dict(),
                "best_metric": best_metric,
                "optimizer": optimizer.state_dict(),
            }, is_best, root=cfg.DIRS.WEIGHTS, filename=f"{cfg.EXP}_{cfg.TRAIN.MODEL}_fold{cfg.TRAIN.FOLD}.pth")

    if cfg.DEBUG == False:
        # #export stats to json
        tb.export_scalars_to_json(os.path.join(cfg.DIRS.OUTPUTS, f"{cfg.EXP}_{cfg.TRAIN.MODEL}_{cfg.COMMENT}_{round(best_metric,4)}.json"))
        # #close tensorboard
        tb.close()

