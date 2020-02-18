import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import cv2
from volumentations import *
from config import get_cfg_defaults
import warnings
warnings.filterwarnings("ignore")


class LNDbDS(Dataset):
    def __init__(self, cfg, csv, mode):
        super(LNDbDS, self).__init__()
        self.df = pd.read_csv(csv) 
        self.cfg = cfg
        self.mode = mode        
        if self.mode == 'train':
            self.aug = self.get_augmentation()

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        """
        """
        datum = self.df.iloc[idx] # (LNDbID, start_slice_index)
        # print(datum)
        
        if self.mode == 'test':
            # get scan
            scan = self._get_test_cube(
                os.path.join(self.cfg.DIRS.DATA, "scan_cubes"),
                datum["LNDbID"], 
                datum["FindingID"]
            )

            self._apply_window(scan, self.cfg.DATA.WINDOW_CENTER, self.cfg.DATA.WINDOW_WIDTH)

            #stack
            scan = np.stack([scan]*self.cfg.DATA.INP_CHANNELS, axis=0)
            scan = torch.tensor(scan, dtype=torch.float)

            return datum["LNDbID"], datum["FindingID"], scan
        else:
            # get scan
            scan = self._get_cube(
                os.path.join(self.cfg.DIRS.DATA, "scan_cubes2"),
                datum["LNDbID"], 
                datum["RadID"],
                datum["FindingID"]
            )
            
            self._apply_window(scan, self.cfg.DATA.WINDOW_CENTER, self.cfg.DATA.WINDOW_WIDTH)
            
            # get mask
            mask = self._get_cube(
                os.path.join(self.cfg.DIRS.DATA, "mask_cubes2"),
                datum["LNDbID"], 
                datum["RadID"],
                datum["FindingID"]
            )
            #data augmentation
            data = {'image': scan, 'mask': mask}
            if self.mode == "train" and self.cfg.TRAIN.AUGMENTATION:
                aug_data = self.aug(**data)
                scan, mask = aug_data['image'], aug_data['mask']
            
            #stack
            scan = np.stack([scan]*self.cfg.DATA.INP_CHANNELS, axis=0)
            scan = torch.tensor(scan, dtype=torch.float)
            mask = torch.tensor(mask, dtype=torch.long)


            return datum["LNDbID"], datum["FindingID"], scan, mask

        
        
        
        # mask = torch.tensor(mask, dtype=torch.long)
        # print("scan shape ###############", scan.size())
        # print("mask shape ###############", mask.size())
        
        # print("LNDbID",datum["LNDbID"])
        # print("FindingID",datum["FindingID"])
        
            

    def _get_test_cube(self, folder, study_id, finding_id):
        file_format = os.path.join(
            folder, 
            "LNDb-{:04d}_finding{}.npy",
        )
        cube = np.load(file_format.format(study_id, finding_id))

        return cube


    def _get_cube(self, folder, study_id, rad_id, finding_id):
        """
        return torch.halftensor of shape (Z,C,H,W) where Z is number of slice (len(slice_list)
        """
        file_format = os.path.join(
            folder, 
            "LNDb-{:04d}_finding{}_rad{}.npy",
        )
        cube = np.load(file_format.format(study_id, finding_id, rad_id))
        # print("@@@@@", len(os.listdir(folder )))
        return cube
    
    def _apply_window(self, x, center, width):
        """
        clip and tranform to [0,1] value range
        """
        upper = center + width//2
        lower = center - width//2
        
        x = np.clip(x, lower, upper)
        return (x-lower)/width

    def get_augmentation(self):
        return Compose([
            # Resize(patch_size, always_apply=True),
            #CropNonEmptyMaskIfExists(patch_size, always_apply=True),
            # Normalize(always_apply=True),
            # ElasticTransform((0, 0.25)),
            # Rotate((-15,15),(-15,15),(-15,15)),
            # Flip(0),
            # Flip(1),
            # Flip(2),
            #Transpose((1,0,2)), # need patch.height = patch.width
            # RandomRotate90((0,1)),
            # RandomGamma(),
            GaussianNoise(p=0.2),
        ])

def get_test(mode, cfg):
    csv = os.path.join(cfg.TEST.CSV,f"test_fold{cfg.TEST.FOLD}.csv")
    dts = LNDbDS(cfg, csv, mode='test')
    batch_size = cfg.TEST.BATCH_SIZE
    dataloader = DataLoader(dts, batch_size=batch_size, 
                                shuffle=False, drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS)
    return dataloader

def get_dataset(mode, cfg):
    
    if mode == 'train':
        csv = os.path.join(cfg.TRAIN.CSV,f"train_fold{cfg.TRAIN.FOLD}.csv")
        dts = LNDbDS(cfg, csv, mode)
        batch_size = cfg.TRAIN.BATCH_SIZE
        dataloader = DataLoader(dts, batch_size=batch_size, 
                                shuffle=True, drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS)
    else:
        csv = os.path.join(cfg.VAL.CSV,f"val_fold{cfg.VAL.FOLD}.csv")
        dts = LNDbDS(cfg, csv, mode)
        batch_size = cfg.VAL.BATCH_SIZE
        dataloader = DataLoader(dts, batch_size=batch_size, 
                                shuffle=False, drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS)
    return dataloader

def get_debug_dataset(mode, cfg):
    # cfg = get_cfg_defaults()
    if mode == 'train':
        csv = os.path.join(cfg.TRAIN.CSV,f"train_fold{cfg.TRAIN.FOLD}.csv")
        dts = LNDbDS(cfg, csv, mode)
        dts = Subset(dts, np.random.choice(np.arange(len(dts)), 5))
        # dts = Subset(dts)
        batch_size = cfg.TRAIN.BATCH_SIZE
        dataloader = DataLoader(dts, batch_size=batch_size, 
                                shuffle=True, drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS)
    else:
        csv = os.path.join(cfg.VAL.CSV,f"val_fold{cfg.VAL.FOLD}.csv")
        dts = LNDbDS(cfg, csv, mode)
        dts = Subset(dts, np.random.choice(np.arange(len(dts)), 2))
        batch_size = cfg.VAL.BATCH_SIZE
        dataloader = DataLoader(dts, batch_size=batch_size, 
                                shuffle=False, drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS)
    return dataloader

if __name__ == "__main__":
    cfg = get_cfg_defaults()
    # # dts = get_dataset('train', cfg)
    # csv = os.path.join(cfg.DIRS.DATA_CSV, "train_fold/train_fold0.csv")
    # dts = LNDbDS(cfg, csv, 'train')
    # # print(type(dts))   
    # id, sc, mask = dts.__getitem__(1) 
    # print(sc.shape)
    # dts = get_debug_dataset('train', cfg)
    scan = np.random.rand(80,80,80)
    mask = np.random.rand(80,80,80)
    # print(scan.shape)
    def get_augmentation():
        return Compose([
            # Resize(patch_size, always_apply=True),
            #CropNonEmptyMaskIfExists(patch_size, always_apply=True),
            # Normalize(always_apply=True),
            # ElasticTransform((0, 0.25)),
            # Rotate((-15,15),(-15,15),(-15,15)),
            #Flip(0),
            #Flip(1),
            #Flip(2),
            #Transpose((1,0,2)), # need patch.height = patch.width
            #RandomRotate90((0,1)),
            # RandomGamma(),
            GaussianNoise(p=0.1),
        ])

    aug = get_augmentation()

    data = {'image': scan, 'mask': mask}
    aug_data = aug(**data)
    scan, mask = aug_data['image'], aug_data['mask']
    print(scan)

