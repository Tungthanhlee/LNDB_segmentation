from yacs.config import CfgNode as CN


_C = CN()

_C.EXP = "exp1" # Experiment name
_C.COMMENT = "dice+no_aug+leaky_value" #comment for tensorboard
_C.DEBUG = False

_C.INFER = CN()
_C.INFER.TTA = False

_C.MODEL = CN()
_C.MODEL.DICE_LOSS = False
_C.MODEL.BCE_LOSS = False
_C.MODEL.CE_LOSS = False
_C.MODEL.WEIGHT = ""


_C.SYSTEM = CN()
_C.SYSTEM.SEED = 42
_C.SYSTEM.FP16 = True
_C.SYSTEM.OPT_L = "O0"
_C.SYSTEM.CUDA = True
_C.SYSTEM.MULTI_GPU = False
_C.SYSTEM.NUM_WORKERS = 8

_C.DIRS = CN()
# _C.DIRS.DATA = "/media/tungthanhlee/SSD/grand_challenge/dataset/LNDB_segmentation/split"
# _C.DIRS.DATA_CSV = "/media/tungthanhlee/SSD/grand_challenge/dataset/LNDB_segmentation/trainset_csv/Fold"
_C.DIRS.DATA = "./split"
_C.DIRS.DATA_CSV = "./trainset_csv/Fold"
_C.DIRS.WEIGHTS = "./weights/"
_C.DIRS.OUTPUTS = "./outputs/"
_C.DIRS.LOGS = "./logs/"
_C.DIRS.TEST = "./test"


_C.DATA = CN()
# _C.DATA.AUGMENT_PROB = 0.5
# _C.DATA.MIXUP_PROB = 0.0
# _C.DATA.CUTMIX_PROB = 0.0
_C.DATA.INP_CHANNELS = 1
_C.DATA.SEG_CLASSES = 1
_C.DATA.WINDOW_CENTER = 700
_C.DATA.WINDOW_WIDTH = 2100

_C.OPT = CN()
_C.OPT.OPTIMIZER = "adamw"
_C.OPT.GD_STEPS = 1 
_C.OPT.WARMUP_EPOCHS = 2
_C.OPT.BASE_LR = 1e-3
_C.OPT.WEIGHT_DECAY = 1e-2
_C.OPT.WEIGHT_DECAY_BIAS = 0.0

_C.METRIC = CN()
#if true, the output will be computed by sigmoid, if false -> softmax
_C.METRIC.SIGMOID = True 
_C.METRIC.THRESHOLD = 0.5

_C.TRAIN = CN()
# _C.TRAIN.CSV = "/media/tungthanhlee/SSD/grand_challenge/dataset/LNDB_segmentation/trainset_csv/Fold/train_fold"
_C.TRAIN.CSV = "./trainset_csv/Fold/train_fold"
_C.TRAIN.FOLD = 0
#model name: 'vnet', 'unet_3D', 'unet_CT_multi_att_dsv_3D', 'unet_CT_single_att_dsv_3D', 'unet_CT_dsv_3D', ''unet_grid_attention_3D''
_C.TRAIN.MODEL = "unet_CT_dsv_3D" # Model name
_C.TRAIN.EPOCHS = 50
_C.TRAIN.BATCH_SIZE = 8 #switch to 32 if train on server
_C.TRAIN.DROPOUT = 0.0
_C.TRAIN.AUGMENTATION = False

_C.VAL = CN()
# _C.VAL.CSV = "/media/tungthanhlee/SSD/grand_challenge/dataset/LNDB_segmentation/trainset_csv/Fold/val_fold"
_C.VAL.CSV = "./trainset_csv/Fold/val_fold"
_C.VAL.FOLD = 0
_C.VAL.BATCH_SIZE = 8 #switch to 32 if train on server

_C.CONST = CN()


_C.TEST = CN()
_C.TEST.CSV = "./trainset_csv/Fold/test_fold"
_C.TEST.FOLD = 0
_C.TEST.BATCH_SIZE = 16

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`