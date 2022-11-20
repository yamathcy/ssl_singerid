import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchaudio
import torchmetrics

"""
train.py 
モデルの学習に関する処理をここに書く
"""


def train(model, train_loader, valid_loader, logger, gpus=1, max_epochs=100):
    trainer = pl.Trainer(gpus=1, max_epochs=max_epochs, deterministic=True, check_val_every_n_epoch=10,logger=logger)
    trainer.fit(model, train_loader, valid_loader)
    
    return model, trainer

