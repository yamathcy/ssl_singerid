import numpy as np
import pytorch_lightning as pl


"""
train.py 
モデルの学習に関する処理をここに書く
"""

def train(model, train_loader, valid_loader, logger, max_epochs=100):
    trainer = pl.Trainer(max_epochs=max_epochs, deterministic=True, check_val_every_n_epoch=10, logger=logger)
    trainer.fit(model, train_loader, valid_loader)
    return model, trainer
