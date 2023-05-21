import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

"""
train.py 
モデルの学習に関する処理をここに書く
"""

def train(model, train_loader, valid_loader, logger, max_epochs=100):
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min")
    trainer = pl.Trainer(max_epochs=max_epochs, check_val_every_n_epoch=10, logger=logger, callbacks=[early_stop_callback],devices=[0])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    return model, trainer
