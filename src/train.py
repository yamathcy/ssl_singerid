import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch

"""
train.py 
モデルの学習に関する処理をここに書く
"""

def train(model, train_loader, valid_loader, logger, max_epochs=100):
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min")
    trainer = pl.Trainer(max_epochs=max_epochs, check_val_every_n_epoch=10, logger=logger, callbacks=[early_stop_callback],devices=[0])
    print("test check...\n")
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(train_loader):
            x, y = data
            if i > 0:
                break
        test_case = x
        model = model.cuda()
        test_case = test_case.cuda()
        out,_ = model(test_case)
        print(out.shape)
        model.train()
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    return model, trainer
