import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import wandb
# from torchvision.utils import make_grid
# from itertools import repeat
# import pandas as pd
"""
train.py 
モデルの学習に関する処理をここに書く
"""

# def inf_loop(data_loader):
#     ''' wrapper function for endless data loader. '''
#     for loader in repeat(data_loader):
#         yield from loader


# class MetricTracker:
#     def __init__(self, *keys, writer=None):
#         self.writer = writer
#         self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
#         self.reset()

#     def reset(self):
#         for col in self._data.columns:
#             self._data[col].values[:] = 0

#     def update(self, key, value, n=1):
#         if self.writer is not None:
#             self.writer.add_scalar(key, value)
#         self._data.total[key] += value * n
#         self._data.counts[key] += n
#         self._data.average[key] = self._data.total[key] / self._data.counts[key]

#     def avg(self, key):
#         return self._data.average[key]

#     def result(self):
#         return dict(self._data.average)


# class Trainer(BaseTrainer):
#     """
#     Trainer class
#     """
#     def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
#                  data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
#         super().__init__(model, criterion, metric_ftns, optimizer, config)
#         self.config = config
#         self.device = device
#         self.data_loader = data_loader
#         if len_epoch is None:
#             # epoch-based training
#             self.len_epoch = len(self.data_loader)
#         else:
#             # iteration-based training
#             self.data_loader = inf_loop(data_loader)
#             self.len_epoch = len_epoch
#         self.valid_data_loader = valid_data_loader
#         self.do_validation = self.valid_data_loader is not None
#         self.lr_scheduler = lr_scheduler
#         self.log_step = int(np.sqrt(data_loader.batch_size))

#         self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
#         self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

#     def _train_epoch(self, epoch):
#         """
#         Training logic for an epoch

#         :param epoch: Integer, current training epoch.
#         :return: A log that contains average loss and metric in this epoch.
#         """
#         self.model.train()
#         self.train_metrics.reset()
#         for batch_idx, (data, target) in enumerate(self.data_loader):
#             data, target = data.to(self.device), target.to(self.device)

#             self.optimizer.zero_grad()
#             output = self.model(data)
#             loss = self.criterion(output, target)
#             loss.backward()
#             self.optimizer.step()

#             self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
#             self.train_metrics.update('loss', loss.item())
#             for met in self.metric_ftns:
#                 self.train_metrics.update(met.__name__, met(output, target))

#             if batch_idx % self.log_step == 0:
#                 self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
#                     epoch,
#                     self._progress(batch_idx),
#                     loss.item()))
#                 self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

#             if batch_idx == self.len_epoch:
#                 break
#         log = self.train_metrics.result()

#         if self.do_validation:
#             val_log = self._valid_epoch(epoch)
#             log.update(**{'val_'+k : v for k, v in val_log.items()})

#         if self.lr_scheduler is not None:
#             self.lr_scheduler.step()
#         return log

#     def _valid_epoch(self, epoch):
#         """
#         Validate after training an epoch

#         :param epoch: Integer, current training epoch.
#         :return: A log that contains information about validation
#         """
#         self.model.eval()
#         self.valid_metrics.reset()
#         with torch.no_grad():
#             for batch_idx, (data, target) in enumerate(self.valid_data_loader):
#                 data, target = data.to(self.device), target.to(self.device)

#                 output = self.model(data)
#                 loss = self.criterion(output, target)

#                 self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
#                 self.valid_metrics.update('loss', loss.item())
#                 for met in self.metric_ftns:
#                     self.valid_metrics.update(met.__name__, met(output, target))
#                 self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

#         # add histogram of model parameters to the tensorboard
#         for name, p in self.model.named_parameters():
#             self.writer.add_histogram(name, p, bins='auto')
#         return self.valid_metrics.result()

#     def _progress(self, batch_idx):
#         base = '[{}/{} ({:.0f}%)]'
#         if hasattr(self.data_loader, 'n_samples'):
#             current = batch_idx * self.data_loader.batch_size
#             total = self.data_loader.n_samples
#         else:
#             current = batch_idx
#             total = self.len_epoch
#         return base.format(current, total, 100.0 * current / total)

def train(model, train_loader, valid_loader, logger, conf):
    print("cuda: {}".format(torch.cuda.is_available()))
    train_loss_list = []
    valid_loss_list = []
    max_epochs=conf.epoch
    # device = "cuda"
    # optimizer  = Adam(lr=conf.lr)
    # model.train()
    # criterion = nn.CrossEntropyLoss()
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     data, target = data.to(device), target.to(device)
    #     optimizer.zero_grad()
    #     out, _ = model(data)
    #     loss = criterion(out, target)
    #     loss.backward()
    #     optimizer.step()

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
    trainer.fit(model, train_loader, valid_loader)
    return model, trainer


def train_wo_lightning(model, train_loader, valid_loader, param, logger=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loss_list = []
    valid_loss_list = []
    model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=param.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # loss
    criterion = nn.CrossEntropyLoss()
    last_loss = np.inf
    epoch = param.epoch
    learned_epoch = 0
    for ep in range(epoch):
        # training loop
        print("epoch: {}".format(ep))
        learned_epoch += 1
        before_loss = np.inf
        running_train_loss = 0
        #train
        model.train()
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            output,_ = model(data)
            loss = criterion(targets,output)
            loss.backward()
            running_train_loss += loss.item()
            optimizer.step()
            
        train_loss = running_train_loss / len(train_loader)
        print("train_loss:{}".format(train_loss))
        
        train_loss_list.append(train_loss)

        #valid
        model.eval()
        with torch.no_grad():
            running_valid_loss = 0
            overall_accuracy = 0
            for batch_idx, (data, targets) in enumerate(tqdm(valid_loader)):
                data = data.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                output,_ = model(data)
                loss = criterion(targets,output)
                running_valid_loss += loss.item()
                # label = torch.reshape(label, (param['model']['batch_size'], params['slen'], ))
                loss_eval = criterion(output, targets)
                running_valid_loss += loss_eval.item()
                accuracy = acc_epoch(output, targets)
                

        valid_loss = running_valid_loss / len(valid_loader)
        acc = overall_accuracy/len(valid_loader)
        # logger.log({"epoch":ep ,"train_loss":train_loss,"valid_loss":valid_loss, "valid_acc":acc})

        # earlystop
        if param.early_stop:
            print("patience count: " + str(stop))
            print("best {} , current{}".format(last_loss,valid_loss))
            if valid_loss > last_loss:
                stop += 1
                print("not improved, count:{}".format(stop))
                if stop >= 1:
                    print("early stopping")
                    break
            else: 
                print("valid loss improved")
                last_loss = valid_loss
                stop = 0
        scheduler.step()

    train_loss_list = np.array(train_loss_list)
    valid_loss_list = np.array(valid_loss_list)
    loss_list = (train_loss_list, valid_loss_list)
    # state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(model.state_dict(), str(ep) + '.pth')
    # logger.log_artifact(str(ep) + '.pth')
    # logger.log("actual_epoch", learned_epoch)
    return (model, loss_list)


def acc_epoch(pred, target, thresholds=0.5):
    pred = torch.where(pred>thresholds, 1, 0)
    acc = torch.sum(pred == target) / torch.numel(pred)

    acc = acc.data.cpu().detach().numpy().copy()
    # print('acc: ', acc)
    return acc


    




