import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchaudio
import torchmetrics
from torchvision.models.feature_extraction import create_feature_extractor

"""
eval.py
学習に用いるモデルについてを書く
"""


def evaluation(model, trainer, test_loader):
    """
    モデルとtrainer, test用のdataloaderを用いてテストデータでの評価を行う
    :param model:
    :param trainer:
    :param test_loader:
    :return:
    """
    trainer.test(model, test_loader)


def single_test(model:torch.nn.Module, data):
    """
    1つのデータをモデルに入力して推定結果を得る
    :param model:
    :param data:
    :return:
    """
    model.eval()
    with torch.no_grad():
        output = model(data)
        # todo: outputをtensorではなくidにする？
    return output

