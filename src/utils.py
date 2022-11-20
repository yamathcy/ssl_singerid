import os,sys
from sklearn.manifold import TSNE
import torch 
import numpy as np
import matplotlib.pylab as plt
from torch.utils.data import DataLoader, Dataset

"""
utils.py 
その他のちょっとした処理を書く
"""


# dict型のラベル-ID変換表を更新する
def label_to_id(label:str, id_dict:dict):
    id = len(id_dict)
    if label not in id_dict.keys():
        id_dict[label] = id

    return id_dict[label], id_dict
