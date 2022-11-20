import os, sys
import glob
import numpy as np
import librosa
import scipy.signal as signal
import mirdata
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import mirdata
from tqdm import tqdm


from .utils import label_to_id

"""
dataio.py
ファイルの入出力に関する処理を書く
"""

class ESC50(torch.utils.data.Dataset):
    def __init__(self, label_df,base='.',sr=44100):
        self.label_df = label_df
        self.sr = sr
        self.data = []
        self.labels = []
        
        # クラス名とIDを相互参照するためのdict
        self.class_to_id = {}
        self.id_to_class = {}

        # クラスカテゴリ名のIDの割り振り
        self.categories = sorted(label_df["category"].unique())

        for i, category in enumerate(self.categories):
            self.class_to_id[category] = i
            self.id_to_class[i] = category
        
        # データとラベルの読み込み
        for ind in tqdm(range(len(label_df))):
            # 行を取り出す
            row = label_df.iloc[ind]

            # ファイルパス
            file_path = os.path.join(base,row["filename"])

            # データ，ラベルの読み込み
            audio, _ = librosa.load(file_path, sr=self.sr)

            # チャンネル次元がないとエラーになるので，次元を生やして，torch tensorに変換
            audio = audio[np.newaxis,...]
            audio = torch.from_numpy(audio)
            self.data.append(audio)
            self.labels.append(self.class_to_id[row['category']])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]