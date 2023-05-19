import os, sys
import numpy as np
import librosa
import torch
from tqdm import tqdm
import glob
from pathlib import Path
from src.utils import chunk_audio

"""
dataio.py
ファイルの入出力に関する処理を書く
"""

class Artist(torch.utils.data.Dataset):
    def __init__(self, dir,sr=44100, chunk_length=5, set=[1,2,3,4,5,6]):
        self.sr = sr
        self.data = []
        self.labels = []

        # クラス名とIDを相互参照するためのdict
        self.class_to_id = {}
        self.id_to_class = {}

        # クラスカテゴリ名のIDの割り振り
        p = Path(dir)
        self.singers = [entry.name for entry in p.iterdir() if entry.is_dir()]
        
        for i, category in enumerate(self.singers):
            self.class_to_id[category] = i
            self.id_to_class[i] = category

        # データとラベルの読み込み
        for singer in self.singers:
            print("load singer: {}".format(singer))
            singer_path = os.path.join(dir, singer)
            p = Path(singer_path)
            albums = [entry.name for entry in p.iterdir() if entry.is_dir()]
            singer_label = self.class_to_id[singer]
            for num, album in tqdm(enumerate(albums)):
                if num not in set:
                    # アルバムスプリット，目的のセットでないならパス
                    continue
                else:
                    audio_list = sorted(glob.glob(os.path.join(dir, singer, album, "*vocal.wav")))

                    for file_path in audio_list:
                        # データ，ラベルの読み込み
                        audio, sr = librosa.load(file_path, sr=self.sr)
                        audio = audio[np.newaxis,...]

                        # チャンク（無音だけのファイルを除去）してデータにappend
                        trimmed = chunk_audio(audio,chunk_length,sr,rms_filter=True)
                        for chunk in trimmed:
                            chunk = torch.from_numpy(chunk)
                            self.data.append(chunk)
                            self.labels.append(singer_label)
                            del chunk
                    # 解放
                    del audio
                    del trimmed

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    def get_class_to_id(self):
        return self.class_to_id

    