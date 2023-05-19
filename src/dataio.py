import os, sys
import numpy as np
import librosa
import torch
from tqdm import tqdm
import glob
from pathlib import Path
import torchaudio
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
                        # audio, sr = librosa.load(file_path, sr=self.sr)
                        audio, sr = torchaudio.load(file_path)
                        audio = derive_desired_wav(audio, sr, self.sr)
                        print(audio.shape)

                        # チャンク（無音だけのファイルを除去）してデータにappend
                        trimmed = chunk_audio(audio,chunk_length,sr,rms_filter=True)
                        label_for_trimmed = [singer_label for x in range(len(trimmed))]
                        self.data.extend(trimmed)
                        self.labels.extend(label_for_trimmed)

                    # 解放
                    del audio
                    del trimmed
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    def get_class_to_id(self):
        return self.class_to_id


def derive_desired_wav(audio, old_fs, new_fs):
    if old_fs != new_fs:
        audio_resample = torchaudio.transforms.Resample(orig_freq=old_fs, new_freq=new_fs)(audio)
    else:
        audio_resample = audio
    # mono
    if audio_resample.shape[0] == 2:
        # stero input
        audio_resample = audio_resample.mean(dim=0, keepdim=True)
    return audio_resample


def torch_rms(y,
              frame_length: int = 512,
              hop_length: int = 512,
              pad_mode: str = "reflect",
              center=True):
    tra = torchaudio.transforms.Spectrogram(n_fft=frame_length, hop_length=hop_length, pad_mode=pad_mode, center=center)
    spec = tra(y)
    spec[..., 0, :] *= 0.5
    # Calculate power
    power = 2 * torch.sum(spec, axis=-2, keepdims=True) /frame_length
    rms = torch.sqrt(power.squeeze())
    return rms



def rms_filtering(wav:np.ndarray, th=0.01):
    rms = torch_rms(wav)
    return torch.mean(rms) > th

def chunk_audio(audio:np.ndarray, chunk_length:int, sr, rms_filter=False):
    # Calculate number of samples per chunk
    samples_per_chunk = int(chunk_length * sr)
    # Chunk the audio
    audio_chunks = [audio[...,i:i + samples_per_chunk] for i in range(0, len(audio), samples_per_chunk)]
    if rms_filter:
        audio_chunks = [item for item in audio_chunks if rms_filtering(item)]
    print("{} chunks".format(len(audio_chunks)))
    return audio_chunks