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
# class Artist(torch.utils.data.Dataset):
#     def __init__(self, dir,sr=44100, chunk_length=5, set=[1,2,3,4,5,6]):
#         self.sr = sr
#         self.data = []
#         self.labels = []
#
#         # クラス名とIDを相互参照するためのdict
#         self.class_to_id = {}
#         self.id_to_class = {}
#
#         # クラスカテゴリ名のIDの割り振り
#         p = Path(dir)
#         self.singers = [entry.name for entry in p.iterdir() if entry.is_dir()]
#
#         for i, category in enumerate(self.singers):
#             self.class_to_id[category] = i
#             self.id_to_class[i] = category
#
#         # データとラベルの読み込み
#         for singer in tqdm(self.singers):
#             print("load singer: {}".format(singer))
#             singer_path = os.path.join(dir, singer)
#             p = Path(singer_path)
#             albums = sorted([entry.name for entry in p.iterdir() if entry.is_dir()])
#             singer_label = self.class_to_id[singer]
#             for num, album in enumerate(albums):
#                 if num not in set:
#                     # アルバムスプリット，目的のセットでないならパス
#                     print("{} is not in set {}, skipped".format(num,set))
#                     continue
#                 else:
#                     audio_list = sorted(glob.glob(os.path.join(dir, singer, album, "*vocal.wav")))
#                     for file_path in audio_list:
#                         # データ，ラベルの読み込み
#                         # audio, sr = librosa.load(file_path, sr=self.sr)
#                         audio, sr = torchaudio.load(file_path)
#                         audio = derive_desired_wav(audio, sr, self.sr)
#                         # print(audio.shape)
#
#                         # チャンク（無音だけのファイルを除去）してデータにappend
#                         trimmed = chunk_audio(audio,chunk_length,sr,rms_filter=True)
#                         label_for_trimmed = [singer_label for x in range(len(trimmed))]
#                         self.data.extend(trimmed)
#                         self.labels.extend(label_for_trimmed)
#
#                     # 解放
#                     del audio
#                     del trimmed
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx], self.labels[idx]
#     def get_class_to_id(self):
#         return self.class_to_id

class Artist(torch.utils.data.Dataset):
    def __init__(self, dir, sr=44100, chunk_length=5, set=[0,1,2,3,4,5,6], transforms=None):
        self.sr = sr
        self.data = []
        self.labels = []
        # クラス名とIDを相互参照するためのdict
        self.class_to_id = {}
        self.id_to_class = {}

        for fold in set:
            folddata = sorted(glob.glob(os.path.join(dir, "*-{}-*-*.wav".format(fold))))
            # print('fold data: {}'.format(folddata))
            self.data.extend(folddata)
        for data in self.data:
            singer = os.path.basename(data.split("-")[0])
            if not singer in self.class_to_id.keys():
                self.class_to_id[singer] = int(len(self.class_to_id))
                self.id_to_class[len(self.class_to_id)] = singer
            lab = self.class_to_id[singer]
            self.labels.append(lab)

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # data = np.load(self.data[idx])
        # data = torch.from_numpy(data).clone()
        data,_ = torchaudio.load(self.data[idx])
        # data = data.squeeze(dim=1)
        # data = torch.rand((1,80000))
        label = self.labels[idx]
        if self.transforms:
            data = self.transforms(data)
        return data,label

    def get_class_to_id(self):
        return self.class_to_id
    
class Artist_from_numpy(torch.utils.data.Dataset):
    def __init__(self, dir, sr=44100, chunk_length=5, set=[0,1,2,3,4,5,6], transforms=None):
        self.sr = sr
        self.data = []
        self.labels = []
        # クラス名とIDを相互参照するためのdict
        self.class_to_id = {}
        self.id_to_class = {}
        print("load audio from", dir)
        for fold in set:
            folddata = sorted(glob.glob(os.path.join(dir, "*-{}-*-*.npy".format(fold))))
            print("data sample:{}".format(len(folddata)))
            # print('fold data: {}'.format(folddata))
            self.data.extend(folddata)
        for data in self.data:
            singer = os.path.basename(data.split("-")[0])
            if not singer in self.class_to_id.keys():
                self.class_to_id[singer] = int(len(self.class_to_id))
                self.id_to_class[len(self.class_to_id)] = singer
            lab = self.class_to_id[singer]
            self.labels.append(lab)

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = np.load(self.data[idx])
        data = torch.from_numpy(data).clone()
        # data,_ = torchaudio.load(self.data[idx])
        # data = data.squeeze(dim=1)
        # data = torch.rand((1,80000))
        label = self.labels[idx]
        if self.transforms:
            data = self.transforms(data)
        return data,label

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
    y = y.squeeze()
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

def chunk_audio(audio:torch.Tensor, chunk_length:int, sr, rms_filter=False):
    # Calculate number of samples per chunk
    samples_per_chunk = int(chunk_length * sr)
    # Chunk the audio
    audio =torch.squeeze(audio)
    audio_chunks = [audio[i:i + samples_per_chunk] for i in range(0, len(audio), samples_per_chunk)]
    # print("{} chunks".format(len(audio_chunks)))
    audio_chunks.pop(-1)
    # print(audio_chunks[-1].shape)
    if rms_filter:
        audio_chunks = [torch.unsqueeze(item,dim=0) for item in audio_chunks if rms_filtering(item)]
    # print("{}  trimed chunks".format(len(audio_chunks)))
    return audio_chunks