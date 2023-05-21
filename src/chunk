
import os, sys
import numpy as np
import librosa
import torch
from tqdm import tqdm
import glob
from pathlib import Path
import torchaudio
import argparse

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


def main(args):
    class_to_id = {}
    id_to_class = {}
    p = Path(args.data)
    singers = [entry.name for entry in p.iterdir() if entry.is_dir()]
    audio_list = sorted(glob.glob(os.path.join(args.data,"**/*.wav")))

    for i, category in enumerate(singers):
        class_to_id[category] = i
        id_to_class[i] = category

    # データとラベルの読み込み
    for singer in tqdm(singers):
        print("load singer: {}".format(singer))
        singer_path = os.path.join(dir, singer)
        p = Path(singer_path)
        albums = sorted([entry.name for entry in p.iterdir() if entry.is_dir()])
        singer_label = class_to_id[singer]
        for albumnum, album in enumerate(albums):
            audio_list = sorted(glob.glob(os.path.join(dir, singer, album, "*vocal.wav")))
            for songnum,file_path in enumerate(audio_list):
                # データ，ラベルの読み込み
                # audio, sr = librosa.load(file_path, sr=self.sr)
                audio, sr = torchaudio.load(file_path)
                audio = derive_desired_wav(audio, sr, args.sr)
                # print(audio.shape)

                # チャンク（無音だけのファイルを除去）してデータにappend
                trimmed = chunk_audio(audio,args.length,sr,rms_filter=True)
                label_for_trimmed = [singer_label for x in range(len(trimmed))]
                for trimid, (w,l) in enumerate(zip(trimmed,label_for_trimmed)):
                    savename = "{}-{}-{}-{}.wav".format(singer,albumnum,songnum,trimid)
                    savefile = os.path.join(args.save, savename)
                    torchaudio.save(savefile, w, args.sr)

#     for audiofile in audio_list:
#         p = Path(singer_path)
# #        albums = sorted([entry.name for entry in p.iterdir() if entry.is_dir()])
#         wav, fs = torchaudio.load(audiofile)
#         if fs != args.sr:
#             # resample
#             audio_resample = torchaudio.transforms.Resample(orig_freq=fs, new_freq=args.sr)(wav)
#         else:
#             audio_resample = wav
#         chunk_list = chunk_audio(wav)
#         for chunk in enumerate(chunk_list):
#             savename = 
#             savefile = os.path.join(args.save, )
#             torchaudio.save(savefile, audio_resample, args.sr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/ubuntu/dataset/artist20", help="The path to the dataset")
    parser.add_argument("--save", type=str, default="/home/ubuntu/dataset/artist20/chunk", help="The path to the dataset")
    parser.add_argument("--sr", type=int, default=16000, help="The threshold to split the songs")
    parser.add_argument("--length", type=float, default=5.0, help="The threshold to split the songs")
    args = parser.parse_args()
    main()
