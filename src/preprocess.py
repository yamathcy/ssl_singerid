import os, sys
import numpy as np
import librosa
import pandas as pd 
import torchaudio
from tqdm import tqdm
import argparse
import glob
"""
preprocess.py
特徴量の計算やデータの前処理等が必要な場合ここに書く
"""


def resample_dataset(data, sr):
    """
    Sample code for resampling the vocal audio data from 44.1kHz to 16kHz, which is the input requirement of wav2vec 2.0
    """
    audio_list = sorted(glob.glob(os.path.join(data, "**/*vocal.wav"), recursive=True))
    for dir in tqdm(audio_list):
        audio_path = dir
        save_path = audio_path
        audio, fs = torchaudio.load(audio_path)  # audio: [1, N] for mono or [2, N] for stero
        # resample
        if fs != sr:
            # resample
            audio_resample = torchaudio.transforms.Resample(orig_freq=fs, new_freq=sr)(audio)
        else:
            audio_resample = audio

        # mono
        if audio_resample.shape[0] == 2:
            # stero input
            audio_resample = audio_resample.mean(dim=0, keepdim=True)

        # save the file
        torchaudio.save(save_path, audio_resample, sr)


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/ubuntu/dataset/artist20", help="The path to the dataset")
    parser.add_argument("--sr", type=int, default=16000, help="The threshold to split the songs")
    args = parser.parse_args()
    resample_dataset(args)