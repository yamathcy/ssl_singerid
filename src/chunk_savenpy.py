import os,glob
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np

def main(args):
    class_to_id = {}
    id_to_class = {}
    args.save_dir = os.path.join(args.dir, "chunked")
    os.makedirs(args.save_dir, exist_ok=True)

    p = Path(args.dir)
    singers = sorted([entry.name for entry in p.iterdir() if entry.is_dir()])

    for i, category in enumerate(singers):
        class_to_id[category] = i
        id_to_class[i] = category

    for singer in tqdm(singers):
        print("load singer: {}".format(singer))
        singer_path = os.path.join(args.dir, singer)
        p = Path(singer_path)
        albums = sorted([entry.name for entry in p.iterdir() if entry.is_dir()])
        singer_label = singer
        for num, album in enumerate(albums):
            audio_list = sorted(glob.glob(os.path.join(args.dir, singer, album, "*vocal.wav")))
            for song, file_path in enumerate(audio_list):
                # audio, sr = librosa.load(file_path, sr=sr)
                audio, fs = torchaudio.load(file_path)
                audio = derive_desired_wav(audio, fs, args.sr)
                audio = torch.FloatTensor(audio)
                # print(audio.shape)
                trimmed = chunk_audio(audio, args.chunk_length, fs, rms_filter=True)
                label_for_trimmed = [singer_label for x in range(len(trimmed))]
                for id,  (chunk, lab) in enumerate(zip(trimmed,label_for_trimmed)):
                    try:
                        save_path = os.path.join(args.save_dir, "{}-{}-{}-{}.npy".format(lab, num, song, id))
                        if not os.path.exists(save_path):
                            chunk = chunk.detach().numpy()
                            np.save(save_path, chunk)
                    except:
                        pass
                del audio

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

def rms_filtering(wav, th=0.01):
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


if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="The path to the dataset")
    parser.add_argument("--save_dir", type=str, help="The path to the dataset")
    parser.add_argument("--chunk_length", type=int, default=5, help="The length of chunk in seconds")
    parser.add_argument("--sr", type=int, default=16000, help="The sampling rate of the audio")

    args = parser.parse_args()
    main(args)
