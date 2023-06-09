
import numpy as np

import librosa


# dict型のラベル-ID変換表を更新する
def label_to_id(label, id_dict):
    id = len(id_dict)
    if label not in id_dict.keys():
        id_dict[label] = id

    return id_dict[label], id_dict


def rms_filtering(wav, th=0.005):
    rms = librosa.feature.rms(y=wav).squeeze()
    return rms.mean() > th


def chunk_audio(audio, chunk_length, sr, rms_filter=False):
    # Calculate number of samples per chunk	
    samples_per_chunk = int(chunk_length * sr)
    # Chunk the audio
    audio_chunks = [audio[...,i:i + samples_per_chunk] for i in range(0, len(audio), samples_per_chunk)]
    if rms_filter:
        audio_chunks = [item for item in audio_chunks if rms_filtering(item)]
    return audio_chunks
