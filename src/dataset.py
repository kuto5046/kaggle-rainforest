
import cv2
import random
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch.utils.data as data

from pathlib import Path


PERIOD = 10

class SpectrogramDataset(data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 datadir: Path,
                 height: int,
                 width: int,
                 waveform_transforms=None,
                 spectrogram_transforms=None,
                 melspectrogram_parameters={},
                 pcen_parameters = {}):
        self.df = df
        self.datadir = datadir
        self.height = height
        self.width = width
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
        self.pcen_parameters = pcen_parameters
        self.count = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        recording_id = sample["recording_id"]
        main_species_id = sample["species_id"]
        # y, sr = sf.read(self.datadir / str(main_species_id) / f"{recording_id}.wav")  # for resample
        y, sr = sf.read(self.datadir / f"{recording_id}.flac")  # for default
        effective_length = sr * PERIOD

        # 破損しているデータはskip
        if len(y) == 0:
            self.count += 1
            print(f"num_unknown_audio: {self.count}")
            print(f"wav_name: {recording_id}")

        y, labels = clip_time_audio2(self.df, y, sr, idx, effective_length, main_species_id)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)
        
        image = wave2image(y, sr, self.width, self.height, self.melspectrogram_parameters, self.pcen_parameters)

        return image, labels


"""
valid/testではtime flagは使わない
60s分にaudioの長さを揃える
10s単位に分割してリスト化してimage変換
"""
class SpectrogramValDataset(data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 datadir: Path,
                 height: int,
                 width: int,
                 shift_time: int,
                 waveform_transforms=None,
                 spectrogram_transforms=None,
                 melspectrogram_parameters={},
                 pcen_parameters={}):

        self.df = df
        self.datadir = datadir
        self.height = height
        self.width = width
        self.shift_time = shift_time
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
        self.pcen_parameters = pcen_parameters

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        recording_id = sample["recording_id"]
        main_species_id = sample["species_id"]

        total_time = 60  # 音声を全て60sに揃える
        # y, sr = sf.read(self.datadir / str(main_species_id) / f"{recording_id}.wav")  # for resample
        y, sr = sf.read(self.datadir / f"{recording_id}.flac")  # for default

        if self.waveform_transforms:
            y = self.waveform_transforms(y)

        # データの長さを全てtotal_time分にする
        len_y = len(y)
        total_length = total_time * sr
        if len_y < total_length:
            new_y = np.zeros(total_length, dtype=y.dtype)
            start = np.random.randint(total_length - len_y)
            new_y[start:start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > total_length:
            start = np.random.randint(len_y - total_length)
            y = y[start:start + total_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        # PERIODO単位に分割(現在は6等分)
        split_y = split_audio(y, total_time, self.shift_time, sr)
        
        images = []
        # 分割した音声を一つずつ画像化してリストで返す
        for y in split_y:
            image = wave2image(y, sr, self.width, self.height, self.melspectrogram_parameters, self.pcen_parameters)
            images.append(image)

        labels = np.zeros(len(self.df['species_id'].unique()), dtype=np.float32)
        labels[main_species_id] = 1.0
        
        return np.asarray(images), labels


class SpectrogramTestDataset(data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 datadir: Path,
                 height: int,
                 width: int,
                 shift_time: int,
                 waveform_transforms=None,
                 spectrogram_transforms=None,
                 melspectrogram_parameters={},
                 pcen_parameters={}):
        self.df = df
        self.datadir = datadir
        self.height = height
        self.width = width
        self.shift_time = shift_time
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
        self.pcen_parameters = pcen_parameters

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        recording_id = sample["recording_id"]
        total_time = 60  # 音声を全て60sに揃える
        y, sr = sf.read(self.datadir / f"{recording_id}.flac")

        if self.waveform_transforms:
            y = self.waveform_transforms(y)

        # データの長さを全てtotal_time分にする
        len_y = len(y)
        total_length = total_time * sr
        if len_y < total_length:
            new_y = np.zeros(total_length, dtype=y.dtype)
            start = np.random.randint(total_length - len_y)
            new_y[start:start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > total_length:
            start = np.random.randint(len_y - total_length)
            y = y[start:start + total_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        # PERIODO単位に分割(現在は6等分)
        split_y = split_audio(y, total_time, self.shift_time ,sr)

        images = []
        # 分割した音声を一つずつ画像化してリストで返す
        for y in split_y:
            image = wave2image(y, sr, self.width, self.height, self.melspectrogram_parameters, self.pcen_parameters)
            images.append(image)

        labels = -1  # labelないので-1を返す
        return np.asarray(images), labels


def wave2image(y, sr, width, height, melspectrogram_parameters, pcen_parameters):
    melspec = librosa.feature.melspectrogram(y, sr=sr, **melspectrogram_parameters)
    pcen = librosa.pcen(melspec, sr=sr, **pcen_parameters)
    clean_mel = librosa.power_to_db(melspec ** 1.5)
    melspec = librosa.power_to_db(melspec)

    norm_melspec = normalize_melspec(melspec)
    norm_pcen = normalize_melspec(pcen)
    norm_clean_mel = normalize_melspec(clean_mel)
    image = np.stack([norm_melspec, norm_pcen, norm_clean_mel], axis=-1)

    # image = mono_to_color(melspec)
    image = cv2.resize(image, (width, height))
    image = np.moveaxis(image, 2, 0)
    image = (image / 255.0).astype(np.float32)
    return image


def normalize_melspec(X: np.ndarray):
    eps = 1e-6
    mean = X.mean()
    X = X - mean
    std = X.std()
    Xstd = X / (std + eps)
    norm_min, norm_max = Xstd.min(), Xstd.max()
    if (norm_max - norm_min) > eps:
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


def mono_to_color(X: np.ndarray,
                  mean=None,
                  std=None,
                  norm_max=None,
                  norm_min=None,
                  eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


def split_audio(y, total_time, shift_time, sr):
    # PERIODO単位に分割(現在は6等分)
    # split_y = np.split(y, total_time/PERIOD)
    num_data = int(total_time / shift_time)
    shift_length = sr*shift_time
    duration_length = sr*PERIOD
    split_y = []
    for i in range(num_data):
        start = shift_length * i
        finish = start + duration_length
        split_y.append(y[start:finish])
    
    return split_y


# こちらはlabelに限定しているのでアライさんの処理は不要
def clip_time_audio1(df, y, sr, idx, effective_length, main_species_id):
    # dfから時間ラベルをflame単位で取得しclip
    t_min = int(np.round(df.t_min.values[idx]*sr))
    t_max = int(np.round(df.t_max.values[idx]*sr))
    y = y[t_min:t_max]

    len_y = len(y)
    effective_length = sr * PERIOD
    if len_y < effective_length:
        new_y = np.zeros(effective_length, dtype=y.dtype)
        start = np.random.randint(effective_length - len_y)
        new_y[start:start + len_y] = y
        y = new_y.astype(np.float32)
    elif len_y > effective_length:
        start = np.random.randint(len_y - effective_length)
        y = y[start:start + effective_length].astype(np.float32)
    else:
        y = y.astype(np.float32)

    labels = np.zeros(len(df['species_id'].unique()), dtype=np.float32)
    labels[main_species_id] = 1.0
    
    return y, labels


# こちらはlabel付けにバッファを取っているのでアライさんの処理が必要
def clip_time_audio2(df, y, sr, idx, effective_length, main_species_id):
    
    t_min = df.t_min.values[idx]*sr
    t_max = df.t_max.values[idx]*sr

    # Positioning sound slice
    t_center = np.round((t_min + t_max) / 2)
    
    # 開始点の仮決定 
    beginning = t_center - effective_length / 2
    # overしたらaudioの最初からとする
    if beginning < 0:
        beginning = 0
    beginning = np.random.randint(beginning, t_center)

    # 開始点と終了点の決定
    ending = beginning + effective_length
    # overしたらaudioの最後までとする
    if ending > len(y):
        ending = len(y)
    beginning = ending - effective_length

    y = y[beginning:ending].astype(np.float32)
    # assert len(y)==effective_length, f"not much audio length in {idx}. The length of y is {len(y)} not {effective_length}."

    # TODO 以下アライさんが追加した部分
    # https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/200922#1102470

    # flame→time変換
    beginning_time = beginning / sr
    ending_time = ending / sr

    # dfには同じrecording_idだけどclipしたt内に別のラベルがあるものもある
    # そこでそれには正しいidを付けたい
    recording_id = df.loc[idx, "recording_id"]
    query_string = f"recording_id == '{recording_id}' & "
    query_string += f"t_min < {ending_time} & t_max > {beginning_time}"

    # 同じrecording_idのものを
    all_tp_events = df.query(query_string)

    labels = np.zeros(len(df['species_id'].unique()), dtype=np.float32)
    for species_id in all_tp_events["species_id"].unique():
        if species_id == main_species_id:
            labels[int(species_id)] = 1.0  # main label
        else:
            labels[int(species_id)] = 0.5  # secondaly label
    
    return y, labels

# 10sのうちの音声の比率でラベル付け
def clip_time_audio3(df, y, sr, idx, effective_length, main_species_id):
    
    t_min = df.t_min.values[idx]*sr
    t_max = df.t_max.values[idx]*sr

    # Positioning sound slice
    t_center = np.round((t_min + t_max) / 2)
    
    # 開始点の仮決定 
    beginning = t_center - effective_length / 2
    # overしたらaudioの最初からとする
    if beginning < 0:
        beginning = 0
    beginning = np.random.randint(beginning, t_center)

    # 開始点と終了点の決定
    ending = beginning + effective_length
    # overしたらaudioの最後までとする
    if ending > len(y):
        ending = len(y)
    beginning = ending - effective_length

    y = y[beginning:ending].astype(np.float32)
    assert len(y)==effective_length, f"not much audio length in {idx}. The length of y is {len(y)} not {effective_length}."

    # 以下アライさんが追加した部分
    # https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/200922#1102470

    # flame→time変換
    beginning_time = beginning / sr
    ending_time = ending / sr
    effective_time = effective_length / sr

    # dfには同じrecording_idだけどclipしたt内に別のラベルがあるものもある
    # そこでそれには正しいidを付けたい
    recording_id = df.loc[idx, "recording_id"]
    query_string = f"recording_id == '{recording_id}' & "
    query_string += f"t_min < {ending_time} & t_max > {beginning_time}"

    # 同じrecording_idのものを
    all_tp_events = df.query(query_string)

    labels = np.zeros(len(df['species_id'].unique()), dtype=np.float32)

    """
    effective timeのうち該当のlabelが何秒間あるかでラベル付け
    重複しているものは足し合わせる
    別の種の場合はいいが同じ種の場合は重複はたしあわせないほうがいい？
    """
    for _, raw in all_tp_events.iterrows():
        labels[int(raw['species_id'])] += (raw['t_max'] - raw['t_min']) / effective_time

    return y, labels