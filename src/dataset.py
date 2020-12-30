
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
                 melspectrogram_parameters={}):
        self.df = df
        self.datadir = datadir
        self.height = height
        self.width = width
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
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

        # y, labels = self.clip_time_audio1(y, sr, idx, effective_length, main_species_id)
        y, labels = self.clip_time_audio2(y, sr, idx, effective_length, main_species_id)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)

        # dataframeから周波数帯域を取り出し更新
        # fmin = self.df.f_min.values[idx]*0.9  # buffer
        # fmax = self.df.f_max.values[idx]*1.1  # buffer
        # self.melspectrogram_parameters["fmin"] = fmin
        # self.melspectrogram_parameters["fmax"] = fmax

        melspec = librosa.feature.melspectrogram(y, sr=sr, **self.melspectrogram_parameters)
        melspec = librosa.power_to_db(melspec).astype(np.float32)

        if self.spectrogram_transforms:
            melspec = self.spectrogram_transforms(melspec)


        image = mono_to_color(melspec)
        # height, width, _ = image.shape
        # image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = cv2.resize(image, (self.width, self.height))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

        return image, labels

    # こちらはlabelに限定しているのでアライさんの処理は不要
    def clip_time_audio1(self, y, sr, idx, effective_length, main_species_id):
        # dfから時間ラベルをflame単位で取得しclip
        t_min = int(np.round(self.df.t_min.values[idx]*sr))
        t_max = int(np.round(self.df.t_max.values[idx]*sr))
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

        labels = np.zeros(len(self.df['species_id'].unique()), dtype=np.float32)
        labels[main_species_id] = 1.0
        
        return y, labels

    # こちらはlabel付けにバッファを取っているのでアライさんの処理が必要
    def clip_time_audio2(self, y, sr, idx, effective_length, main_species_id):
        
        t_min = self.df.t_min.values[idx]*sr
        t_max = self.df.t_max.values[idx]*sr

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
        recording_id = self.df.loc[idx, "recording_id"]
        query_string = f"recording_id == '{recording_id}' & "
        query_string += f"t_min < {ending_time} & t_max > {beginning_time}"

        # 同じrecording_idのものを
        all_tp_events = self.df.query(query_string)

        labels = np.zeros(len(self.df['species_id'].unique()), dtype=np.float32)
        for species_id in all_tp_events["species_id"].unique():
            if species_id == main_species_id:
                labels[int(species_id)] = 1.0  # main label
            else:
                labels[int(species_id)] = 1.0  # secondaly label
        
        return y, labels

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
                 melspectrogram_parameters={}):
        self.df = df
        self.datadir = datadir
        self.height = height
        self.width = width
        self.shift_time = shift_time
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
        self.count = 0

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
        split_y = split_audio(y, total_time, self.shift_time ,sr)
        
        images = []
        # 分割した音声を一つずつ画像化してリストで返す
        for y in split_y:
            melspec = librosa.feature.melspectrogram(y, sr=sr, **self.melspectrogram_parameters)
            melspec = librosa.power_to_db(melspec).astype(np.float32)

            if self.spectrogram_transforms:
                melspec = self.spectrogram_transforms(melspec)
            else:
                pass
            image = mono_to_color(melspec)
            # height, width, _ = image.shape
            # image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
            image = cv2.resize(image, (self.width, self.height))
            image = np.moveaxis(image, 2, 0)
            image = (image / 255.0).astype(np.float32)
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
                 melspectrogram_parameters={}):
        self.df = df
        self.datadir = datadir
        self.height = height
        self.width = width
        self.shift_time = shift_time
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
        self.count = 0

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
            melspec = librosa.feature.melspectrogram(y, sr=sr, **self.melspectrogram_parameters)
            melspec = librosa.power_to_db(melspec).astype(np.float32)

            if self.spectrogram_transforms:
                melspec = self.spectrogram_transforms(melspec)
            else:
                pass
            image = mono_to_color(melspec)
            # height, width, _ = image.shape
            # image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
            image = cv2.resize(image, (self.width, self.height))
            image = np.moveaxis(image, 2, 0)
            image = (image / 255.0).astype(np.float32)
            images.append(image)

        labels = -1  # labelないので-1を返す
        return np.asarray(images), labels


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