
import cv2
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
                 img_size=224,
                 waveform_transforms=None,
                 spectrogram_transforms=None,
                 melspectrogram_parameters={}):
        self.df = df
        self.datadir = datadir
        self.img_size = img_size
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["resampled_filename"]
        main_species_id = sample["species_id"]

        y, sr = sf.read(self.datadir / str(main_species_id) / wav_name)
        effective_length = sr * PERIOD

        y, labels = self.clip_time_audio1(y, sr, idx, effective_length, main_species_id)
        # y, labels = self.clip_time_audio2(y, sr, idx, effective_length, main_species_id)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)

        # dataframeから周波数帯域を取り出し更新
        fmin = int(np.round(self.df.f_min.values[idx]))*0.9  # buffer
        fmax = int(np.round(self.df.f_max.values[idx]))*1.1  # buffer
        self.melspectrogram_parameters["fmin"] = fmin
        self.melspectrogram_parameters["fmax"] = fmax

        melspec = librosa.feature.melspectrogram(y, sr=sr, **self.melspectrogram_parameters)
        melspec = librosa.power_to_db(melspec).astype(np.float32)

        if self.spectrogram_transforms:
            melspec = self.spectrogram_transforms(melspec)
        else:
            pass

        image = mono_to_color(melspec)
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
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
                labels[int(species_id)] = 0.5  # secondaly label
        
        return y, labels


class SpectrogramTestDataset(data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 datadir: Path,
                 img_size=224,
                 waveform_transforms=None,
                 spectrogram_transforms=None,
                 melspectrogram_parameters={}):
        self.df = df
        self.datadir = datadir
        self.img_size = img_size
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        recording_id = sample["recording_id"]

        y, sr = sf.read(self.datadir / f"{recording_id}.wav")

        if self.waveform_transforms:
            y = self.waveform_transforms(y)
        else:
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

        melspec = librosa.feature.melspectrogram(y, sr=sr, **self.melspectrogram_parameters)
        melspec = librosa.power_to_db(melspec).astype(np.float32)

        if self.spectrogram_transforms:
            melspec = self.spectrogram_transforms(melspec)
        else:
            pass

        image = mono_to_color(melspec)
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

        return image


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

