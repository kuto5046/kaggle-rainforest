
import cv2
import random
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch.utils.data as data
import warnings
from pathlib import Path

PSEUDO_LABEL_VALUE = 1.0
"""
valid/testではtime flagは使わない
60s分にaudioの長さを揃える
10s単位に分割してリスト化してimage変換
"""
class SpectrogramDataset(data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 phase: str,
                 datadir: Path,
                 height: int,
                 width: int,
                 period: int,
                 shift_time: int,
                 strong_label_prob: int, 
                 waveform_transforms=None,
                 spectrogram_transforms=None,
                 melspectrogram_parameters={},
                 pcen_parameters = {}):
        self.df = df
        self.phase = phase
        self.datadir = datadir
        self.height = height
        self.width = width
        self.period = period
        self.shift_time = shift_time
        self.strong_label_prob = strong_label_prob
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
        self.pcen_parameters = pcen_parameters
        
        # pseudo labeling
        self.train_pseudo = pd.read_csv('./input/rfcx-species-audio-detection/train_ps60_thr9.csv').reset_index(drop=True)
        label_columns = [f"{col}" for col in range(24)]
        self.train_pseudo[label_columns] = np.where(self.train_pseudo[label_columns] > 0, PSEUDO_LABEL_VALUE, 0)  # label smoothing
        
        # self.train_pseudo = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        
        # train_pseudo = self.train_pseudo.sample(frac=0.5)  # 毎回50%sampling
        train_pseudo = self.train_pseudo
        
        sample = self.df.loc[idx, :]
        recording_id = sample["recording_id"]
        y, sr = sf.read(self.datadir / f"{recording_id}.flac")  # for default
        effective_length = sr * self.period
        total_time = 60  # 音声を全て60sに揃える
        y = adjust_audio_length(y, sr, total_time)
        if self.waveform_transforms:
            y = self.waveform_transforms(y)

        if self.phase == 'train':
            p = random.random()
            if p < self.strong_label_prob:
                y, labels = strong_clip_audio(self.df, y, sr, idx, effective_length, train_pseudo)
            else:
                y, labels = random_clip_audio(self.df, y, sr, idx, effective_length, train_pseudo)
            image = wave2image_normal(y, sr, self.width, self.height, self.melspectrogram_parameters)
            # image = wave2image_channel(y, sr, self.width, self.height, self.melspectrogram_parameters, self.pcen_parameters)
            # image = wave2image_custom_melfilter(y, sr, self.width, self.height, self.melspectrogram_parameters)
            return image, labels
        else:  # valid or test
            # PERIODO単位に分割(現在は6等分)
            split_y = split_audio(y, total_time, self.period, self.shift_time, sr)
            
            images = []
            # 分割した音声を一つずつ画像化してリストで返す
            for y in split_y:
                image = wave2image_normal(y, sr, self.width, self.height, self.melspectrogram_parameters)
                # image = wave2image_channel(y, sr, self.width, self.height, self.melspectrogram_parameters, self.pcen_parameters)
                # image = wave2image_custom_melfilter(y, sr, self.width, self.height, self.melspectrogram_parameters)
                images.append(image)

            if self.phase == 'valid':
                query_string = f"recording_id == '{recording_id}'"
                all_events = self.df.query(query_string)
                labels = np.zeros(24, dtype=np.float32)
                for idx, row in all_events.iterrows():
                    if row['data_type'] == 'tp':
                        labels[int(row['species_id'])] = 1.0
                    else:
                        labels[int(row['species_id'])] = -1.0

                labels = add_pseudo_label(labels, recording_id, train_pseudo)  # pseudo label
                return np.asarray(images), labels

            elif self.phase == 'test':
                labels = -1  # testなので-1を返す
                return np.asarray(images), labels
            else:
                raise NotImplementedError



def adjust_audio_length(y, sr, total_time=60):
    try:
        assert len(y)==total_time * sr
    except:
        print('Assert Error')
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
    return y

"""
############
wave → image 変換
############
"""
def wave2image_normal(y, sr, width, height, melspectrogram_parameters):
    """
    通常のmelspectrogram変換
    """
    melspec = librosa.feature.melspectrogram(y, sr=sr, **melspectrogram_parameters)
    melspec = librosa.power_to_db(melspec).astype(np.float32)

    image = mono_to_color(melspec)
    image = cv2.resize(image, (width, height))
    image = np.moveaxis(image, 2, 0)
    image = (image / 255.0).astype(np.float32)
    return image


def wave2image_channel(y, sr, width, height, melspectrogram_parameters, pcen_parameters):
    """
    ３つのmelspectrogramを作りstackして3chにして画像化
    """
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


def wave2image_custom_melfilter(y, sr, width, height, melspectrogram_parameters):
    filterbank = mel(
        sr, 
        n_fft=melspectrogram_parameters["n_fft"], 
        n_mels=melspectrogram_parameters["n_mels"], 
        fmin=melspectrogram_parameters["fmin"],
        fmax=melspectrogram_parameters["fmax"])
    S = np.abs(librosa.stft(y, n_fft=melspectrogram_parameters["n_fft"]))**melspectrogram_parameters["power"]
    melspec = np.dot(filterbank, S)
    melspec = librosa.power_to_db(melspec).astype(np.float32)

    image = mono_to_color(melspec)
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


def split_audio(y, total_time, period, shift_time, sr):
    # PERIODO単位に分割(現在は6等分)
    # split_y = np.split(y, total_time/PERIOD)
    num_data = int(total_time / shift_time)
    shift_length = sr*shift_time
    effective_length = sr*period
    split_y = []
    for i in range(num_data):
        start = shift_length * i
        finish = start + effective_length
        split_y.append(y[start:finish])
    
    return split_y

"""
############
clip method
only use train
############
"""

def random_clip_audio(df, y, sr, idx, effective_length, pseudo_df):
    len_y = len(y)
    if len_y < effective_length:
        new_y = np.zeros(effective_length, dtype=y.dtype)
        start = np.random.randint(effective_length - len_y)
        new_y[start:start + len_y] = y
        y = new_y.astype(np.float32)
    elif len_y > effective_length:
        start = np.random.randint(len_y - effective_length)
        end = start + effective_length
        y = y[start:end].astype(np.float32)
    else:
        y = y.astype(np.float32)
    
    # dfには同じrecording_idだけどclipしたt内に別のラベルがあるものもある
    # そこでそれには正しいidを付けたい
    recording_id = df.loc[idx, "recording_id"]
    query_string = f"recording_id == '{recording_id}'"

    # 同じrecording_idのものを
    all_tp_events = df.query(query_string)

    labels = np.zeros(len(df['species_id'].unique()), dtype=np.float32)
    for species_id in all_tp_events["species_id"].unique():
        labels[int(species_id)] = 1.0
    labels = add_pseudo_label(labels, recording_id, pseudo_df)
     
    return y, labels


# こちらはlabelに限定しているのでアライさんの処理は不要
# こちらはlabel付けにバッファを取っているのでアライさんの処理が必要
# これがベース
def strong_clip_audio(df, y, sr, idx, effective_length, pseudo_df):
    
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
    # try:  # tp data
    #     main_species_id = df.loc[idx, "species_id"]
    # except:  # fp data
    #     main_species_id = None

    query_string = f"recording_id == '{recording_id}' & "
    query_string += f"t_min < {ending_time} & t_max > {beginning_time}"

    # 同じrecording_idのものを
    all_events = df.query(query_string)

    labels = np.zeros(24, dtype=np.float32)
    for idx, row in all_events.iterrows(): 
        if row['data_type'] == 'tp':  # もしかしたらfpも混ざっているかもしれないので
            labels[int(row['species_id'])] = 1.0  # tp label
        else:
            labels[int(row['species_id'])] = -1.0  # fp label

    labels = add_pseudo_label(labels, recording_id, pseudo_df, beginning_time, ending_time)
    return y, labels


def add_pseudo_label(labels, recording_id, pseudo_df, beginning_time=None, ending_time=None):
    
    try:
        query_string = f"recording_id == '{recording_id}'"
        if beginning_time == None and ending_time == None:
            pass
        else:
            query_string += f" & t_min < {ending_time} & t_max > {beginning_time}"

        # 同じrecording_idのものを
        all_tp_events = pseudo_df.query(query_string)
        pseudo_labels = (np.sum(all_tp_events.loc[:, "0":"23"].values, axis=0) > 0).astype('float32')
        pseudo_labels = np.where(pseudo_labels > 0, PSEUDO_LABEL_VALUE, pseudo_labels)  # label smoothing
    # pseudo_dfがNoneだったり該当のrecording_idのpseudo_labelがない場合はスキップされる    
    except:
        pseudo_labels = np.zeros(24)
    labels = np.sum([labels, pseudo_labels], axis=0)  # labelsとpseudo labelを合体
    labels = np.where(labels >= 1.0, 1.0, labels).astype('float32')  # 1以上のものは1にする 
    return labels