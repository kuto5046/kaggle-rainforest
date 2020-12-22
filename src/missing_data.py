import os
import warnings
from pathlib import Path
from joblib import delayed, Parallel

import librosa
import audioread
import soundfile as sf
from tqdm import tqdm
import pandas as pd

TARGET_SR = 48000
ext = '.flac'
re_ext = '.wav'
# NUM_THREAD = 4  # for joblib.Parallel

# # define resampling function
warnings.simplefilter("ignore")

TRAIN_AUDIO_DIR = Path("../input/rfcx-species-audio-detection/train/")
TEST_AUDIO_DIR = Path("../input/rfcx-species-audio-detection/test/")

train_tp = pd.read_csv("../input/rfcx-species-audio-detection/train_tp.csv")
train_fp = pd.read_csv("../input/rfcx-species-audio-detection/train_fp.csv")
train_tp["data_type"] = "tp"
train_fp["data_type"] = "fp"
train = pd.concat([train_tp, train_fp])

# # make directories for saving resampled audio
TRAIN_RESAMPLED_DIR = Path("../input/rainforest-resample-data/resample-train/")
TEST_RESAMPLED_DIR = Path("../input/rainforest-resample-data/resample-test/")
TRAIN_RESAMPLED_DIR.mkdir(parents=True, exist_ok=True)
TEST_RESAMPLED_DIR.mkdir(parents=True, exist_ok=True)

# # extract "recording_id" and  "species_id"
train_audio_infos = train[["recording_id", "species_id"]].values.tolist()
test_audio_infos = list(TEST_AUDIO_DIR.glob("*.flac"))

# make directories for each species
for species_id in train_tp.species_id.unique():
    species_dir = TRAIN_RESAMPLED_DIR / str(species_id)
    species_dir.mkdir(parents=True, exist_ok=True)


# train missing data processing
train_resample_audio_infos = list(TRAIN_RESAMPLED_DIR.glob("*/*.wav"))
missing_train_audio_id = []

for train_resample_dir in tqdm(train_resample_audio_infos):
    y, _ = librosa.load(train_resample_dir, sr=TARGET_SR, mono=True, res_type="kaiser_fast")

    # 1s以下のものは破損しているとみなす
    length = TARGET_SR * 1
    if len(y) < length:
        print(train_resample_dir.stem)
        missing_train_audio_id.append(train_resample_dir.stem)

for recording_id, species_id in tqdm(train_audio_infos):
    if recording_id in missing_train_audio_id:
        y, _ = librosa.load(TRAIN_AUDIO_DIR / (recording_id + ext), sr=TARGET_SR, mono=True, res_type="kaiser_fast")
        sf.write(TRAIN_RESAMPLED_DIR / str(species_id) / (recording_id + re_ext), y, samplerate=TARGET_SR)


# test missing data processing
test_resample_audio_infos = list(TEST_RESAMPLED_DIR.glob("*.wav"))
missing_test_audio_id = []
for test_resample_dir in tqdm(test_resample_audio_infos):
    y, _ = librosa.load(test_resample_dir, sr=TARGET_SR, mono=True, res_type="kaiser_fast")
    length = TARGET_SR * 60
    if len(y) != length:
        print(test_resample_dir.stem)
        missing_test_audio_id.append(test_resample_dir.stem)

for audio_dir in tqdm(test_audio_infos):
    if audio_dir.stem in missing_test_audio_id:
        y, _ = librosa.load(audio_dir, sr=TARGET_SR, mono=True, res_type="kaiser_fast")
        sf.write(TEST_RESAMPLED_DIR / audio_dir.name.replace(ext, re_ext), y, samplerate=TARGET_SR) 


# add information of resampled audios to train.csv
# train["resampled_filename"] = train["recording_id"].map(lambda x: x + ".wav")
# train.to_csv(TRAIN_RESAMPLED_DIR / "train.csv", index=False)
