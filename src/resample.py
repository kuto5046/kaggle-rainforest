import os
import warnings
from pathlib import Path
from joblib import delayed, Parallel

import librosa
import audioread
import soundfile as sf
import tqdm
import pandas as pd

TARGET_SR = 32000
# NUM_THREAD = 4  # for joblib.Parallel

TRAIN_AUDIO_DIR = Path("../input/train/")
TEST_AUDIO_DIR = Path("../input/test/")
TRAIN_TP_RESAMPLED_DIR = Path("../input/resample-train-tp/")
TRAIN_FP_RESAMPLED_DIR = Path("../input/resample-train-fp/")
TEST_RESAMPLED_DIR = Path("../input/resample-test/")


test_flacfiles = list(TEST_AUDIO_DIR.glob("*.flac"))

train_tp = pd.read_csv("../input/train_tp.csv")
train_fp = pd.read_csv("../input/train_fp.csv")

# # make directories for saving resampled audio
TRAIN_TP_RESAMPLED_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_FP_RESAMPLED_DIR.mkdir(parents=True, exist_ok=True)
TEST_RESAMPLED_DIR.mkdir(parents=True, exist_ok=True)

# # extract "recording_id" and  "species_id"
train_tp_audio_infos = train_tp[["recording_id", "species_id"]].values.tolist()
train_fp_audio_infos = train_fp[["recording_id", "species_id"]].values.tolist()
test_audio_infos = list(TEST_AUDIO_DIR.glob("*.flac"))

# make directories for each species
for species_id in train_tp.species_id.unique():
    species_dir = TRAIN_TP_RESAMPLED_DIR / str(species_id)
    species_dir.mkdir(parents=True, exist_ok=True)

for species_id in train_fp.species_id.unique():
    species_dir = TRAIN_FP_RESAMPLED_DIR / str(species_id)
    species_dir.mkdir(parents=True, exist_ok=True)

# # define resampling function
warnings.simplefilter("ignore")


def tp_resample(recording_id: str, species_id: str, target_sr: int):
    ext = '.flac'
    re_ext = '.wav'
    species_dir = TRAIN_TP_RESAMPLED_DIR / str(species_id)
    y, _ = librosa.load(TRAIN_AUDIO_DIR / (recording_id + ext), sr=target_sr, mono=True, res_type="kaiser_fast")
    sf.write(species_dir / (recording_id + re_ext), y, samplerate=target_sr)

def fp_resample(recording_id: str, species_id: str, target_sr: int):
    ext = '.flac'
    re_ext = '.wav'
    species_dir = TRAIN_FP_RESAMPLED_DIR / str(species_id)
    y, _ = librosa.load(TRAIN_AUDIO_DIR / (recording_id + ext), sr=target_sr, mono=True, res_type="kaiser_fast")
    sf.write(species_dir / (recording_id + re_ext), y, samplerate=target_sr)

def test_resample(audio_dir:str, target_sr: int):
    y, _ = librosa.load(audio_dir, sr=target_sr, mono=True, res_type="kaiser_fast")
    sf.write(TEST_RESAMPLED_DIR / audio_dir.name.replace('.flac', '.wav'), y, samplerate=target_sr)  


for recording_id, species_id in tqdm(train_tp_audio_infos):
    tp_resample(recording_id, species_id, TARGET_SR)

for recording_id, species_id in tqdm(train_fp_audio_infos):
    fp_resample(recording_id, species_id, TARGET_SR)

for audio_dir in tqdm(test_audio_infos):
    test_resample(audio_dir, TARGET_SR)


# # add information of resampled audios to train.csv
train_tp["resampled_filename"] = train_tp["recording_id"].map(lambda x: x + ".wav")
train_fp["resampled_filename"] = train_fp["recording_id"].map(lambda x: x + ".wav")

train_tp.to_csv(TRAIN_TP_RESAMPLED_DIR / "train_tp.csv", index=False)
train_fp.to_csv(TRAIN_FP_RESAMPLED_DIR / "train_fp.csv", index=False)
