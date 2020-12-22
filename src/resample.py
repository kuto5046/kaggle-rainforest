import os
import warnings
from pathlib import Path
from joblib import delayed, Parallel

import librosa
import audioread
import soundfile as sf
from tqdm import tqdm
import pandas as pd
from joblib import delayed, Parallel


def resample():
    TARGET_SR = 48000
    ext = '.flac'
    re_ext = '.wav'
    NUM_THREAD = 4  # for joblib.Parallel

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

    for recording_id, species_id in tqdm(train_audio_infos):
        species_dir = TRAIN_RESAMPLED_DIR / str(species_id)
        y, _ = librosa.load(TRAIN_AUDIO_DIR / (recording_id + ext), sr=TARGET_SR, mono=True, res_type="kaiser_fast")
        sf.write(species_dir / (recording_id + re_ext), y, samplerate=TARGET_SR)

    for audio_dir in tqdm(test_audio_infos):
        y, _ = librosa.load(audio_dir, sr=TARGET_SR, mono=True, res_type="kaiser_fast")
        sf.write(TEST_RESAMPLED_DIR / audio_dir.name.replace(ext, re_ext), y, samplerate=TARGET_SR) 
    
    # add information of resampled audios to train.csv
    train["resampled_filename"] = train["recording_id"].map(lambda x: x + ".wav")
    train.to_csv(TRAIN_RESAMPLED_DIR / "train.csv", index=False)


if __name__=="__main__":
    resample()


        

