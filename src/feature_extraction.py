import numpy as np
import librosa  # type: ignore
import os
import glob
import warnings
import pickle

DATASET_DIR  = "data/recordings"
TARGET_SR    = 16000
N_MFCC       = 13
N_MELS       = 40
WIN_MS       = 25
HOP_MS       = 10
PRE_EMPH     = 0.97

def extract_features(file_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, temp = librosa.load(file_path, sr=TARGET_SR)

    y = np.append(y[0], y[1:] - PRE_EMPH * y[:-1])

    win_length = int(TARGET_SR * WIN_MS / 1000)
    hop_length = int(TARGET_SR * HOP_MS / 1000)
    n_fft = 512

    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y) + 1), mode="reflect")

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=TARGET_SR,
        n_mfcc=N_MFCC,
        n_mels=N_MELS,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window="hamming",
    )

    if mfcc.shape[1] < 9:
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, 9 - mfcc.shape[1])),
            mode="edge",
        )

    delta1 = librosa.feature.delta(mfcc, width=9)
    delta2 = librosa.feature.delta(mfcc, width=9, order=2)

    features = np.vstack([mfcc, delta1, delta2]).T

    mean = features.mean(axis=0)
    std = features.std(axis=0)
    features = (features - mean) / (std + 1e-8)

    return features

def split_dataset(dataset_dir = DATASET_DIR):
    train_data = {}
    test_data = {}

    for i in range(10):
        train_data[str(i)] = []
        test_data[str(i)] = []

    wav_files = sorted(glob.glob(os.path.join(dataset_dir, "*.wav")))

    skipped = 0
    for fp in wav_files:
        fname = os.path.basename(fp)
        try:
            parts = fname.split("_")
            digit = parts[0]
            speaker_id = parts[1].replace("speaker", "")
        except IndexError:
            skipped += 1
            continue

        feats = extract_features(fp)

        if speaker_id in ["1", "2", "3", "4", "5"]:
            train_data[digit].append(feats)
        elif speaker_id == "6":
            test_data[digit].append(feats)

    with open("train_data.pkl", "wb") as f:
        pickle.dump(train_data, f)
    with open("test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)

    return train_data, test_data

train_set, test_set = split_dataset()

total_train = 0
total_test = 0

for d in range(10):
    train_dataset = train_set[str(d)]
    test_dataset = test_set[str(d)]
    n_train = len(train_dataset)
    n_test = len(test_dataset)
    avg_fr= int(np.mean([x.shape[0] for x in train_dataset])) if n_train else 0
    total_train += n_train
    total_test += n_test
