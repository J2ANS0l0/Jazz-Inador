import csv
from pathlib import Path

import librosa
import numpy as np

project_dir = Path(__file__).parent.parent
input_dir = project_dir / "Data/genres_original"
csv_path = project_dir / "Data/features_3_sec.csv"
segment_seconds = 3
segment_count = 10
mfcc_count = 20

columns = [
    "filename",
    "length",
    "chroma_stft_mean",
    "chroma_stft_var",
    "rms_mean",
    "rms_var",
    "spectral_centroid_mean",
    "spectral_centroid_var",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_var",
    "rolloff_mean",
    "rolloff_var",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_var",
    "harmony_mean",
    "harmony_var",
    "perceptr_mean",
    "perceptr_var",
    "tempo",
]
for index in range(1, mfcc_count + 1):
    columns.extend([f"mfcc{index}_mean", f"mfcc{index}_var"])
columns.append("label")


def extract_features(segment, sr):
    chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
    rms = librosa.feature.rms(y=segment)
    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
    zero_crossing = librosa.feature.zero_crossing_rate(y=segment)
    harmonic, percussive = librosa.effects.hpss(segment)
    tempo, _ = librosa.beat.beat_track(y=segment, sr=sr)
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=mfcc_count)

    values = {
        "length": len(segment),
        "chroma_stft_mean": chroma.mean(),
        "chroma_stft_var": chroma.var(),
        "rms_mean": rms.mean(),
        "rms_var": rms.var(),
        "spectral_centroid_mean": centroid.mean(),
        "spectral_centroid_var": centroid.var(),
        "spectral_bandwidth_mean": bandwidth.mean(),
        "spectral_bandwidth_var": bandwidth.var(),
        "rolloff_mean": rolloff.mean(),
        "rolloff_var": rolloff.var(),
        "zero_crossing_rate_mean": zero_crossing.mean(),
        "zero_crossing_rate_var": zero_crossing.var(),
        "harmony_mean": harmonic.mean(),
        "harmony_var": harmonic.var(),
        "perceptr_mean": percussive.mean(),
        "perceptr_var": percussive.var(),
        "tempo": float(np.atleast_1d(tempo)[0]),
    }
    for index in range(mfcc_count):
        values[f"mfcc{index + 1}_mean"] = mfcc[index].mean()
        values[f"mfcc{index + 1}_var"] = mfcc[index].var()
    return values


with open(csv_path, "w", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=columns)
    writer.writeheader()

    for audio_path in sorted(input_dir.rglob("*.wav")):
        genre = audio_path.parent.name
        numeric_id = audio_path.stem.split(".")[-1]

        try:
            y, sr = librosa.load(audio_path, sr=None)
        except Exception as error:
            print(f"Skipped {audio_path.name}: {error}")
            continue

        segment_length = segment_seconds * sr
        y = librosa.util.fix_length(y, size=segment_count * segment_length)

        for fragment in range(segment_count):
            segment = y[fragment * segment_length:(fragment + 1) * segment_length]
            row = extract_features(segment, sr)
            row["filename"] = f"{genre}{numeric_id}{fragment:03d}.wav"
            row["label"] = genre
            writer.writerow(row)
            csv_file.flush()
            print(f"Processed {row['filename']}")