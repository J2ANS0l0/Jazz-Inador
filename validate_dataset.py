import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path

import soundfile
from PIL import Image

project_dir = Path(__file__).parent
data_dir = project_dir / "Data"
wav_dir = data_dir / "genres_original"
png_dir = data_dir / "mel-spectrograms"
csv_path = data_dir / "features_3_sec.csv"

genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
samples_per_genre = 100
segment_count = 10
mfcc_count = 20
expected_length = 66150
image_size = (432, 288)
min_duration = 29.0
max_duration = 31.0
excluded_audios = {"jazz00054"}
max_errors_shown = 20

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

wav_pattern = re.compile(r"^([a-z]+)\.(\d{5})\.wav$")
png_pattern = re.compile(r"^([a-z]+)(\d{5})(\d{3})\.png$")
csv_name_pattern = re.compile(r"^([a-z]+)(\d{5})(\d{3})\.wav$")

expected_audios = {
    f"{genre}{number:05d}": genre
    for genre in genres
    for number in range(samples_per_genre)
    if f"{genre}{number:05d}" not in excluded_audios
}
expected_fragments = {
    f"{audio}{fragment:03d}": genre
    for audio, genre in expected_audios.items()
    for fragment in range(segment_count)
}


def validate_wavs():
    errors = []
    if not wav_dir.is_dir():
        return [f"Folder not found: {wav_dir}"]

    for genre in genres:
        genre_dir = wav_dir / genre
        if not genre_dir.is_dir():
            errors.append(f"Missing folder: {genre}")
            continue

        found_ids = set()
        for path in sorted(genre_dir.glob("*.wav")):
            match = wav_pattern.match(path.name)
            if not match or match.group(1) != genre:
                errors.append(f"Invalid filename: {genre}/{path.name}")
                continue

            numeric_id = match.group(2)
            found_ids.add(numeric_id)
            if f"{genre}{numeric_id}" in excluded_audios:
                continue

            try:
                info = soundfile.info(path)
            except Exception as error:
                errors.append(f"Unreadable file: {genre}/{path.name}: {error}")
                continue

            if not min_duration <= info.duration <= max_duration:
                errors.append(f"Unexpected duration {info.duration:.2f} s: {genre}/{path.name}")

        for number in range(samples_per_genre):
            numeric_id = f"{number:05d}"
            if numeric_id not in found_ids and f"{genre}{numeric_id}" not in excluded_audios:
                errors.append(f"Missing file: {genre}/{genre}.{numeric_id}.wav")

    return errors


def validate_pngs():
    errors = []
    if not png_dir.is_dir():
        return [f"Folder not found: {png_dir}"]

    found = Counter()
    for genre in genres:
        genre_dir = png_dir / genre
        if not genre_dir.is_dir():
            errors.append(f"Missing folder: {genre}")
            continue

        for path in sorted(genre_dir.glob("*.png")):
            match = png_pattern.match(path.name)
            if not match or match.group(1) != genre:
                errors.append(f"Invalid filename: {genre}/{path.name}")
                continue

            key = path.stem
            found[key] += 1
            if key not in expected_fragments:
                errors.append(f"Unexpected file: {genre}/{path.name}")
                continue

            try:
                with Image.open(path) as image:
                    size = image.size
            except Exception as error:
                errors.append(f"Unreadable image: {genre}/{path.name}: {error}")
                continue

            if size != image_size:
                errors.append(f"Unexpected size {size}: {genre}/{path.name}")

    for key, genre in expected_fragments.items():
        if key not in found:
            errors.append(f"Missing file: {genre}/{key}.png")

    return errors


def validate_csv():
    errors = []
    if not csv_path.is_file():
        return [f"File not found: {csv_path}"]

    found = Counter()
    rows_checked = 0

    with open(csv_path, newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)
        if header != columns:
            missing_columns = [name for name in columns if name not in (header or [])]
            extra_columns = [name for name in (header or []) if name not in columns]
            errors.append(f"Header mismatch. Missing columns: {missing_columns}. Extra columns: {extra_columns}")

        for line_number, row in enumerate(reader, start=2):
            rows_checked += 1
            if len(row) != len(columns):
                errors.append(f"Line {line_number}: expected {len(columns)} fields, found {len(row)}")
                continue

            record = dict(zip(columns, row))
            filename = record["filename"]
            found[filename] += 1

            match = csv_name_pattern.match(filename)
            if not match:
                errors.append(f"Line {line_number}: invalid filename '{filename}'")
            else:
                key = filename[:-4]
                if key not in expected_fragments:
                    errors.append(f"Line {line_number}: unexpected record '{filename}'")
                if match.group(1) != record["label"]:
                    errors.append(f"Line {line_number}: label '{record['label']}' does not match '{filename}'")

            for name in columns:
                if name in ("filename", "label"):
                    continue
                try:
                    value = float(record[name])
                except ValueError:
                    errors.append(f"Line {line_number}: non-numeric value in '{name}' for '{filename}'")
                    continue
                if not math.isfinite(value):
                    errors.append(f"Line {line_number}: non-finite value in '{name}' for '{filename}'")

            try:
                if int(float(record["length"])) != expected_length:
                    errors.append(f"Line {line_number}: length {record['length']} differs from {expected_length}")
            except ValueError:
                pass

    for filename, count in found.items():
        if count > 1:
            errors.append(f"Duplicated record: {filename} x{count}")

    found_keys = {name[:-4] for name in found}
    for key, genre in expected_fragments.items():
        if key not in found_keys:
            errors.append(f"Missing record: {key}.wav")

    return errors


checks = [
    ("genres_original", "Music samples", validate_wavs),
    ("mel_spectrograms", "3 Seconds music spectrogram", validate_pngs),
    ("features_3_sec.csv", "3 Seconds music features", validate_csv),
]

results = [(name, description, check()) for name, description, check in checks]

print("Path Data Validation")
for name, description, errors in results:
    print(f"{name} | {description} | {'OK' if not errors else 'FAIL'}")

for name, description, errors in results:
    if not errors:
        continue
    print()
    print(f"{name}: {len(errors)} problems")
    for error in errors[:max_errors_shown]:
        print(f"  {error}")
    if len(errors) > max_errors_shown:
        print(f"  ... and {len(errors) - max_errors_shown} more")

if any(errors for _, _, errors in results):
    sys.exit(1)