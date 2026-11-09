import csv
import subprocess
import sys
from pathlib import Path

GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
INPUT_DIR = Path("test/jazz")
OUTPUT_CSV = Path("results.csv")

MODELS = {
    "random_forest": ["uv", "run", "random_forest/random_forest_genres_classifier.py"],
    "cnn": ["uv", "run", "cnn/cnn_genres_classifier.py"],
}


def parse_output(output):
    scores = {}
    for line in output.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        name, value = line.rsplit(":", 1)
        name = name.strip().lower()
        if name in GENRES:
            try:
                scores[name] = float(value.strip())
            except ValueError:
                pass
    return scores


def run_model(command, audio_file):
    result = subprocess.run(
        command + [str(audio_file)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error running {' '.join(command)} on {audio_file}:\n{result.stderr}", file=sys.stderr)
        return None
    scores = parse_output(result.stdout)
    if len(scores) != len(GENRES):
        print(f"Incomplete output for {audio_file}: {scores}", file=sys.stderr)
        return None
    return scores


def main():
    audio_files = sorted(
        p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "model"] + GENRES)

        for audio_file in audio_files:
            for model_name, command in MODELS.items():
                print(f"Processing {audio_file} with {model_name}")
                scores = run_model(command, audio_file)
                if scores is None:
                    continue
                row = [audio_file.as_posix(), model_name] + [f"{scores[g]:.4f}" for g in GENRES]
                writer.writerow(row)
                f.flush()


if __name__ == "__main__":
    main()