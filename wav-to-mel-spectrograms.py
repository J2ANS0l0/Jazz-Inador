from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

project_dir = Path(__file__).parent
input_dir = project_dir / "Data/genres_original"
output_dir = project_dir / "Data/mel-spectrograms"
segment_seconds = 3
segment_count = 10

def main():
    mel_spectrograms(input_dir, output_dir, segment_seconds, segment_count)
    
def mel_spectrograms(input_dir: Path, output_dir: Path, segment_seconds: int, segment_count: int):

    for audio_path in sorted(input_dir.rglob("*.wav")):
        genre = audio_path.parent.name
        numeric_id = audio_path.stem.split(".")[-1]
        genre_dir = output_dir / genre
        genre_dir.mkdir(parents=True, exist_ok=True)

        try:
            y, sr = librosa.load(audio_path, sr=None)
        except Exception as error:
            print(f"Skipped {audio_path.name}: {error}")
            continue

        segment_length = segment_seconds * sr
        y = librosa.util.fix_length(y, size=segment_count * segment_length)

        for fragment in range(segment_count):
            segment = y[fragment * segment_length:(fragment + 1) * segment_length]

            mel_spectrogram = librosa.feature.melspectrogram(
                y=segment,
                sr=sr,
                n_fft=8192,
                hop_length=128,
                window="hamming",
                n_mels=192,
                fmin=0.0,
                fmax=None,
                htk=False,
                norm="slaney",
                power=2.0,
            )
            mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram, ref=np.max, top_db=75)

            out_path = genre_dir / f"{genre}{numeric_id}{fragment:03d}.png"

            fig = plt.figure(figsize=(4.32, 2.88), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1])
            librosa.display.specshow(mel_spectrogram_db, sr=sr, cmap="magma", ax=ax)
            ax.axis("off")
            fig.savefig(out_path, dpi=100)
            plt.close(fig)
#            print(f"Saved {out_path}")

if __name__ == "__main__":
    main()