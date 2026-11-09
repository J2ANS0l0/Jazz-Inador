"""Live jazz filter for a shared YouTube Music playlist.

People at the club add songs to the playlist. This script checks the playlist every few seconds; for each new song it
downloads the audio as mp3, classifies its genre with the CNN (models/cnn.json) and decides:
    - the most likely genre is jazz  -> keep it in the playlist
    - any other genre                -> remove it from the playlist
Every decision is appended to playlist_filter/log.txt.

Setup (once): uv sync --group playlist, then create playlist_filter/browser.json (see --help / README).
Run:          uv run --group playlist playlist_filter/playlist_filter.py
"""

import argparse
import importlib.util
import json
import logging
import time
from pathlib import Path

import imageio_ffmpeg
import yt_dlp
from ytmusicapi import YTMusic

FILTER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILTER_DIR.parent
CLASSIFIER_PATH = PROJECT_ROOT / "cnn" / "cnn_genres_classifier.py"

PLAYLIST_ID = "PLR2w2VNjsTrs"  # "JAM"
AUTH_PATH = FILTER_DIR / "browser.json"
STATE_PATH = FILTER_DIR / "state.json"
LOG_PATH = FILTER_DIR / "log.txt"
DOWNLOAD_DIR = FILTER_DIR / "downloads"
TARGET_GENRE = "jazz"
POLL_SECONDS = 30
MAX_ANALYZED_SECONDS = 600  # classify at most the first 10 minutes of a song
MAX_ATTEMPTS = 3  # give up on a song (and leave it in the playlist) after this many failed downloads/classifications

log = logging.getLogger("playlist_filter")


def setup_logging():
    log.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        log.addHandler(handler)


def load_classifier():
    spec = importlib.util.spec_from_file_location("cnn_genres_classifier", CLASSIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_state():
    # setVideoId identifies one entry in the playlist (the same song added twice gets two setVideoIds).
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"decided": {}, "attempts": {}}


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def entry_key(track):
    # setVideoId is only returned when logged in as the playlist owner; dry runs fall back to the videoId.
    return track.get("setVideoId") or f"video:{track.get('videoId')}"


def describe(track):
    artists = ", ".join(artist["name"] for artist in track.get("artists") or [])
    return f"\"{track.get('title')}\" by {artists or 'unknown artist'} ({track.get('videoId')})"


def download_mp3(video_id):
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(DOWNLOAD_DIR / "%(id)s.%(ext)s"),
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "0"}],
        "quiet": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.download([f"https://music.youtube.com/watch?v={video_id}"])
    path = DOWNLOAD_DIR / f"{video_id}.mp3"
    if not path.is_file():
        raise FileNotFoundError(f"yt-dlp did not produce {path.name}")
    return path


def classify(classifier, mp3_path, model_path):
    spectrograms = classifier.audio_to_spectrograms(mp3_path, max_seconds=MAX_ANALYZED_SECONDS)
    return classifier.cnn_genre_classifier(spectrograms, model_path), len(spectrograms)


def process_track(yt, track, classifier, args, state):
    entry = entry_key(track)
    name = describe(track)
    mp3_path = None
    try:
        started = time.perf_counter()
        mp3_path = download_mp3(track["videoId"])
        genres, segments = classify(classifier, mp3_path, args.model)
    except Exception as error:
        attempts = state["attempts"].get(entry, 0) + 1
        state["attempts"][entry] = attempts
        if attempts >= MAX_ATTEMPTS:
            state["decided"][entry] = {"track": name, "decision": "error", "error": str(error)}
            log.error(f"GAVE UP after {attempts} attempts, left in playlist: {name} | {error}")
        else:
            log.warning(f"Attempt {attempts}/{MAX_ATTEMPTS} failed, will retry: {name} | {error}")
        return
    finally:
        if mp3_path is not None and mp3_path.exists() and not args.keep_audio:
            mp3_path.unlink()

    top_genre, top_probability = next(iter(genres.items()))
    top3 = ", ".join(f"{genre} {probability:.0%}" for genre, probability in list(genres.items())[:3])
    seconds = time.perf_counter() - started
    keep = top_genre == TARGET_GENRE

    if keep:
        decision = "keep"
        log.info(f"KEEP   {name} | {top3} | {segments} segments | {seconds:.0f}s")
    elif args.dry_run:
        decision = "would remove (dry run)"
        log.info(f"DRY-RUN would remove {name} | {top3} | {segments} segments | {seconds:.0f}s")
    elif not track.get("setVideoId"):
        decision = "remove failed: no setVideoId (is browser.json logged in as the playlist owner?)"
        log.error(f"REMOVE {name} | {top3} | cannot remove: YouTube did not return a setVideoId for this entry")
    else:
        status = yt.remove_playlist_items(args.playlist, [track])
        decision = "removed" if "SUCCEEDED" in str(status) else f"remove failed: {status}"
        level = logging.INFO if decision == "removed" else logging.ERROR
        log.log(level, f"REMOVE {name} | {top3} | {segments} segments | {seconds:.0f}s | YouTube: {status}")

    state["decided"][entry] = {
        "track": name,
        "decision": decision,
        "genre": top_genre,
        "probability": round(top_probability, 4),
        "jazz_probability": round(genres.get(TARGET_GENRE, 0.0), 4),
    }
    state["attempts"].pop(entry, None)


def check_playlist(yt, classifier, args, state):
    playlist = yt.get_playlist(args.playlist, limit=None)
    new_tracks = [t for t in playlist["tracks"] if entry_key(t) not in state["decided"]]
    if new_tracks:
        log.info(f"{len(new_tracks)} new song(s) in \"{playlist.get('title')}\" ({len(playlist['tracks'])} in total)")
    for track in new_tracks:
        if track.get("videoId"):
            process_track(yt, track, classifier, args, state)
        else:
            log.warning(f"Skipping unavailable track (no videoId): {track.get('title')}")
            state["decided"][entry_key(track)] = {"track": track.get("title"), "decision": "unavailable"}
        save_state(state)


def main():
    parser = argparse.ArgumentParser(description="Keep only jazz in a shared YouTube Music playlist.")
    parser.add_argument("--playlist", default=PLAYLIST_ID, help=f"Playlist id (default: {PLAYLIST_ID}, \"JAM\")")
    parser.add_argument("--auth", type=Path, default=AUTH_PATH, help="ytmusicapi browser.json of the playlist owner")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "cnn.json", help="CNN model JSON")
    parser.add_argument("--interval", type=float, default=POLL_SECONDS, help="Seconds between playlist checks")
    parser.add_argument("--once", action="store_true", help="Check the playlist once and exit")
    parser.add_argument("--dry-run", action="store_true", help="Classify and log, but never remove anything (no login needed)")
    parser.add_argument("--keep-audio", action="store_true", help="Keep the downloaded mp3 files")
    args = parser.parse_args()

    setup_logging()
    if args.auth.is_file():
        yt = YTMusic(str(args.auth))
    elif args.dry_run:
        yt = YTMusic()  # a public or unlisted playlist can be read without logging in
    else:
        parser.error(f"{args.auth} not found. Create it with: uv run --group playlist ytmusicapi browser --file {args.auth}")
    classifier = load_classifier()
    state = load_state()

    log.info(f"Started | playlist {args.playlist} | model {args.model.name} | every {args.interval:.0f}s"
             f"{' | DRY RUN' if args.dry_run else ''}")
    try:
        while True:
            try:
                check_playlist(yt, classifier, args, state)
            except Exception as error:  # network hiccups, YouTube changes: log and keep going
                log.error(f"Playlist check failed: {type(error).__name__}: {error}")
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        save_state(state)
        log.info("Stopped")


if __name__ == "__main__":
    main()
