"""Live jazz filter for a shared YouTube Music playlist.

People at the club add songs to the playlist. This script checks the playlist every few seconds; for each new song it
downloads the audio, classifies the genre of CLIP_COUNT short clips spread across the song with the CNN
(models/cnn.json) and decides:
    - the most likely genre is jazz  -> keep it in the playlist
    - any other genre                -> remove it from the playlist
Several songs are processed in parallel, and the playlist keeps being checked while they are processed.
Every decision is appended to playlist_filter/log.txt.

Setup (once): uv sync --group playlist, then create playlist_filter/browser.json (see --help / README).
Run:          uv run --group playlist playlist_filter/playlist_filter.py
"""

import argparse
import importlib.util
import json
import logging
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import imageio_ffmpeg
import numpy as np
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
POLL_SECONDS = 10
WORKERS = 4  # songs downloaded/processed at the same time
# Classify CLIP_COUNT clips of CLIP_SECONDS spread evenly across the song instead of the whole song.
# On 18 test songs, 6 x 15 s gave the same keep/remove decision as classifying the whole song, ~10x faster.
CLIP_COUNT = 6
CLIP_SECONDS = 15
MAX_ATTEMPTS = 3  # give up on a song (and leave it in the playlist) after this many failed downloads/classifications
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

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


def download_audio(video_id):
    # Download the audio stream as is (webm/m4a, a few MB); no mp3 conversion, ffmpeg decodes only the clips we need.
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(DOWNLOAD_DIR / "%(id)s.%(ext)s"),
        "quiet": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(f"https://music.youtube.com/watch?v={video_id}", download=True)
    path = Path(info["requested_downloads"][0]["filepath"])
    if not path.is_file():
        raise FileNotFoundError(f"yt-dlp did not produce {path.name}")
    return path, info.get("duration")


def decode(path, start, seconds, sample_rate):
    # Decode [start, start + seconds) of the file to mono float32 samples at sample_rate.
    command = [FFMPEG, "-v", "error", "-ss", f"{start:.2f}", "-t", f"{seconds:.2f}", "-i", str(path),
               "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"]
    samples = np.frombuffer(subprocess.run(command, capture_output=True, check=True).stdout, dtype=np.float32)
    if samples.size == 0:
        raise ValueError(f"ffmpeg decoded no audio at {start:.0f}s")
    return samples


def clip_starts(duration):
    # CLIP_COUNT clips centered at 1/(n+1), 2/(n+1), ... of the song; one clip from the start if the song is short.
    if not duration or duration <= CLIP_COUNT * CLIP_SECONDS:
        return [0.0], CLIP_COUNT * CLIP_SECONDS
    starts = [duration * (i + 1) / (CLIP_COUNT + 1) - CLIP_SECONDS / 2 for i in range(CLIP_COUNT)]
    return [max(0.0, start) for start in starts], CLIP_SECONDS


def analyze(track, classifier, args, classify_lock):
    # Runs in a worker thread: download + decode + classify. Never touches the YouTube Music client or state.
    started = time.perf_counter()
    path = None
    try:
        path, duration = download_audio(track["videoId"])
        starts, seconds = clip_starts(duration)
        clips = [decode(path, start, seconds, classifier.SAMPLE_RATE) for start in starts]
    finally:
        if path is not None and path.exists() and not args.keep_audio:
            path.unlink()
    with classify_lock:  # matplotlib rendering isn't guaranteed thread safe; classifying takes ~2 s, downloads overlap
        spectrograms = np.concatenate([classifier.samples_to_spectrograms(clip) for clip in clips])
        genres = classifier.cnn_genre_classifier(spectrograms, args.model)
    return genres, len(spectrograms), time.perf_counter() - started


def record_failure(track, error, state):
    entry, name = entry_key(track), describe(track)
    attempts = state["attempts"].get(entry, 0) + 1
    state["attempts"][entry] = attempts
    if attempts >= MAX_ATTEMPTS:
        state["decided"][entry] = {"track": name, "decision": "error", "error": str(error)}
        log.error(f"GAVE UP after {attempts} attempts, left in playlist: {name} | {error}")
    else:
        log.warning(f"Attempt {attempts}/{MAX_ATTEMPTS} failed, will retry: {name} | {error}")


def apply_decision(yt, track, genres, segments, seconds, args, state):
    entry, name = entry_key(track), describe(track)
    top_genre, top_probability = next(iter(genres.items()))
    top3 = ", ".join(f"{genre} {probability:.0%}" for genre, probability in list(genres.items())[:3])
    details = f"{top3} | {segments} segments | {seconds:.0f}s"

    if top_genre == TARGET_GENRE:
        decision = "keep"
        log.info(f"KEEP   {name} | {details}")
    elif args.dry_run:
        decision = "would remove (dry run)"
        log.info(f"DRY-RUN would remove {name} | {details}")
    elif not track.get("setVideoId"):
        decision = "remove failed: no setVideoId (is browser.json logged in as the playlist owner?)"
        log.error(f"REMOVE {name} | {top3} | cannot remove: YouTube did not return a setVideoId for this entry")
    else:
        status = yt.remove_playlist_items(args.playlist, [track])
        decision = "removed" if "SUCCEEDED" in str(status) else f"remove failed: {status}"
        level = logging.INFO if decision == "removed" else logging.ERROR
        log.log(level, f"REMOVE {name} | {details} | YouTube: {status}")

    state["decided"][entry] = {
        "track": name,
        "decision": decision,
        "genre": top_genre,
        "probability": round(top_probability, 4),
        "jazz_probability": round(genres.get(TARGET_GENRE, 0.0), 4),
    }
    state["attempts"].pop(entry, None)


def submit_new_tracks(yt, pool, in_flight, classifier, args, state, classify_lock):
    playlist = yt.get_playlist(args.playlist, limit=None)
    new_tracks = [t for t in playlist["tracks"] if entry_key(t) not in state["decided"] and entry_key(t) not in in_flight]
    if new_tracks:
        log.info(f"{len(new_tracks)} new song(s) in \"{playlist.get('title')}\" ({len(playlist['tracks'])} in total)")
    for track in new_tracks:  # playlist order: the song that will play first is submitted first
        if track.get("videoId"):
            future = pool.submit(analyze, track, classifier, args, classify_lock)
            in_flight[entry_key(track)] = (track, future)
        else:
            log.warning(f"Skipping unavailable track (no videoId): {track.get('title')}")
            state["decided"][entry_key(track)] = {"track": track.get("title"), "decision": "unavailable"}
            save_state(state)


def handle_finished(yt, in_flight, done, args, state):
    for entry, (track, future) in list(in_flight.items()):
        if future not in done:
            continue
        del in_flight[entry]
        try:
            genres, segments, seconds = future.result()
            apply_decision(yt, track, genres, segments, seconds, args, state)
        except Exception as error:
            record_failure(track, error, state)
        save_state(state)


def main():
    parser = argparse.ArgumentParser(description="Keep only jazz in a shared YouTube Music playlist.")
    parser.add_argument("--playlist", default=PLAYLIST_ID, help=f"Playlist id (default: {PLAYLIST_ID}, \"JAM\")")
    parser.add_argument("--auth", type=Path, default=AUTH_PATH, help="ytmusicapi browser.json of the playlist owner")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "cnn.json", help="CNN model JSON")
    parser.add_argument("--interval", type=float, default=POLL_SECONDS, help="Seconds between playlist checks")
    parser.add_argument("--workers", type=int, default=WORKERS, help="Songs processed at the same time")
    parser.add_argument("--once", action="store_true", help="Check the playlist once, process its new songs and exit")
    parser.add_argument("--dry-run", action="store_true", help="Classify and log, but never remove anything (no login needed)")
    parser.add_argument("--keep-audio", action="store_true", help="Keep the downloaded audio files")
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
    classify_lock = threading.Lock()
    in_flight = {}  # entry key -> (track, future) of songs being processed

    log.info(f"Started | playlist {args.playlist} | model {args.model.name} | every {args.interval:.0f}s"
             f" | {args.workers} workers{' | DRY RUN' if args.dry_run else ''}")
    pool = ThreadPoolExecutor(max_workers=args.workers)
    try:
        next_check = 0.0
        while True:
            if time.monotonic() >= next_check and not (args.once and next_check):  # --once: a single check
                try:
                    submit_new_tracks(yt, pool, in_flight, classifier, args, state, classify_lock)
                except Exception as error:  # network hiccups, YouTube changes: log and keep going
                    log.error(f"Playlist check failed: {type(error).__name__}: {error}")
                next_check = time.monotonic() + args.interval
            if args.once and not in_flight:
                break
            # Handle songs as soon as they finish, but go back to checking the playlist every interval.
            timeout = None if args.once else max(0.0, next_check - time.monotonic())
            if in_flight:
                done, _ = wait([future for _, future in in_flight.values()], timeout=timeout, return_when=FIRST_COMPLETED)
                handle_finished(yt, in_flight, done, args, state)
            else:
                time.sleep(timeout)
    except KeyboardInterrupt:
        pass
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        save_state(state)
        log.info("Stopped")


if __name__ == "__main__":
    main()
