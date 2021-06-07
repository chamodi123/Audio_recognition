import os
import numpy as np
from multiprocessing import Pool
from tinytag import TinyTag
from record import record_audio
from fingerprint import fingerprint_file, fingerprint_audio
from storage import store_song, get_matches, get_info_for_song_id, song_in_db
from settings import NUM_WORKERS

KNOWN_EXTENSIONS = ["mp3", "wav", "flac", "m4a"]


def get_song_info(filename):
    """Gets the ID3 tags for a file. Returns None for tuple values that don't exist.
    """
    tag = TinyTag.get(filename)
    print(filename)
    #print(str(tag))
    return (filename) #tag.title


def register_song(filename):
    """Register a single song."""
    if song_in_db(filename):
        return
    hashes = fingerprint_file(filename)
    song_info = get_song_info(filename)
    print(song_info)
    store_song(hashes, song_info)


def register_directory(path):
   
    to_register = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.split('.')[-1] not in KNOWN_EXTENSIONS:
                continue
            file_path = os.path.join(path, root, f)
            to_register.append(file_path)
    with Pool(NUM_WORKERS) as p:
        p.map(register_song, to_register)


def score_match(offsets):
    """Score a matched song.
    """
    tks = list(map(lambda x: x[0] - x[1], offsets))
    hist, _ = np.histogram(tks)
    return np.max(hist)


def best_match(matches):
    """For a dictionary of song_id: offsets, returns the best song_id.

    """
    matched_song = None
    best_score = 0
    for song_id, offsets in matches.items():
        if len(offsets) < best_score:
            # can't be best score, avoid expensive histogram
            continue
        score = score_match(offsets)
        if score > best_score:
            best_score = score
            matched_song = song_id
    return matched_song


def recognise_song(filename):
    """Recognises a pre-recorded sample.

   
    """
    hashes = fingerprint_file(filename)
    matches = get_matches(hashes)
    matched_song = best_match(matches)
    info = get_info_for_song_id(matched_song)
    if info is not None:
        return info
    return matched_song


def listen_to_song(filename=None):
    """Recognises a song using the microphone.

    """
    audio = record_audio(filename=filename)
    hashes = fingerprint_audio(audio)
    matches = get_matches(hashes)
    matched_song = best_match(matches)
    info = get_info_for_song_id(matched_song)
    if info is not None:
        return info
    return matched_song
