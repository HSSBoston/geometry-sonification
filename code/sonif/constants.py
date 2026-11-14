import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).resolve().parents[1]  # 2 levels up
sys.path.append(str(PRJ_DIR))

from music21.chord import Chord

DATA_SONIF_DIR = PRJ_DIR / "data" / "sonif-input"

IMAGE_DATES = ("20240501", "20240801", "20241101", "20250201")

# Diatonic triads in C major.
CHORDS = {
    "I":   Chord("C E G"),
    "II":  Chord("D F A"),
    "IV":  Chord("F A C"),
    "V":   Chord("G B D"),
    "VI":  Chord("A C E"),
    "VII": Chord("B D F") }



