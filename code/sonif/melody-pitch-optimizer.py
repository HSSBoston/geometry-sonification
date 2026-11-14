import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).resolve().parents[1]  # 2 levels up
sys.path.append(str(PRJ_DIR))

import csv
from music21.pitch import Pitch
from constants import SONIF_DIR, IMAGE_DATES, CHORDS

# Chord progressions for the four dates.
progression20240501 = ("I", "I", "IV", "I", "V", "I", "VI", "IV", "II", "V", "I")
progression20240801 = ("I", "I", "IV", "I", "V", "I", "VI", "IV", "II", "V", "I")
progression20241101 = ("I", "I", "I", "I", "V", "I", "VI", "IV", "II", "V", "I")
progression20250201 = ("I", "I", "IV", "I", "V", "I", "VI", "IV", "II", "V", "I")

chordProgressions = {
    "20240501": progression20240501,
    "20240801": progression20240801,
    "20241101": progression20241101,
    "20250201": progression20250201 }

minPitch   = Pitch("C3").midi
maxPitch   = Pitch("B5").midi
firstPitch = Pitch("C4").midi
lastPitch  = Pitch("C4").midi

def pitchName(midi):
    pitch = Pitch()
    pitch.midi = midi
    return pitch.nameWithOctave

# Load the previously generated melody rhythm.
#
def loadMelodyEvents(imageDate):
    inputCsv = f"{imageDate}-north-melody-rhythm.csv"

    with open(SONIF_DIR / inputCsv, newline="") as file:
        rows = list(csv.DictReader(file))

    melodyEvents = []
    for index, row in enumerate(rows):
        event = {
            "eventIndex":      int(  row["eventIndex"]),
            "chordEventIndex": int(  row["chordEventIndex"]),
            "startBeat":       float(row["startBeat"]),
            "durationBeats":   float(row["durationBeats"]),
            "landFraction":    float(row["landFraction"]),
            "landGreenness":   float(row["landGreenness"]) }

        assert event["eventIndex"] == index + 1
        melodyEvents.append(event)

    assert melodyEvents
    assert melodyEvents[0]["startBeat"] == 0.0

    totalBeats = sum(event["durationBeats"] for event in melodyEvents)
    assert abs(totalBeats - 16.0) < 1e-9

    return melodyEvents


# Generate all chord-tone candidate pitches between C3 and B5.
#
def generateCandidatePitches(chordState):
    chordPitchClasses = { pitch.pitchClass for pitch in CHORDS[chordState].pitches }

    candidatePitches = []
    for midi in range(minPitch, maxPitch + 1):
        if midi % 12 in chordPitchClasses:
            candidatePitches.append(midi)
    return candidatePitches

# Calculate the pitch-selection objective.
# To be implemented later. Potential components:
#   E_register
#   E_contour
#   E_movement
#
def calculatePitchError(candidatePitch, melodyEvent, previousPitch):
    # TODO: Implement pitch-selection objective.
    return None


# Evaluate every candidate and retain the lowest-error pitch.
#
def optimizePitch(candidatePitches, melodyEvent, previousPitch):
    bestPitch = None
    bestError = float("inf")

    for candidatePitch in candidatePitches:
        ePitch = calculatePitchError( candidatePitch, melodyEvent, previousPitch )

        # No objective has been implemented yet.
        if ePitch is None:
            continue

        if ePitch < bestError:
            bestError = ePitch
            bestPitch = candidatePitch

    return bestPitch, bestError


# Generate the melody pitches for one date.
#
def generateMelodyPitches(melodyEvents, chordProgression):
    selectedPitches = []
    numEvents = len(melodyEvents)

    for index, event in enumerate(melodyEvents):
        chordIndex = event["chordEventIndex"] - 1

        if not (0 <= chordIndex < len(chordProgression)):
            raise ValueError("Invalid chordEventIndex.")

        chordState = chordProgression[chordIndex]
        candidatePitches = generateCandidatePitches(chordState)

        # Enforce fixed first and last notes.
        if index == 0:
            assert firstPitch in candidatePitches
            candidatePitches = [firstPitch]
        elif index == numEvents - 1:
            assert lastPitch in candidatePitches
            candidatePitches = [lastPitch]

        # Select a pitch.
        if index == 0:
            bestPitch = firstPitch
            bestError = None
        elif index == numEvents - 1:
            bestPitch = lastPitch
            bestError = None
        else:
            previousPitch = selectedPitches[-1]
            bestPitch, bestError = optimizePitch( candidatePitches, event, previousPitch )
            
        selectedPitches.append(bestPitch)

        print(
            f"  event={event['eventIndex']:>2} "
            f"chord={chordState:>3} "
            f"start={event['startBeat']:>4.1f} "
            f"duration={event['durationBeats']:>3.1f} "
            f"greenness={event['landGreenness']:.3f}" )
        
        print("    Candidates:", [pitchName(p) for p in candidatePitches] )

        if bestPitch is not None:
            print("    Selected:", pitchName(bestPitch))
        else:
            print("    Selected: pending objective function")

    return selectedPitches


# Generate melodies for all four dates.
#
for imageDate in IMAGE_DATES:
    print(f"\nProcessing {imageDate}")

    melodyEvents     = loadMelodyEvents(imageDate)
    chordProgression = chordProgressions[imageDate]
    selectedPitches = generateMelodyPitches( melodyEvents, chordProgression )

    print("Total melody events:", len(melodyEvents))
    print("Selected pitches:", [pitchName(p) if p is not None else None
                                for p in selectedPitches] )
