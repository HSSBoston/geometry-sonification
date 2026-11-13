import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).resolve().parents[1]  # 2 levels up
sys.path.append(str(PRJ_DIR))

import numpy as np
from itertools import product
from music21.chord import Chord
from music21.scale import ChromaticScale
from orbichord.chordinate import EfficientVoiceLeading, Permutation

from constants
from ndvi import calculateNdviDistances
from orbifold import calculateOrbifoldDistances

states = ("I", "II", "IV", "V", "VI", "VII")
stateIndices = {state: index for index, state in enumerate(states)}

firstChord = "I"
cadence = ("V", "I")

# Diatonic triads in C major.
chords = {
    "I":   Chord("C E G"),
    "II":  Chord("D F A"),
    "IV":  Chord("F A C"),
    "V":   Chord("G B D"),
    "VI":  Chord("A C E"),
    "VII": Chord("B D F") }

# Calculate E_geometory.
#   progression: list of chords expressed as roman numerals; e.g., ["I", "IV", "V"]
#   transitionErrors[i, a, b]: squared mismatch between the environmental target at
#     chord state transition i and the orbifold distance from chord state a to chord state b
#
def calculateGeometryError(progression, transitionErrors):
    stateSequence = [stateIndices[state] for state in progression]

    totalError = 0.0

    for i in range(len(stateSequence) - 1):
        stateA = stateSequence[i]
        stateB = stateSequence[i + 1]
        totalError += transitionErrors[i, stateA, stateB]

    return totalError / (len(stateSequence) - 1)


def optimizeProgression(ndviDistances, orbifoldDistances):
    numChordEvents = len(ndviDistances) + 1
    numFreeChords = numChordEvents - 1 - len(cadence)

    if numFreeChords < 0:
        raise ValueError("Not enough chord events for chord progression generation.")

    # For each chord state transition, precompute the squared geometry error for
    # every possible pair of chord states
    #   Shape: (numChordEvents-1, 6, 6)
    transitionErrors = (
        ndviDistances[:, np.newaxis, np.newaxis] - orbifoldDistances[np.newaxis, :, :]) ** 2

    bestProgression = None
    bestGeometryError = float("inf")
    count = 0

    # Enumerate all progressions satisfying the fixed opening and ending cadence
    for middleChords in product(states, repeat=numFreeChords):
        progression = (firstChord,) + middleChords + cadence

        eGeometry = calculateGeometryError(progression, transitionErrors)        
        if eGeometry < bestGeometryError:
            bestGeometryError = eGeometry
            bestProgression = progression
        
        count += 1
    return bestProgression, bestGeometryError, count


# Compute the orbifold distance matrix
normalizedOrbifoldDistances = calculateOrbifoldDistances(states)
print("Normalized orbifold distance matrix:")
print(np.round(normalizedOrbifoldDistances, 3))

# Optimize chord progressions for each date
for imageDate in constants.IMAGE_DATES:
    print(f"\nProcessing {imageDate}")

    normalizedNdviDistances = calculateNdviDistances(imageDate)
    print("\nNormalized NDVI spatial-gradient distances:")
    print(np.round(normalizedNdviDistances, 4))

    bestProgression, bestGeometryError, count = optimizeProgression(
        normalizedNdviDistances,
        normalizedOrbifoldDistances )

    print("\nTotal progressions evaluated:", count)
    print("Best chord progression:", bestProgression)
    print("Minimum E_geometry:", bestGeometryError)

    # Display realized orbifold distances of the best progression.
    realizedDistances = np.array([
        normalizedOrbifoldDistances[
            stateIndices[bestProgression[i]],
            stateIndices[bestProgression[i + 1]]
        ]
        for i in range(len(bestProgression) - 1)
    ])

    print("\nRealized orbifold distances:")
    print(np.round(realizedDistances, 4))

