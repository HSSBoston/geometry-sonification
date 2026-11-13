import numpy as np
from itertools import product

from constants import IMAGE_DATES
from ndvi import calculateNdviDistances
from orbifold import calculateOrbifoldDistances

states = ("I", "II", "IV", "V", "VI", "VII")
stateIndices = {state: index for index, state in enumerate(states)}

firstChord = "I"
cadence = ("V", "I")

wGeometry    = 1.0
wHarmony     = 1.0
wAlternation = 0.05

# Markov state transition probabilities
P = np.array(
    [[0,    0.25, 0.25, 0.25, 0.25, 0],
     [0.2,  0,    0,    0.3,  0.3,  0.2],
     [0.2,  0.2,  0,    0.2,  0.2,  0.2],
     [0.45, 0,    0,    0,    0.45, 0.1],
     [0,    0.3,  0.3,  0.3,  0,    0.1],
     [1,    0,    0,    0,    0,    0]] )

assert P.shape == (len(states), len(states))
assert np.all((P >= 0) & (P <= 1))
assert np.allclose(np.sum(P, axis=1), 1.0)


# Calculate E_geometry.
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


# Calculate E_harmony.
#   progression: list of chords expressed as roman numerals; e.g., ["I", "IV", "V"]
#   harmonyErrors[a, b]: harmonic error for transitioning from chord state a to
#     chord state b, defined as 1 - P[a, b]
#
def calculateHarmonyError(progression, harmonyErrors):
    stateSequence = [stateIndices[state] for state in progression]

    totalError = 0.0

    for i in range(len(stateSequence) - 1):
        stateA = stateSequence[i]
        stateB = stateSequence[i + 1]
        totalError += harmonyErrors[stateA, stateB]

    return totalError / (len(stateSequence) - 1)

# Calculate E_alternation. Penalize extended A-B-A-B alteration patterns in the chord progression.
#   progression: list of chords expressed as roman numerals; e.g., ["I", "IV", "V"]
#
def calculateAlternationError(progression):
    numAlternations = 0

    for i in range(len(progression) - 3):
        if (progression[i] == progression[i + 2] and
            progression[i + 1] == progression[i + 3] and
            progression[i] != progression[i + 1]):
            numAlternations += 1

    return numAlternations / max(len(progression) - 3, 1)

def optimizeProgression(ndviDistances, orbifoldDistances, transitionProbabilities):
    numChordEvents = len(ndviDistances) + 1
    numFreeChords = numChordEvents - 1 - len(cadence)

    if numFreeChords < 0:
        raise ValueError("Not enough chord events for chord progression generation.")

    # For each chord state transition, precompute the squared geometry error for
    # every possible pair of chord states
    #   Shape: (numChordEvents-1, 6, 6)
    transitionErrors = (
        ndviDistances[:, np.newaxis, np.newaxis] - orbifoldDistances[np.newaxis, :, :]) ** 2
        # ndviDistances: shape (10,)
        # ndviDistances[:, np.newaxis, np.newaxis]: shape (10,1,1)
        # orbifoldDistances: shape (6,6)
        # orbifoldDistances[np.newaxis, :, :]: shape (1,6,6)
        # transitionErrors: shape (10,6,6). A stack of 10 6x6 matrices.

    # Precompute harmony errors for all chord state transitions
    #   Shape: (6,6)
    harmonyErrors = 1.0 - transitionProbabilities

    bestProgression      = None
    bestGeometryError    = float("inf")
    bestHarmonyError     = float("inf")
    bestAlternationError = float("inf")
    bestTotalError       = float("inf")
    count = 0

    # Enumerate all progressions satisfying the fixed opening and ending cadence
    for middleChords in product(states, repeat=numFreeChords):
        progression = (firstChord,) + middleChords + cadence

        eGeometry = calculateGeometryError(progression, transitionErrors)
        eHarmony = calculateHarmonyError(progression, harmonyErrors)
        eAlternation = calculateAlternationError(progression)

#        eTotal = wGeometry * eGeometry + wHarmony * eHarmony
        eTotal = wGeometry * eGeometry + wHarmony * eHarmony + wAlternation * eAlternation

        if eTotal < bestTotalError:
            bestTotalError    = eTotal
            bestGeometryError = eGeometry
            bestHarmonyError  = eHarmony
            bestAlternationError = eAlternation
            bestProgression   = progression

        count += 1

    return bestProgression, bestGeometryError, bestHarmonyError, bestTotalError, count


# Compute the orbifold distance matrix
normalizedOrbifoldDistances = calculateOrbifoldDistances(states)
print("Normalized orbifold distance matrix:")
print(np.round(normalizedOrbifoldDistances, 3))

print("\nMarkov state transition matrix:")
print(P)

print("\nOptimization weights:")
print("wGeometry:", wGeometry)
print("wHarmony:",  wHarmony)
print("wAlternation:",  wAlternation)

# Optimize chord progressions for each date
for imageDate in IMAGE_DATES:
    print(f"\nProcessing {imageDate}")

    normalizedNdviDistances = calculateNdviDistances(imageDate)
    print("\nNormalized NDVI spatial-gradient distances:")
    print(np.round(normalizedNdviDistances, 4))

    bestProgression, bestGeometryError, bestHarmonyError, bestTotalError, count = optimizeProgression(
        normalizedNdviDistances,
        normalizedOrbifoldDistances,
        P )

    print("\nTotal progressions evaluated:", count)
    print("Best chord progression:", bestProgression)
    print("E_geometry:", bestGeometryError)
    print("E_harmony:", bestHarmonyError)
    print("E_alternation:", bestAlternationError)
    print("E_total:", bestTotalError)

    # Display realized orbifold distances of the best progression
    realizedDistances = np.array([
        normalizedOrbifoldDistances[stateIndices[bestProgression[i]],
                                    stateIndices[bestProgression[i + 1]]]
        for i in range(len(bestProgression) - 1) ])

    print("\nRealized orbifold distances:")
    print(np.round(realizedDistances, 4))

    squaredErrors = (normalizedNdviDistances - realizedDistances) ** 2
    print("\nGeometry errors per transition:")
    print(np.round(squaredErrors, 4))
    print("Mean geometry error:", np.mean(squaredErrors))

    # Display Markov transition probabilities of the best progression
    realizedProbabilities = np.array([
        P[stateIndices[bestProgression[i]],
          stateIndices[bestProgression[i + 1]]]
        for i in range(len(bestProgression) - 1) ])

    print("\nMarkov probabilities per transition:")
    print(np.round(realizedProbabilities, 4))

    harmonyErrors = 1.0 - realizedProbabilities
    print("\nHarmony errors per transition:")
    print(np.round(harmonyErrors, 4))
    print("Mean harmony error:", np.mean(harmonyErrors))
