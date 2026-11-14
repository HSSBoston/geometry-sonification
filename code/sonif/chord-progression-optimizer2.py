import csv, numpy as np
from itertools import product

from constants import IMAGE_DATES, DATA_SONIF_DIR
from ndvi import calculateNdviDistances
from orbifold import calculateOrbifoldDistances

states = ("I", "II", "IV", "V", "VI", "VII")
stateIndices = {state: index for index, state in enumerate(states)}

firstChord = "I"

halfCadences = (
    ("I", "V"),
    ("IV", "V"),
    ("II", "V") )

cadences = (
    ("IV", "V", "I"),
    ("II", "V", "I") )
#     ("IV", "I"),
#     ("IV","V"),
#     ("II", "V"),
#     ("V", "VI") )

wGeometry    = 1.0
wHarmony     = 0.5

# Markov state transition probabilities
P = np.array(
    [[0.02, 0.245, 0.245, 0.245, 0.245, 0],
     [0.2,  0,     0,     0.3,   0.3,   0.2],
     [0.2,  0.2,   0,     0.2,   0.2,   0.2],
     [0.45, 0,     0,     0,     0.45,  0.1],
     [0,    0.3,   0.3,   0.3,   0,     0.1],
     [1,    0,     0,     0,     0,     0]] )

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

# Detect extended A-B-A-B alternation patterns in the chord progression.
# A positive value indicates that the progression violates the alternation constraint.
#
def calculateAlternationError(progression):
    numAlternations = 0

    for i in range(len(progression) - 3):
        if (progression[i] == progression[i + 2] and
            progression[i + 1] == progression[i + 3] and
            progression[i] != progression[i + 1]):
            numAlternations += 1

    return numAlternations / max(len(progression) - 3, 1)


# Find the chord event that ends measure 2, using region labels from the input CSV.
# Region 4 may be represented by one event (4) or split into two (4a, 4b).
# The half cadence occupies this event and the preceding event.
#
def findHalfCadenceEndIndex(imageDate, numChordEvents):
    inputCsv = f"{imageDate}-south-regions-chords.csv"
    with open(DATA_SONIF_DIR / inputCsv, newline="") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != numChordEvents:
        raise ValueError("Number of chord events does not match region data.")

    indices = [i for i, row in enumerate(rows)
               if row["regionLabel"] in ("4", "4b")]

    if len(indices) != 1:
        raise ValueError("Expected exactly one final region of measure 2.")

    endIndex = indices[0]
    if endIndex == 0:
        raise ValueError("Half cadence requires a preceding chord event.")

    endBeat = float(rows[endIndex]["startBeat"]) + float(rows[endIndex]["durationBeats"])
    if not np.isclose(endBeat, 8.0):
        raise ValueError("Half cadence must end at beat 8.")

    return endIndex

def optimizeProgression(ndviDistances, orbifoldDistances, transitionProbabilities, halfCadenceEndIndex):
    numChordEvents = len(ndviDistances) + 1

    # For each chord state transition, precompute the squared geometry error for
    # every possible pair of chord states
    #   Shape: (numChordEvents-1, 6, 6)
    transitionErrors = (
        ndviDistances[:, np.newaxis, np.newaxis] - orbifoldDistances[np.newaxis, :, :]) ** 2

    # Precompute harmony errors for all chord state transitions.
    # Zero-probability transitions are prohibited.
    #   Shape: (6,6)
    validTransitions = transitionProbabilities > 0
    harmonyErrors = 1.0 - transitionProbabilities

    bestProgression   = None
    bestCadence       = None
    bestGeometryError = float("inf")
    bestHarmonyError  = float("inf")
    bestTotalError    = float("inf")

    count = 0
    markovValidCount = 0
    validCount = 0

    # Enumerate all progressions satisfying the fixed opening and a selected cadence
    for cadence in cadences:
        numFreeChords = numChordEvents - 1 - len(cadence)

        if numFreeChords < 0:
            raise ValueError("Not enough chord events for the selected cadence.")

        for middleChords in product(states, repeat=numFreeChords):
            progression = (firstChord,) + middleChords + cadence
            count += 1

            stateSequence = [stateIndices[state] for state in progression]

            # Eliminate progressions containing zero-probability transitions
            if any(not validTransitions[stateSequence[i], stateSequence[i + 1]]
                   for i in range(len(stateSequence) - 1)):
                continue

            markovValidCount += 1

            # Require a half cadence at the end of measure 2
            halfCadence = progression[halfCadenceEndIndex - 1:halfCadenceEndIndex + 1]
            if halfCadence not in halfCadences:
                continue

            # Eliminate progressions containing extended A-B-A-B alternation patterns
            if calculateAlternationError(progression) > 0:
                continue

            validCount += 1

            eGeometry = calculateGeometryError(progression, transitionErrors)
            eHarmony = calculateHarmonyError(progression, harmonyErrors)

            eTotal = wGeometry * eGeometry + wHarmony * eHarmony

            if eTotal < bestTotalError:
                bestTotalError    = eTotal
                bestGeometryError = eGeometry
                bestHarmonyError  = eHarmony
                bestProgression   = progression
                bestCadence       = cadence

    if bestProgression is None: raise ValueError("No valid chord progressions found.")

    return (bestProgression, bestCadence, bestGeometryError, bestHarmonyError,
            bestTotalError, count, markovValidCount, validCount)


# Compute the orbifold distance matrix
normalizedOrbifoldDistances = calculateOrbifoldDistances(states)
# print("Normalized orbifold distance matrix:")
# print(np.round(normalizedOrbifoldDistances, 3))

print("wGeometry:", wGeometry)
print("wHarmony:",  wHarmony)

# Optimize chord progressions for each date
for imageDate in IMAGE_DATES:
    print(f"\nProcessing {imageDate}")

    normalizedNdviDistances = calculateNdviDistances(imageDate)
    print("\nNormalized NDVI spatial-gradient distances:")
    print(np.round(normalizedNdviDistances, 4))

    halfCadenceEndIndex = findHalfCadenceEndIndex(
        imageDate, len(normalizedNdviDistances) + 1 )

    (bestProgression, bestCadence, bestGeometryError, bestHarmonyError,
     bestTotalError, count, markovValidCount, validCount) = optimizeProgression(
        normalizedNdviDistances,
        normalizedOrbifoldDistances,
        P,
        halfCadenceEndIndex )

    print("\nTotal progressions generated:", count)
    print("Markov-valid progressions:", markovValidCount)
    print("Valid progressions after alternation filtering:", validCount)

    print("Best chord progression:", bestProgression)
    print("Selected half cadence:", bestProgression[halfCadenceEndIndex - 1:halfCadenceEndIndex + 1])
    print("Selected cadence:", bestCadence)
    print("E_geometry:", bestGeometryError)
    print("E_harmony:", bestHarmonyError)
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
