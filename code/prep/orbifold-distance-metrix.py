import numpy as np
from music21.chord import Chord
from music21.scale import ChromaticScale
from orbichord.chordinate import EfficientVoiceLeading, Permutation

# Same six states as the Markov model.
states = ("I", "II", "IV", "V", "VI", "VII")

# Diatonic triads in C major.
chords = {
    "I":   Chord("C E G"),
    "II":  Chord("D F A"),
    "IV":  Chord("F A C"),
    "V":   Chord("G B D"),
    "VI":  Chord("A C E"),
    "VII": Chord("B D F") }

# Calculate minimum Euclidean voice-leading distances
# in the chromatic pitch-class orbifold.
voiceLeading = EfficientVoiceLeading(
    scale=ChromaticScale("C"),
    metric=lambda delta: np.linalg.norm(delta, ord=2),
    permutation=Permutation.ANY )

# Construct the 6 x 6 orbifold distance matrix.
numStates = len(states)
orbifoldDistances = np.zeros((numStates, numStates))

for i, stateA in enumerate(states):
    for j, stateB in enumerate(states):
        movement, distance = voiceLeading( chords[stateA], chords[stateB] )
        orbifoldDistances[i, j] = distance

# Normalize using the maximum distance within the fixed six-chord vocabulary.
maxOrbifoldDistance = np.max(orbifoldDistances)

normalizedOrbifoldDistances = (
    orbifoldDistances / maxOrbifoldDistance
)

# Verify fundamental distance properties.
assert np.allclose(orbifoldDistances, orbifoldDistances.T)
assert np.allclose(np.diag(orbifoldDistances), 0)
assert np.all(orbifoldDistances >= 0)

print("States:", states)
print("\nOrbifold distance matrix:")
print(np.round(orbifoldDistances, 3))

print("\nMaximum orbifold distance:", maxOrbifoldDistance)

print("\nNormalized orbifold distance matrix:")
print(np.round(normalizedOrbifoldDistances, 3))
