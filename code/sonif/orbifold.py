import numpy as np
from orbichord.chordinate import EfficientVoiceLeading, Permutation
from music21.scale import ChromaticScale

from constants import CHORDS

def calculateOrbifoldDistances(states):
    # Euclidean voice-leading distance in chromatic pitch-class space
    voiceLeading = EfficientVoiceLeading(
        scale=ChromaticScale("C"),
        metric=lambda delta: np.linalg.norm(delta, ord=2),
        permutation=Permutation.ANY )

    numStates = len(states)
    distances = np.zeros((numStates, numStates))

    for i, stateA in enumerate(states):
        for j, stateB in enumerate(states):
            movement, distance = voiceLeading( CHORDS[stateA], CHORDS[stateB] )
            distances[i, j] = distance

    # Verify basic distance properties.
    assert np.allclose(distances, distances.T)
    assert np.allclose(np.diag(distances), 0)
    assert np.all(distances >= 0)

    # Normalize over the fixed six-chord vocabulary
    maxDistance = np.max(distances)
    if maxDistance <= 0: raise ValueError("Maximum orbifold distance must be positive.")
    return distances / maxDistance

