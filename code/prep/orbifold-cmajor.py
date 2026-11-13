# Visualize a fundamental domain of the three-pitch orbifold.
# The two triangular end faces are identified with a twist.
# Straight-line distances in these plotting coordinates do not
# generally equal the efficient voice-leading distances.

import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).resolve().parents[1] # 2 levels up
sys.path.append(str(PRJ_DIR))

import numpy as np
from music21.chord import Chord
from music21.scale import ChromaticScale
from orbichord.chordinate import standardSimplex, EfficientVoiceLeading, Permutation
import plotly.graph_objects as go

OUTPUT_DIR = PRJ_DIR / "data" / "orbifold"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

outputHtml = OUTPUT_DIR / "orbifold-cmajor-six-triads.html"

states = ("I", "II", "IV", "V", "VI", "VII")
stateIndices = {state: index for index, state in enumerate(states)}

chords = {"I":   Chord("C E G"),
          "II":  Chord("D F A"),
          "IV":  Chord("F A C"),
          "V":   Chord("G B D"),
          "VI":  Chord("A C E"),
          "VII": Chord("B D F") }

romanLabels = {"I":   "I (CEG)",
               "II":  "ii (DFA)",
               "IV":  "IV (FAC)",
               "V":   "V (GBD)",
               "VI":  "vi (ACE)",
               "VII": "vii° (BDF)" }

orbifoldScale = ChromaticScale("C")

chordData = []
coordsByState = {}

for state in states:
    chord = chords[state]
    coords = standardSimplex(chord, orbifoldScale)
    label = romanLabels[state]
    quality = chord.commonName

    chordData.append((chord, coords, label, quality))
    coordsByState[state] = np.array(coords)

print("Number of points:", len(chordData))

for chord, coords, label, quality in chordData:
    x, y, z = coords
    print(f"{label:20s} {quality:20s} x={x:.4f}, y={y:.4f}, z={z:.4f}")

voiceLeading = EfficientVoiceLeading(
    scale=orbifoldScale,
    metric=lambda delta: np.linalg.norm(delta, ord=2),
    permutation=Permutation.ANY)

numStates = len(states)
orbifoldDistances = np.zeros((numStates, numStates))

for i, stateA in enumerate(states):
    for j, stateB in enumerate(states):
        movement, distance = voiceLeading(chords[stateA], chords[stateB])
        orbifoldDistances[i, j] = distance

maxOrbifoldDistance = np.max(orbifoldDistances)
normalizedOrbifoldDistances = orbifoldDistances / maxOrbifoldDistance

assert len(chordData) == 6
assert np.allclose(orbifoldDistances, orbifoldDistances.T)
assert np.allclose(np.diag(orbifoldDistances), 0)
assert np.isclose(maxOrbifoldDistance, np.sqrt(12))

print("\nOrbifold distance matrix:")
print(np.round(orbifoldDistances, 3))

print("\nNormalized orbifold distance matrix:")
print(np.round(normalizedOrbifoldDistances, 3))

fig = go.Figure()

# Define the six vertices of the triangular prism.
vertices = [
    (0, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 0),
    (1, 1, 0),
    (1, 0, 1) ]

# Add transparent prism surfaces.
faces = [
    (0, 1, 2), (3, 4, 5),
    (0, 1, 4), (0, 4, 3),
    (0, 2, 5), (0, 5, 3),
    (1, 2, 5), (1, 5, 4) ]

i, j, k = zip(*faces)

fig.add_trace( go.Mesh3d( x=[v[0] for v in vertices],
                          y=[v[1] for v in vertices],
                          z=[v[2] for v in vertices],
                          i=i, j=j, k=k,
                          color="lightsteelblue",
                          opacity=0.10,
                          hoverinfo="skip",
                          showlegend=False ))

# Plot the six diatonic triads with colors by chord quality.
colors = {
    "major triad": "royalblue",
    "minor triad": "seagreen",
    "diminished triad": "darkorange"}

for quality, color in colors.items():
    # Select only chords of the current quality
    group = [ (chord, coords, label) for chord, coords, label, q in chordData if q == quality ]
    if not group: 
        continue

    fig.add_trace(go.Scatter3d(
        x=[coords[0] for chord, coords, label in group],
        y=[coords[1] for chord, coords, label in group],
        z=[coords[2] for chord, coords, label in group],

        mode="markers+text",
        marker=dict(
            size=7,
            color=color,
            opacity=0.95),
        text=[label for chord, coords, label in group],
        textposition="top center",
        hovertemplate=(
            "%{text}<br>"
            "x=%{x:.4f}<br>"
            "y=%{y:.4f}<br>"
            "z=%{z:.4f}"
            "<extra></extra>"),
        name=quality.title()
    ))

# Draw the nine edges of the prism.
edges = [
    (0, 1), (1, 2), (2, 0),
    (3, 4), (4, 5), (5, 3),
    (0, 3), (1, 4), (2, 5) ]

edgeX = []
edgeY = []
edgeZ = []

for a, b in edges:
    edgeX.extend([vertices[a][0], vertices[b][0], None])
    edgeY.extend([vertices[a][1], vertices[b][1], None])
    edgeZ.extend([vertices[a][2], vertices[b][2], None])

fig.add_trace(go.Scatter3d(
    x=edgeX,
    y=edgeY,
    z=edgeZ,
    mode="lines",
    line=dict(color="black", width=5),
    hoverinfo="skip",
    name="Prism edges" ))

# Draw the augmented-triad center line.
fig.add_trace(go.Scatter3d(
    x=[0, 1],
    y=[1/3, 1/3],
    z=[1/3, 1/3],
    mode="lines",
    line=dict(color="orange", width=6, dash="dash"),
    name="Augmented-triad line" ))

# Improve layout.
fig.update_layout(
    title="Six C-Major Triads in the 3-Pitch Orbifold",
    width=1150,
    height=850,
    scene=dict(
        xaxis=dict(title="x (transposition)", range=[0, 1]),
        yaxis=dict(title="y (first pitch gap)", range=[0, 1]),
        zaxis=dict(title="z (second pitch gap)", range=[0, 1]),
        aspectmode="manual",
        aspectratio=dict(x=1.6, y=1, z=1),
        camera=dict(eye=dict(x=1.8, y=1.5, z=1.2))
    ))

# Highlight selected chord pairs with straight-line visual guides.
# These segments are NOT shortest paths through the orbifold.
# The legend shows the actual distances calculated by Orbichord.
pairsToHighlight = [
    ("IV", "VI"),
    ("IV", "V"),
    ("V", "I") ]
pairColors = ["seagreen", "crimson", "royalblue"]

for (stateA, stateB), color in zip(pairsToHighlight, pairColors):
    coordsA = coordsByState[stateA]
    coordsB = coordsByState[stateB]

    i = stateIndices[stateA]
    j = stateIndices[stateB]

    distance = orbifoldDistances[i, j]
    normalizedDistance = normalizedOrbifoldDistances[i, j]

    fig.add_trace(go.Scatter3d(
        x=[coordsA[0], coordsB[0]],
        y=[coordsA[1], coordsB[1]],
        z=[coordsA[2], coordsB[2]],
        mode="lines",
        line=dict(color=color, width=7),
        name=(f"{stateA} → {stateB}: "
              f"d={distance:.3f}, "
              f"normalized={normalizedDistance:.3f}")
    ))

fig.write_html(str(outputHtml))
print(f"Saved: {outputHtml}")

fig.show(renderer="browser")
