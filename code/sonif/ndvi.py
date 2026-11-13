import pathlib, csv, numpy as np
from constants import DATA_SONIF_DIR

def calculateNdviDistances(imageDate):
    inputCsv = f"{imageDate}-south-regions-chords.csv"
    with open(DATA_SONIF_DIR / inputCsv, newline="") as file:
        rows = list(csv.DictReader(file))

    # Extract land-adjusted greenness
    greenness = np.array(
        [ float(row["landGreenness"]) for row in rows ] )

    # Extract geographical longitude centers
    longitudeCenters = np.array(
        [ (float(row["longitudeStart"]) + float(row["longitudeEnd"])) / 2.0
          for row in rows ] )

    # Spatial separation normalized by 45 degrees
    spatialDistances = np.abs( np.diff(longitudeCenters) ) / 45.0
    if np.any(spatialDistances <= 0): raise ValueError("Spatial distances must be positive.")

    # Absolute differences between consecutive greenness values
    greennessDifferences = np.abs(np.diff(greenness))

    # Raw spatial gradients
    ndviGradients = greennessDifferences / spatialDistances

    # Theoretical normalization for 2-column/4-column regions:
    # minimum normalized spatial separation = 0.5
    normalizedNdviDistances = ndviGradients / 2.0

    if not np.all(np.isfinite(normalizedNdviDistances)):
        raise ValueError("Nonfinite NDVI distances.")

    if np.any((normalizedNdviDistances < 0) | (normalizedNdviDistances > 1 + 1e-10)):
        raise ValueError("Normalized NDVI distances outside [0, 1].")

    return normalizedNdviDistances
