import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).resolve().parents[1]  # 2 levels up
sys.path.append(str(PRJ_DIR))

import numpy as np

DATA_DIR = PRJ_DIR / "data" / "greenness"
OUTPUT_DIR = PRJ_DIR / "data" / "sonif-input"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

imageDate = "20240801"
# imageDate = "20250201"

inputNpz  = f"{imageDate}-griddata-d35.npz"
outputNpz = f"{imageDate}-south-regions-d35.npz"
outputCsv = f"{imageDate}-south-regions-d35.csv"

numRows = 16
numCols = 32
colsPerRegion = 4
southStartRow = numRows // 2
numRegions = numCols // colsPerRegion

# Load the five 16 x 32 statistics produced by process-image.py.
with np.load(DATA_DIR / inputNpz) as data:
    requiredKeys = ("meanScores", "stdScores", "validFractions",
                    "snowFractions", "landFractions")
    for key in requiredKeys:
        if key not in data:
            raise KeyError(f"Missing {key} in {inputNpz}")

    meanScores     = data["meanScores"]
    stdScores      = data["stdScores"]
    validFractions = data["validFractions"]
    snowFractions  = data["snowFractions"]
    landFractions  = data["landFractions"]

for name, values in (
    ("meanScores", meanScores),
    ("stdScores", stdScores),
    ("validFractions", validFractions),
    ("snowFractions", snowFractions),
    ("landFractions", landFractions)):
    if values.shape != (numRows, numCols):
        raise ValueError(f"{name}'s shape:{values.shape}; expected ({numRows}, {numRows})")

assert np.all(np.isfinite(validFractions))
assert np.all(np.isfinite(snowFractions))
assert np.all(np.isfinite(landFractions))
assert np.all((validFractions >= 0) & (validFractions <= 1))
assert np.all((snowFractions >= 0) & (snowFractions <= 1))
assert np.all((landFractions >= 0) & (landFractions <= 1 + 1e-10))
assert np.all(validFractions + snowFractions <= 1 + 1e-10)
assert np.allclose(landFractions, validFractions + snowFractions, atol=1e-10)
assert np.array_equal(np.isnan(meanScores), validFractions == 0)
assert np.array_equal(np.isnan(stdScores), validFractions == 0)
assert np.all((meanScores[np.isfinite(meanScores)] >= 0) &
              (meanScores[np.isfinite(meanScores)] <= 1))
assert np.all((stdScores[np.isfinite(stdScores)] >= 0) &
              (stdScores[np.isfinite(stdScores)] <= 0.5 + 1e-10))

# Area represented by an equirectangular cell is proportional to cos(latitude).
# Rows run from 90 N to 90 S; the Southern Hemisphere is rows 8 through 15.
latCenters = 90.0 - (np.arange(numRows) + 0.5) * (180.0 / numRows)
latWeights = np.cos(np.deg2rad(latCenters))

regionMeanScores =     np.full(numRegions, np.nan)
regionStdScores =      np.full(numRegions, np.nan)
regionValidFractions = np.zeros(numRegions)
regionSnowFractions =  np.zeros(numRegions)
regionLandFractions =  np.zeros(numRegions)

for region in range(numRegions):
    colStart = region * colsPerRegion
    colEnd = colStart + colsPerRegion
    regionSlice = (slice(southStartRow, numRows), slice(colStart, colEnd))

    means = meanScores[regionSlice]
    stds  = stdScores[regionSlice]
    valid = validFractions[regionSlice]
    snow  = snowFractions[regionSlice]
    land  = landFractions[regionSlice]

    # Equal longitude widths; repeat latitude-area weights over the 4 columns.
    areaWeights = np.broadcast_to( 
        latWeights[southStartRow:, np.newaxis], valid.shape)
    totalAreaWeight = np.sum(areaWeights)

    # All three fractions use the entire mapped region as their denominator.
    regionValidFractions[region] = np.sum(areaWeights * valid) / totalAreaWeight
    regionSnowFractions[region] = np.sum(areaWeights * snow) / totalAreaWeight
    regionLandFractions[region] = np.sum(areaWeights * land) / totalAreaWeight
