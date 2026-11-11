import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).resolve().parents[1] # 2 levels up
sys.path.append(str(PRJ_DIR))

import cv2, numpy as np
from colorbar import pixel2vegetationScore

DATA_DIR = PRJ_DIR / "data"
OUTPUT_DIR = PRJ_DIR / "output"

imageFileName  = "SNPP_VIIRS.20240801.ndvi_w_snow.oceanblue.png"
outputFileName = "ndviGrid32x16.png"

ndviImage = cv2.imread(str(DATA_DIR / imageFileName))
height, width = ndviImage.shape[:2]

# Divide the image into 32 columns and 16 rows.
numCols = 32
numRows = 16
xEdges = np.linspace(0, width,  numCols + 1, dtype=int)
yEdges = np.linspace(0, height, numRows + 1, dtype=int)

cells = []
for row in range(numRows):
    rowCells = []
    for col in range(numCols):
        xStart, xEnd = xEdges[col], xEdges[col + 1]
        yStart, yEnd = yEdges[row], yEdges[row + 1]

        cell = ndviImage[yStart:yEnd, xStart:xEnd].copy()
        rowCells.append(cell)

    cells.append(rowCells)

print(f"Image size: {width} x {height}")
print(f"Grid size: {numCols} x {numRows}")
print(f"Total cells: {numCols * numRows}")
print(f"First cell size: {cells[0][0].shape}")

# Draw grid lines for visualization.
gridImage = ndviImage.copy()

for x in xEdges:
    cv2.line(gridImage, (int(x), 0), (int(x), height - 1), (0, 0, 255), 2)
    
for y in yEdges:
    cv2.line(gridImage, (0, int(y)), (width - 1, int(y)), (0, 0, 255), 2)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUTPUT_DIR / outputFileName), gridImage)
print(outputFileName, "generated.")

# Compute vegetation statistics for each cell.
meanScores = np.full((numRows, numCols), np.nan)
validFractions = np.zeros((numRows, numCols))

maxColorDistance = 20.0    # Preliminary threshold; requires validation.
oceanBgr = ndviImage[0, 0] # Known ocean pixel in this image.

for row in range(numRows):
    for col in range(numCols):
        cell = cells[row][col]
        validScores = []

        for bgrPixel in cell.reshape(-1, 3):
            # Exclude ocean pixels.
            if np.array_equal(bgrPixel, oceanBgr):
                continue

            score, colorDistance = pixel2vegetationScore(bgrPixel)

            if colorDistance <= maxColorDistance:
                validScores.append(score)

        if validScores:
            meanScores[row, col] = np.mean(validScores)

        validFractions[row, col] = len(validScores) / (cell.shape[0] * cell.shape[1])
