import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).resolve().parents[1] # 2 levels up
sys.path.append(str(PRJ_DIR))

import cv2, numpy as np
from colorbar import pixel2vegetationScore

DATA_DIR = PRJ_DIR / "data"
OUTPUT_DIR = PRJ_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

inputImage = "SNPP_VIIRS.20240801.ndvi_w_snow.oceanblue.png"
outputImage        = "20240801-32x16.png"
outputBinaryMask   = "20240801-valid-mask.png"
outputOverlay      = "20240801-valid-maskoverlay.png"
outputNpz          = "20240801-griddata.npz"
outputCsvGreenness = "20240801-greenness.csv"
outputCsvFractions = "20240801-fractions.csv"
outputMismatchMask = "20240801-color-mismatch.png"

def inspectColorDistances(row, col, sampleStep=10):
    cell = ndviImage[ yEdges[row]:yEdges[row + 1],
                      xEdges[col]:xEdges[col + 1] ]
    samplePixels = cell[::sampleStep, ::sampleStep].reshape(-1, 3)
    distances = []

    for bgrPixel in samplePixels:
        if (np.array_equal(bgrPixel, oceanBgr)
            or np.array_equal(bgrPixel, whiteIceBgr)):
            continue

        snowDistance = np.linalg.norm( bgrPixel.astype(np.int16) - snowBgrInt )
        if snowDistance <= maxSnowDistance:
            continue

        score, colorDistance = pixel2vegetationScore(bgrPixel)
        distances.append(colorDistance)

    print(f"\nCell ({row}, {col})")
    print(f"Candidate samples: {len(distances)} / {len(samplePixels)}")
    print(f"Candidate fraction: {len(distances) / len(samplePixels):.3f}")

    if len(distances) == 0:
        print("No sampled land candidates")
        return

    distances = np.array(distances)

    print("Median distance:", np.median(distances))
    print("90th percentile:", np.percentile(distances, 90))

    for threshold in [20, 25, 30, 35, 40]:
        fraction = np.mean(distances <= threshold)
        print(f"Threshold {threshold}: {fraction:.3f}")


ndviImage = cv2.imread(str(DATA_DIR / inputImage))
height, width = ndviImage.shape[:2]

# Divide the image into 32 columns and 16 rows.
numCols = 32
numRows = 16
xEdges = np.linspace(0, width,  numCols + 1, dtype=int)
yEdges = np.linspace(0, height, numRows + 1, dtype=int)

firstCell = ndviImage[yEdges[0]:yEdges[1], xEdges[0]:xEdges[1]]

print(f"Image size: {width} x {height}")
print(f"Grid size: {numCols} x {numRows}")
print(f"Total cells: {numCols * numRows}")
print(f"First cell size: {firstCell.shape}")
# print(f"First cell size: {cells[0][0].shape}")

# Draw grid lines for visualization.
gridImage = ndviImage.copy()

for x in xEdges:
    cv2.line(gridImage, (int(x), 0), (int(x), height - 1), (0, 0, 255), 2)
    
for y in yEdges:
    cv2.line(gridImage, (0, int(y)), (width - 1, int(y)), (0, 0, 255), 2)

cv2.imwrite(str(OUTPUT_DIR / outputImage), gridImage)
print(outputImage, "saved.")

# Compute vegetation statistics for each cell.
meanScores        = np.full((numRows, numCols), np.nan)
validFractions    = np.zeros((numRows, numCols))
validMask         = np.zeros((height, width), dtype=np.uint8)
colorMismatchMask = np.zeros((height, width), dtype=np.uint8)

maxColorDistance = 20.0     # Preliminary threshold; requires validation
maxSnowDistance  = 8.0
oceanBgr = ndviImage[0, 0]  # Known ocean (top-left corner) pixel in the image
snowBgr  = ndviImage[-1, 0] # known snow (bottom-left corner) pixel in the image
snowBgrInt = snowBgr.astype(np.int16)
whiteIceBgr = np.array([255, 255, 255], dtype=np.uint8)

# Inspect representative regions
inspectColorDistances(8, 10)   # Amazon rainforest
inspectColorDistances(2, 19)   # Northern Russian forest
inspectColorDistances(6, 16)   # Sahara Desert
inspectColorDistances(1, 12)   # Greenland
inspectColorDistances(14, 10)  # Antarctic region

print("Processing row: ", end="")
for row in range(numRows):
    for col in range(numCols):
        xStart, xEnd = xEdges[col], xEdges[col + 1]
        yStart, yEnd = yEdges[row], yEdges[row + 1]

        cell = ndviImage[yStart:yEnd, xStart:xEnd]
        validScores = []
        validCount = 0

        for localY in range(cell.shape[0]):
            for localX in range(cell.shape[1]):
                bgrPixel = cell[localY, localX]

                # Exclude ocean and white ice pixels.
                if (np.array_equal(bgrPixel, oceanBgr)
                    or np.array_equal(bgrPixel, whiteIceBgr)):
                    continue

                # Exclude pixels close to the reference snow color.
                snowDistance = np.linalg.norm(bgrPixel.astype(np.int16) - snowBgrInt)
                if snowDistance <= maxSnowDistance:
                    continue

                score, colorDistance = pixel2vegetationScore(bgrPixel)
                if colorDistance <= maxColorDistance:
                    validScores.append(score)
                    validCount += 1
                    validMask[yStart + localY, xStart + localX] = 255
                else:
                    colorMismatchMask[yStart + localY, xStart + localX] = 255

        if validScores: meanScores[row, col] = np.mean(validScores)
        validFractions[row, col] = validCount / (cell.shape[0] * cell.shape[1])

    print(row, end=" ")
print()

cv2.imwrite(str(OUTPUT_DIR / outputBinaryMask), validMask)
print(outputBinaryMask, "saved.")

cv2.imwrite( str(OUTPUT_DIR / outputMismatchMask), colorMismatchMask)
print(outputMismatchMask, "saved.")

overlayImage = ndviImage.copy()
rejected = (validMask == 0)
overlayImage[rejected] = (0.5 * overlayImage[rejected] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
# overlayImage[validMask == 0] = [0, 0, 255]  # rejected pixels shown in red
cv2.imwrite(str(OUTPUT_DIR / outputOverlay), overlayImage)
print(outputOverlay, "saved.")

np.savez( OUTPUT_DIR / outputNpz,
          meanScores    = meanScores,
          validFractions= validFractions )
print(outputNpz, "saved.")

np.savetxt( OUTPUT_DIR / outputCsvGreenness,
            meanScores, delimiter=",", fmt="%.6f" )
print(outputCsvGreenness, "saved.")

np.savetxt( OUTPUT_DIR / outputCsvFractions,
            validFractions, delimiter=",", fmt="%.6f" )
print(outputCsvFractions, "saved.")


