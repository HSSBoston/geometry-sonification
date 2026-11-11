import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).resolve().parents[1] # 2 levels up
sys.path.append(str(PRJ_DIR))

import cv2, numpy as np
from colorbar import pixel2vegetationScore

DATA_DIR = PRJ_DIR / "data"
OUTPUT_DIR = PRJ_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

imageDate = "20250201"

inputImage         = f"SNPP_VIIRS.{imageDate}.ndvi_w_snow.oceanblue.png"
outputImage        = f"{imageDate}-32x16.png"
outputBinaryMask   = f"{imageDate}-valid-mask-d35.png"
outputOverlay      = f"{imageDate}-valid-maskoverlay-d35.png"
outputMismatchMask = f"{imageDate}-color-mismatch-d35.png"
outputNpz          = f"{imageDate}-griddata-d35.npz"
outputCsvGreenness = f"{imageDate}-greenness-d35.csv"
outputCsvFractions = f"{imageDate}-fractions-d35.csv"

def inspectColorDistances(row, col, sampleStep=10):
    cell = ndviImage[ yEdges[row]:yEdges[row + 1],
                      xEdges[col]:xEdges[col + 1] ]

    sampledCell  = cell[::sampleStep, ::sampleStep]
    samplePixels = sampledCell.reshape(-1, 3)
    sampleCols   = sampledCell.shape[1]

    distances = []
    candidateColors = []

    for sampleIndex, bgrPixel in enumerate(samplePixels):
        imageY = yEdges[row] + (sampleIndex // sampleCols) * sampleStep
        if (np.array_equal(bgrPixel, oceanBgr)
            or np.array_equal(bgrPixel, whiteIceBgr)):
            continue

        snowDistance = np.linalg.norm( bgrPixel.astype(np.int16) - snowBgrInt )
        if snowDistance <= maxSnowDistance:
            continue

        if imageY >= antarcticStartY:
            antarcticSnowDistance = np.min(
                np.linalg.norm( antarcticSnowColors - bgrPixel.astype(np.int16), axis=1 ))
            if antarcticSnowDistance <= maxAntarcticSnowDistance:
                continue
    
        score, colorDistance = pixel2vegetationScore(bgrPixel)
        distances.append(colorDistance)
        candidateColors.append(tuple(int(v) for v in bgrPixel))

    print(f"\nCell ({row}, {col})")
    print(f"Samples passing ocean/snow filters: {len(distances)} / {len(samplePixels)}")
    print(f"Candidate fraction: {len(distances) / len(samplePixels):.3f}")

    if len(distances) == 0:
        print("No sampled pixels passed the exclusion filters")
        return

    if row == 14 and col == 10:
        from collections import Counter
        print("Reference snow BGR:", snowBgr)
        print("Most common candidate colors:")

        for color, count in Counter(candidateColors).most_common(10):
            snowDistance = np.linalg.norm( np.array(color, dtype=np.int16) - snowBgrInt )
            print(color, count, f"Snow distance: {snowDistance:.2f}")

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

maxColorDistance = 35.0
maxSnowDistance  = 8.0
maxAntarcticSnowDistance = 4.0

oceanBgr = ndviImage[0, 0]  # Known ocean (top-left corner) pixel in the image
snowBgr  = ndviImage[-1, 0] # known snow (bottom-left corner) pixel in the image
snowBgrInt = snowBgr.astype(np.int16)
whiteIceBgr = np.array([255, 255, 255], dtype=np.uint8)

print("Ocean BGR:", oceanBgr)
print("Snow BGR:", snowBgr)

antarcticSnowColors = np.array([
    [221, 229, 252],
    [217, 227, 250],
    [211, 222, 247],
    [208, 221, 246],
    [215, 225, 249]
], dtype=np.int16)
antarcticStartY = int(height * 150 / 180)

# Inspect representative regions
inspectColorDistances(2, 5)    # Northern North America
inspectColorDistances(2, 19)   # Northern Russian
inspectColorDistances(1, 12)   # Greenland

inspectColorDistances(8, 10)   # Amazon rainforest
inspectColorDistances(6, 16)   # Sahara Desert
inspectColorDistances(14, 10)  # Antarctica

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

                imageY = yStart + localY
                if imageY >= antarcticStartY:
                    antarcticSnowDistance = np.min(
                        np.linalg.norm(antarcticSnowColors - bgrPixel.astype(np.int16), axis=1))
                    if antarcticSnowDistance <= maxAntarcticSnowDistance:
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


