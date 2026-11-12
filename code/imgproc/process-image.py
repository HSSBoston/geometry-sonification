import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).resolve().parents[1] # 2 levels up
sys.path.append(str(PRJ_DIR))

import cv2, numpy as np
from colorbar import pixel2vegetationScore

DATA_DIR = PRJ_DIR / "data" / "input"
OUTPUT_DIR = PRJ_DIR / "data" / "greenness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#imageDate = "20240801"
imageDate = "20250201"

inputImage         = f"SNPP_VIIRS.{imageDate}.ndvi_w_snow.oceanblue.png"
outputImage        = f"{imageDate}-32x16.png"
outputBinaryMask   = f"{imageDate}-valid-mask-d35.png"
outputOverlay      = f"{imageDate}-valid-maskoverlay-d35.png"
outputMismatchMask = f"{imageDate}-color-mismatch-d35.png"
outputSnowMask     = f"{imageDate}-snow-mask-d35.png"

outputNpz          = f"{imageDate}-griddata-d35.npz"
outputCsvGreenness = f"{imageDate}-greenness-d35.csv"
outputCsvStdScores = f"{imageDate}-std-scores-d35.csv"
outputCsvValidFractions = f"{imageDate}-valid-fractions-d35.csv"
outputCsvSnowFractions  = f"{imageDate}-snow-fractions-d35.csv"
outputCsvLandFractions  = f"{imageDate}-land-fractions-d35.csv"

# Valid fraction: Fraction of pixels accepted for greenness estimation,
#                 excluding snow/ice, ocean, and palette mismatches
# Snow fraction: Fraction of pixels classified as snow/ice
# Land fraction: Fraction of pixels for land or snow/ice, excluding
#                ocean (valid + snow)

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
meanScores     = np.full((numRows, numCols), np.nan)
stdScores      = np.full((numRows, numCols), np.nan)
validFractions = np.zeros((numRows, numCols))
snowFractions  = np.zeros((numRows, numCols))

validMask         = np.zeros((height, width), dtype=np.uint8)
snowMask          = np.zeros((height, width), dtype=np.uint8)
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
        snowCount  = 0

        for localY in range(cell.shape[0]):
            for localX in range(cell.shape[1]):
                bgrPixel = cell[localY, localX]
                imageY = yStart + localY
                imageX = xStart + localX

                # Exclude ocean pixels.
                if np.array_equal(bgrPixel, oceanBgr):
                    continue

                # Exclude white ice pixels.
                if np.array_equal(bgrPixel, whiteIceBgr):
                    snowCount += 1
                    snowMask[imageY, imageX] = 255
                    continue

                # Exclude pixels close to the reference snow color.
                snowDistance = np.linalg.norm(
                    bgrPixel.astype(np.int16) - snowBgrInt
                )
                if snowDistance <= maxSnowDistance:
                    snowCount += 1
                    snowMask[imageY, imageX] = 255
                    continue

                # Additional Antarctic snow filtering.
                if imageY >= antarcticStartY:
                    antarcticSnowDistance = np.min(
                        np.linalg.norm(
                            antarcticSnowColors - bgrPixel.astype(np.int16),
                            axis=1
                        )
                    )
                    if antarcticSnowDistance <= maxAntarcticSnowDistance:
                        snowCount += 1
                        snowMask[imageY, imageX] = 255
                        continue

                # Classify vegetation pixels.
                score, colorDistance = pixel2vegetationScore(bgrPixel)

                if colorDistance <= maxColorDistance:
                    validScores.append(score)
                    validCount += 1
                    validMask[imageY, imageX] = 255
                else:
                    colorMismatchMask[imageY, imageX] = 255
            
        # Calculate cell-level statistics.
        if validScores:
            meanScores[row, col] = np.mean(validScores)
            stdScores[row, col]  = np.std(validScores)

        numPixels = cell.shape[0] * cell.shape[1]

        validFractions[row, col] = validCount / numPixels
        snowFractions[row, col]  = snowCount / numPixels

    print(row, end=" ")
print()

cv2.imwrite(str(OUTPUT_DIR / outputBinaryMask), validMask )
print(outputBinaryMask, "saved.")

cv2.imwrite( str(OUTPUT_DIR / outputMismatchMask), colorMismatchMask )
print(outputMismatchMask, "saved.")

cv2.imwrite( str(OUTPUT_DIR / outputSnowMask), snowMask )
print(outputSnowMask, "saved.")

overlayImage = ndviImage.copy()
rejected = (validMask == 0)
overlayImage[rejected] = (0.5 * overlayImage[rejected] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
cv2.imwrite(str(OUTPUT_DIR / outputOverlay), overlayImage)
print(outputOverlay, "saved.")

landFractions = validFractions + snowFractions

assert np.all(np.isnan(meanScores) == np.isnan(stdScores))
assert np.all((validFractions >= 0) & (validFractions <= 1))
assert np.all((snowFractions >= 0) & (snowFractions <= 1))
assert np.all(validFractions + snowFractions <= 1.000001)
assert not np.any((validMask > 0) & (snowMask > 0))
assert np.all((landFractions >= 0) & (landFractions <= 1.000001))

print("Mean and standard deviation ranges:")
print(np.nanmin(meanScores), np.nanmax(meanScores))
print(np.nanmin(stdScores), np.nanmax(stdScores))
print("Mean valid fraction:", np.mean(validFractions))
print("Mean snow fraction:", np.mean(snowFractions))
print("Mean land fraction:", np.mean(landFractions))

np.savez( OUTPUT_DIR / outputNpz,
          meanScores=meanScores,
          stdScores=stdScores,
          validFractions=validFractions,
          snowFractions=snowFractions,
          landFractions=landFractions)
print(outputNpz, "saved.")

np.savetxt( OUTPUT_DIR / outputCsvGreenness,
            meanScores, delimiter=",", fmt="%.6f" )
print(outputCsvGreenness, "saved.")

np.savetxt( OUTPUT_DIR / outputCsvValidFractions,
            validFractions, delimiter=",", fmt="%.6f" )
print(outputCsvValidFractions, "saved.")

np.savetxt( OUTPUT_DIR / outputCsvStdScores,
            stdScores, delimiter=",", fmt="%.6f" )
print(outputCsvStdScores, "saved.")

np.savetxt( OUTPUT_DIR / outputCsvSnowFractions,
            snowFractions, delimiter=",", fmt="%.6f" )
print(outputCsvSnowFractions, "saved.")

np.savetxt( OUTPUT_DIR / outputCsvLandFractions,
            landFractions, delimiter=",", fmt="%.6f" )
print(outputCsvLandFractions, "saved.")

