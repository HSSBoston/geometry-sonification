import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).resolve().parents[1] # 2 levels up
sys.path.append(str(PRJ_DIR))

import cv2, numpy as np

DATA_DIR    = PRJ_DIR / "data" / "input"
OUTPUT_DIR = PRJ_DIR / "output" / "greenness"

colorbar = cv2.imread(str(DATA_DIR / "NDVI_colorbar.png"))

# Sample the middle of the colored gradient.
# Approximate coordinates for NASA's 1000x250 legend.
palette = colorbar[125, 102:798]

# Convert palette colors to OpenCV's 8-bit LAB representation.
paletteLab = cv2.cvtColor(palette.reshape(1, -1, 3),
                          cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)

# Takes a BGR pixel and returns:
#   * the normalized vegetation score: [0,1]
#   * the Euclidean distance between the pixel and the nearest color
#     in the color bar: [0,~441]. Two colors are represented in OpenCV's 8-bit
#     CIELAB color space. <5: small color difference. ~20: larger color difference.
#     ~50: substantial color difference.
#
def pixel2vegetationScore(bgrPixel):
    pixel = np.uint8([[bgrPixel]])
    pixelLab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    
    distances = np.linalg.norm(paletteLab - pixelLab, axis=1)
    index = np.argmin(distances)
    # Relative scale: 0 = low, 1 = high
    score = index / (len(paletteLab) - 1)
    # Distance can also flag colors unlike the legend.
    return score, distances[index]

if __name__ == "__main__":
    bgrPixel = colorbar[125, 450]
    score, colorDistance = pixel2vegetationScore(bgrPixel)
    print(f"Vegetation score: {score:.3f}")
    print(f"Color distance: {colorDistance:.2f}")    
