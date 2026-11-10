import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).resolve().parents[1] # 2 levels up
sys.path.append(str(PRJ_DIR))

import cv2
import numpy as np

DATA_DIR    = PRJ_DIR / "data" 
OUTPUT_DIR = PRJ_DIR / "output"

colorbar = cv2.imread(str(DATA_DIR / "NDVI_colorbar.png"))

# Sample the middle of the colored gradient.
# Approximate coordinates for NASA's 1000x250 legend.
palette = colorbar[125, 100:800]

# Convert palette colors to LAB.
paletteLab = cv2.cvtColor(palette.reshape(1, -1, 3),
                          cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)

def pixel2vegetationScore(bgr_pixel):
    pixel = np.uint8([[bgr_pixel]])
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
