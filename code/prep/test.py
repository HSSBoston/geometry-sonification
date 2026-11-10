import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).resolve().parents[1] # 2 levels up
sys.path.append(str(PRJ_DIR))

import cv2, numpy as np

DATA_DIR    = PRJ_DIR / "data" 
OUTPUT_DIR = PRJ_DIR / "output"

if __name__ == "__main__":
    imagePath = DATA_DIR / "NDVI_colorbar.png"

    print("Project directory:", PRJ_DIR)
    print("Image path:", imagePath)
    print("File exists:", imagePath.is_file())
    print("Files in data directory:", list(DATA_DIR.iterdir()))

    ndviImage = cv2.imread(str(imagePath))

    if ndviImage is None:
        raise RuntimeError(f"Cannot read image: {imagePath}")