import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).resolve().parents[1] # 2 levels up
sys.path.append(str(PRJ_DIR))

import cv2, numpy as np
from colorbar import pixel2vegetationScore

DATA_DIR = PRJ_DIR / "data" / "greenness"
OUTPUT_DIR = PRJ_DIR / "data" / "sonif-input"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

imageDate = "20240801"
#imageDate = "20250201"

inputNpz = f"{imageDate}-griddata-d35.npz"


