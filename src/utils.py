import numpy as np
import cv2
import json
from enum import Enum

def read_create_seg_map(sample_x, sample_y):
    image_x = cv2.imread(sample_x, cv2.IMREAD_GRAYSCALE)
    h, w = image_x.shape[:2]
    data_y = np.zeros((h, w), dtype=np.uint8)
    for seg in sample_y:
        pts = np.array(seg).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(data_y, [pts], 1)
    return image_x, data_y

class DATASETS(Enum):
    TRAIN: str = "../data/train"
    TEST: str = "../data/test"
    VALIDATION: str = "../data/valid"

def get_dataset(dataset: str):
    path = f"{dataset}/_annotations.coco.json"
    with open(path, "r") as f:
        annotations = json.load(f)

    X = []
    for image in annotations["images"]:
        full_file_path = f"{dataset}/{image['file_name']}"
        X.append(full_file_path)

    Y = [[] for _ in range(len(X))]
    for annotation in annotations["annotations"]:
        Y[annotation["image_id"]].append(np.int32(annotation["segmentation"]))

    return X, Y
