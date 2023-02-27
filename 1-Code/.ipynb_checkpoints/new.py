import os
import cv2
import sys
import glob
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image

print("The Python Version Number is : ", sys.version)
print("The Pytorch Version is : ",torch.__version__)
DATASET_PATH = os.path.join("../2-Dataset/")
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "2d_images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "2d_masks")
IMAGE_FILES_DIR = sorted(os.listdir(IMAGE_DATASET_PATH))
MASK_FILES_DIR  = sorted(os.listdir(MASK_DATASET_PATH))

print("The number of image is :", len(IMAGE_FILES_DIR))
print("The number of mask is :", len(MASK_FILES_DIR))

view_original_image01 = Image.open("../2-Dataset/2d_images/ID_0000_Z_0142.tif")
plt.imshow(view_original_image01)


view_original_image02 = Image.open("../2-Dataset/2d_masks/ID_0000_Z_0142.tif")
plt.imshow(view_original_image02)