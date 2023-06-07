import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from PIL import Image
import torch
from torchvision import models
import torchvision.transforms as T
import segmentation_models_pytorch as smp
import os
from pprint import pprint
import torch
from torch.utils.data import DataLoader
from PIL import Image
import splitfolders
import random
import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

images = ('C:/Users/Noah/Desktop/DeepFish2/Segmentation/images/valid')
masks = ('C:/Users/Noah/Desktop/DeepFish2/Segmentation/masks/valid')

img = Image.open("C:/Users/Noah/Desktop/DeepFish2/Segmentation/images/valid/7398_F2_f000010.jpg")
mask = cv2.imread("C:/Users/Noah/Desktop/DeepFish2/Segmentation/masks/valid/7398_F2_f000010.png")

# plt.imshow(img); plt.show()
DIR = "C:/Users/Noah/Desktop/DeepFish2/Segmentation/Images_fix/"
x_train_dir = os.path.join(DIR, 'images-valid/')
y_train_dir = os.path.join(DIR, 'masks-valid/')

# print(x_train_dir)
# print(y_train_dir)
# print(os.path.join(x_train_dir, os.listdir(x_train_dir)[0]))

# print(len(os.listdir(x_train_dir)))

img = Image.open(os.path.join(y_train_dir, os.listdir(y_train_dir)[35]))
# img.show()

random.seed(1)
randomTrain = []
for i in range(0, 248):
    n = random.randint(1, len(os.listdir(x_train_dir)))
    randomTrain.append(n)
print(randomTrain)

randomTest = []
for i in range(0, 31):
    n = random.randint(1, len(os.listdir(x_train_dir)))
    randomTest.append(n)
print(randomTest)

randomVal = []
for i in range(0, 31):
    n = random.randint(1, len(os.listdir(x_train_dir)))
    randomVal.append(n)
print(randomVal)

train_image_dest = "C:/Users/Noah/Desktop/DeepFish2/Segmentation/FishCount/Train/Images/"
train_mask_dest = "C:/Users/Noah/Desktop/DeepFish2/Segmentation/FishCount/Train/Masks/"
test_image_dest = "C:/Users/Noah/Desktop/DeepFish2/Segmentation/FishCount/Test/Images/"
test_mask_dest = "C:/Users/Noah/Desktop/DeepFish2/Segmentation/FishCount/Test/Masks/"

for i in randomTrain:
    shutil.copy2(os.path.join(x_train_dir, os.listdir(x_train_dir)[i]), train_image_dest)
    shutil.copy2(os.path.join(y_train_dir, os.listdir(y_train_dir)[i]), train_mask_dest)

for i in randomTest:
    shutil.copy2(os.path.join(x_train_dir, os.listdir(x_train_dir)[i]), test_image_dest)
    shutil.copy2(os.path.join(y_train_dir, os.listdir(y_train_dir)[i]), test_mask_dest)
