'''
Yichao 
95% PCA implementation and result image sets
'''
import cv2
import glob
import os
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import RandomizedPCA
image_folder = "datasets/s{}"
from PIL import Image, ImageFilter
images = []
for i in range(1,41):
    image = image_folder.format(i)
    image1 = glob.glob(os.path.join(image, "*.pgm"))
    images.append(image1)

#set fixed image size
size = 112,92
# set original data set vector
original_data = []
pca_data = []

# Obtain .95 variance of original data
pca = PCA(.95)
# iterate through iamges
for i in range(40):
    for image in images[i]:
        im = cv2.imread(image, 0)
        original_data.append(im)
    #     fit transform pca and project from high dimension
        pca.fit(im)
        pca_image = pca.fit_transform(im)
        projected = pca.inverse_transform(pca_image)
        pca_data.append(projected)

fig, ax = plt.subplots(80, 10, figsize=(10, 100),
                       subplot_kw={'xticks':[], 'yticks':[]},
                       gridspec_kw=dict(hspace=0, wspace=0.1))
for i in range(40):
    for j in range(10):
        ax[2*i, j].imshow(original_data[10*i+j].reshape(size), cmap='binary_r')
        ax[2*i+1, j].imshow(pca_data[10*i+j].reshape(size), cmap='binary_r')
plt.show()