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

# define RBF network
# reference http://blogs.ekarshi.com/wp/2017/03/20/rbfradial-basis-function-neural-network-in-python-machine-learning/
class RBFNetwork:
    def __init__(self, pTypes, scaledData):
        self.pTypes = pTypes
        self.protos = np.zeros(shape=(0,40))
        self.scaledData = scaledData
        self.spread = 0
        self.weights = 0
         
    def generatePrototypes(self):
        group1 = np.random.randint(0,30,size=self.pTypes)
        group2 = np.random.randint(31,61,size=self.pTypes)
        group3 = np.random.randint(62,92,size=self.pTypes)
        self.protos = np.vstack([self.protos,self.scaledData[group1,:],self.scaledData[group2,:],self.scaledData[group3,:]])
        return self.protos
    
    def sigma(self):
        dTemp = 0
        for i in range(0,self.pTypes*3):
            for k in range(0,self.pTypes*3):
                dist = np.square(np.linalg.norm(self.protos[i] - self.protos[k]))
                if dist and dTemp:
                    dTemp = dist
        self.spread = dTemp/np.sqrt(self.pTypes*3)
 
    def train(self):
        self.generatePrototypes()
        self.sigma()
        hiddenOut = np.zeros(shape=(0,self.pTypes*3))
        for item in self.scaledData:
            out=[]
            for proto in self.protos:
                distance = np.square(np.linalg.norm(item - proto))
                neuronOut = np.exp(-(distance)/(np.square(self.spread)))
                out.append(neuronOut)
            hiddenOut = np.vstack([hiddenOut,np.array(out)])
        print(hiddenOut)
        self.weights = np.dot(pinv(hiddenOut),self.labels)
        print(self.weights)
    def test(self):
        items = [3,4,72,82,91,24,66,98,67,101,19]
        for item in items:
            data = self.scaledData[item]
            out = []
            for proto in self.protos:
                distance = np.square(np.linalg.norm(data-proto))
                neuronOut = np.exp(-(distance)/np.square(self.spread))
                out.append(neuronOut)
             
            netOut = np.dot(np.array(out),self.weights)
            print('---------------------------------')
            print(netOut)
            print('Class is ',netOut.argmax(axis=0) + 1)
#             print('Given Class ',self.labels[item].argmax(axis=0) +1)

label = list(range(40))
network = RBFNetwork(pca_data,label)
network.train()
network.test()