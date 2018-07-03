#David Helale

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import os
import PIL.Image as Image

def pca(X, Z, components=0):
	[a,b] = X.shape
	if (components>a) or (components <= 0):
		components = a
	mean = X.mean(axis=0)
	X= X - mean
	if a>b:
		q = np.dot(X.T,X)
		[eigenval,eigenvec] = np.linalg.eigh(q)
	else:
		q = np.dot(X,X.T)
		[eigenval,eigenvec] = np.linalg.eigh(C)
		eigenvec = np.dot(X.T,eigenvec)
		for num in range(a):
			eigenvec[:,num] = eigenvec[:,num]/np.linalg.norm(eigenvec[:,num])
	eigenval = np.argsort(-eigenval)
	eigenvec = eigenvec[:,eigenval]
	eigenval = eigenval[0:components]
	eigenvec = eigenvec[:,0:components]
	return [eigenval, eigenvec, mean]


def projection(W_proj, X, mean):
	return np.dot(X - mean, W_proj)


def reconstruction(W_proj, z, mean):
	return np.dot(z, W_proj.T) + mean


def lda(X, Z, components):
	Z = np.asarray(Z)
	[a,b] = X.shape
	q = np.unique(Z)
	if (components>(len(q)-1) or (components <= 0)):
		components = (len(q)-1)
	Total_mean = X.mean(axis=0)

	Scatter1,Scatter2 = np.zeros((b, b)), np.zeros((b,b))
	for num in q:
		qq = X[np.where(Z==num)[0],:]
		Class_mean = qq.mean(axis=0)
		Scatter1 = Scatter1 + np.dot((qq-Class_mean).T, (qq-Class_mean))
		Scatter2 = Scatter2 + a * np.dot((Class_mean - Total_mean).T, (Class_mean - Total_mean))
	eigenval, eigenvec = np.linalg.eig(np.linalg.inv(Scatter1)*Scatter2)
	temp = np.argsort(-eigenval.real)
	eigenval, eigenvec = eigenvalues[temp], eigenvectors[:,temp]

	eigenval = np.array(eigenval[0:components].real)
	eigenvec = np.array(eigenvec[0:,0:components].real)
	return [eigenval, eigenvec]

def fisher(X,Z,components=0):

	Z = np.asarray(Z)
	[a,b] = X.shape
	q = len(np.unique(Z))
	[eigenval_pca, eigenvec_pca, mean_pca] = pca(X, Z, (a-q))
	[eigenval_lda, eigenvec_lda] = lda(projection(eigenvec_pca, X, mean_pca), Z, components)
	eigenvec = np.dot(eigenvec_pca,eigenvec_lda)
	return [eigenval_lda, eigenvec, mean_pca]

class EuclDist():
	def __call__(self, r, s):
		r = np.asarray(r).flatten()
		s = np.asarray(s).flatten()
		return np.sqrt(np.sum(np.power((r-s),2)))

class Fisherfaces():

	def __init__(self, X=None, Z=None, metric_dist=EuclDist(), components=0):
		self.metric_dist = metric_dist
		self.components = 0
		self.mean = []
		self.compute(X,Z)
		self.projections = []
		self.W_proj = []
		



	def compute(self, X, Z):
		for q in X:
			self.projections.append(projection(self.W_proj, q.reshape(1,-1), self.mean))

	def make_prediction(self, X):
		Dist_min = np.finfo('float').max
		Class_min = -1
		P = projection(self.W_proj, X.reshape(1,-1), self.mean)
		for num in range(len(self.projection)):
			distance = self.metric_dist(self.projection[num], P)
			if distance < Dist_min:
				Dist_min = distance
				Class_min = self.Z[num]
		return Class_min

if __name__ == '__main__':
    
    [X,Z] = read_images(sys.argv[1])
    
    numcorrect=0
    numtotal=0
    for q in range(len(X)):
	    tempX=X[:q]+X[q+1:]
	    tempZ=Z[:q]+Z[q+1:]
	    model = Fisherfaces(tempX, tempZ)
	    ans=model.predict(X[q])
	    print "image ",q+1," expected image # =", Z[q], "   ", "predicted image # =", ans
	    if Z[q]==ans:
		numcorrect+=1
	    numtotal+=1
    print "The total % is: ", (1.0*numcorrect)/numtotal

    
    
