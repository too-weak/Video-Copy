import os
import numpy as np
from myKsvd import *
from sklearn.decomposition import  PCA
# from ksvd import ApproximateKSVD
from segmentFeature_train import *
input = 'sample.txt'
output = 'trainmodel.mat'
dim = 256

# sampleFile = open(input, 'r')
# rawData = sampleFile.read()
rawData = np.loadtxt(input)
name = os.path.basename(input)
newData = dict()
newData[name] = rawData
vars = [name]
X = rawData

# l2 norm
for i in range(len(X)):
    X[i] = X[i]/np.linalg.norm(X[i])

pca = PCA()
pca.fit(X)
coef = np.transpose(pca.components_)
S = pca.fit_transform(X)
latent = pca.explained_variance_

cum = np.cumsum(latent, 0)

S = np.split(S, [dim],1)[0]
coef = np.split(coef, [dim], 1)[0]

Cov = np.cov(S, rowvar=False)
d = np.sqrt(np.diag(Cov))
whiten = np.transpose(d)

S = S / whiten

meanX = np.mean(X, 0)

# l2 norm
for i in range(len(S)):
    S[i] = S[i]/np.linalg.norm(S[i])

X = np.transpose(S)
for i in range(len(X)):
    X[i] = X[i].tolist()
params =  {}
params['data'] = X.tolist()
params['Tdata'] = 32
params['dictsize'] = 1024
params['iternum'] = 50
params['memusage'] = 'high'
ksvd = KSVD(1024)
U, sparsecode = ksvd.fit(X)
result = dict()
result['U'] = U
result['G'] = np.matmul(U, np.transpose(U)).tolist()
result['whiten'] = whiten.tolist()
result['coef'] = coef.tolist()
result['meanX'] = meanX.tolist()



# segmengFeature_train(U, meanX, coef, whiten)