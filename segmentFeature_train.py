import os
import numpy as np
import scipy.io as sio
import PCA_ksvd_max

configFile = 'configDir\config.txt'
mainDir = 'out\\fc6\\frameDir\\'
outDir = 'segment\\'
inputMat = 'trainmodel.mat'

def segmengFeature_train(U:np.ndarray, meanX, coef, whiten):
    name = list()
    format = list()
    fps = list()
    config = open(configFile, 'r')
    line = config.readline()
    while len(line) > 0:
        name.append(os.path.splitext(line)[0])
        format.append(os.path.splitext(line)[1].replace('\n', ''))
        line = config.readline()
        fps.append(float(line))
        line = config.readline()

    if not os.path.isdir(outDir):
        os.mkdir(outDir)

    subdir = os.listdir(mainDir)
    count = 0
    nameList = []
    total = []
    G = np.matmul(U, np.transpose(U))

    for item in subdir:
        if os.path.isdir(item):
            continue
        datPath = os.path.join(mainDir, item)
        outdatPath = os.path.join(outDir, os.path.splitext(item)[0])

        count += 1
        index = name.index(os.path.splitext(item)[0])
        res = PCA_ksvd_max.PCA_ksvd_max(datPath, outdatPath, fps[index], meanX, coef, whiten, U, G)
        # res = PCA_ksvd_max(datPath, outdatPath, fps[index], meanX.tolist(), coef.tolist(), whiten.tolist(), U, G.tolist())
        nameList.append(item)
        total.append(total[-1]+len(res[0]))

    total = np.split(total, [1], 1)[1]
    sio.savemat(outDir + 'all.mat', {'namelist': nameList, 'total':total})


# test = sio.loadmat('data.mat')
# params = test['params']
# U = params['U']
# whiten = params['whiten']
# coef = params['coef']
# meanX = params['meanX']
# a = 10