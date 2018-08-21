import readData
import numpy as np

# A = np.reshape(range(1, 26), [5, 5])
# B = np.reshape(range(1, 10), [3, 3])
# readData.dlmwrite('test.mat', A)

C = readData.dlmread('test.mat')
print(C)