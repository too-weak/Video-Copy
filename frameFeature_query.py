#!/home/baoyu/anaconda/bin/python
# #####1
import numpy as np
import matplotlib.pyplot as plt

# display plots in this notebook
# %matplotlib inline

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)  # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# #####2
# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys

caffe_root = 'D:\Libraries\caffe-windows\\'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

# sys.path.append('/home/ysp0623/ccd/caffe/python/')
# sys.path.append('/home/ysp0623/ccd/caffe/')
print(sys.path)
import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

# #####3
import os

if os.path.isfile(caffe_root + '\\models\\bvlc_reference_caffenet\\bvlc_reference_caffenet.caffemodel'):
    print('CaffeNet found.')
else:
    print('Downloading pre-trained CaffeNet model...')

# #####4
caffe.set_mode_cpu()

model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,  # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)  # use test mode (e.g., don't perform dropout)

# #####5
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

# #####6
# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(1,  # batch size
                          3,  # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

# #####my_1
import os
import time

INDIR = r'D:\Programs\copyPython\frameDir'
OUTDIR = r'D:\Programs\copyPython\out'

if not os.path.isdir(INDIR):
    print('Input dir is invalid!')

if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)
if not os.path.isdir(OUTDIR + os.sep + 'fc6'):
    os.mkdir(OUTDIR + os.sep + 'fc6')

start = time.time()
for DIR in os.walk(INDIR):
    count = len(DIR[2])

    if count != 0:
        outDir_fc6 = OUTDIR + os.sep + 'fc6' + os.sep + os.path.split(os.path.split(DIR[0])[0])[1]
        if not os.path.isdir(outDir_fc6):
            os.mkdir(outDir_fc6)

        outFile_fc6 = outDir_fc6 + os.sep + os.path.split(DIR[0])[1] + '.txt'
        feature_fc6 = []

        for i in range(0, count):
            inputDir = DIR[0] + os.sep + str(i) + '.jpg'
            image = caffe.io.load_image(inputDir)
            transformed_image = transformer.preprocess('data', image)
            # plt.imshow(image)
            # copy the image data into the memory allocated for the net
            net.blobs['data'].data[...] = transformed_image
            ### perform classification
            output = net.forward()
            feature_fc6.append(net.blobs['fc6'].data[0].copy())

        np.savetxt(outFile_fc6, feature_fc6, fmt='%.4f')

print('Done')
end = time.time()
