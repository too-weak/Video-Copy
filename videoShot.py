import os
import cv2


# 和源代码逻辑不同
def CreateDirIfNotExist(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    return False


FrameDir = ''
configFilePath = ''

frameFile = open(FrameDir, 'r')
configFile = open(configFilePath, 'a')
lines = []
for line in frameFile.readlines():
    lines.append(line)

if not os.path.exists(FrameDir):
    os.mkdir(FrameDir)

for line in lines:
    if not os.path.exists(line):
        print(line + " not exists")
        exit(-1)

    capture = cv2.VideoCapture(line)

    if not capture.isOpened():
        print("fail to open" + line)
        exit(-1)

    totalFrameNumber = capture.get(7)-1  #CV_CAP_PROP_FRAME_COUNT的值是7，Number of frames in the video file.
    mediaName = os.path.basename(line)
    path = FrameDir + os.path.splitext(mediaName)[1]
    if not CreateDirIfNotExist(path):
        print("创建视频帧目录[" + mediaName + "]失败")
        exit(-1)

    fps = capture.get(5)  #CV_CAP_PROP_FPS
    if fps is None:
        continue

    configFile.write(mediaName + ' ' + fps + '\n')

