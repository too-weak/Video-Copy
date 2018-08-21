import os
import cv2
import sys

step = 10  #每隔step帧保存一次
resize_width=227  #要保存的图像的尺寸

# 和源代码逻辑不同
def CreateDirIfNotExist(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    return False

def extractFrame(videoFile, frameDir, configFilePath):
    videoFile = open(videoFile, 'r')
    configFile = open(configFilePath, 'a')
    lines = []
    for line in videoFile.readlines():
        lines.append(line.replace('\n', ''))

    if not os.path.exists(frameDir):
        os.mkdir(frameDir)

    for line in lines:
        if not os.path.exists(line):
            print(line + " not exists")
            exit(-1)

        capture = cv2.VideoCapture(line)

        if not capture.isOpened():
            print("fail to open" + line)
            exit(-1)

        totalFrameNumber = capture.get(7) - 1  # CV_CAP_PROP_FRAME_COUNT的值是7，Number of frames in the video file.
        mediaName = os.path.basename(line)
        framePath = os.path.join(frameDir, os.path.splitext(mediaName)[0])
        if not CreateDirIfNotExist(framePath):
            print("创建视频帧目录[" + mediaName + "]失败，目录已存在")
            exit(-1)

        fps = capture.get(5)  # CV_CAP_PROP_FPS
        if fps is None:
            continue

        configFile.write(mediaName + '\n' + str(fps) + '\n')

        success = True
        idx = 0
        frameCount = 0
        while success:
            success, frame = capture.read()
            if idx % step == 0:
                path = os.path.join(framePath, str(frameCount) + '.jpg')
                size = (resize_width, resize_width)
                dst = cv2.resize(frame, size)
                cv2.imwrite(path, dst)
                frameCount += 1
            idx += 1
        capture.release()
    videoFile.close()
    configFile.close()

videoList = sys.argv[1]
FrameDir = sys.argv[2]
configDir = sys.argv[3]
extractFrame(videoList, FrameDir, configDir)