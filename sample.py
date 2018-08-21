import sys
import os

def endBySlash(s:str):
    length = len(s)
    if length == 0 or not s.endswith(os.path.sep):
        s += os.path.sep
    return s


def readFileName(list:str, featureAdd:str):
    file = []
    try:
        fpFileName = open(list, 'r')
    except IOError:
        print('打开' + list + '失败')
    else:
        line = fpFileName.readline()
        while len(line) > 0:
            s = featureAdd + os.path.splitext(line)[0] + '.txt'
            file.append(s)
            fpFileName.readline()
            line = fpFileName.readline()
        fpFileName.close()
        return file



def sampling(file:[], outFilePath:str):
    try:
        outFile = open(outFilePath, 'a')
    except IOError:
        print('打开输出文件失败')
    else:
        for item in file:
            try: readIn = open(item, 'r')
            except IOError:
                print('打开' + item + '失败')
            else:
                for l in readIn.readlines():
                    if len(l) > 0:
                        outFile.write(l)
                readIn.close()
        outFile.close()



featureAdd = sys.argv[1]
if featureAdd == '--help':
    print('sample features from features\
parameters: featureAdd outFile list\n')
    exit(0)

outFile = sys.argv[2]
list = sys.argv[3]

featureAdd = endBySlash(featureAdd)

file = readFileName(list, featureAdd)

sampling(file, outFile )