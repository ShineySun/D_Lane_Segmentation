import os
import torch
import importlib
import subprocess
import math
import numpy as np
import random
import ref


def create(opt, split):
    info = exec(opt, ref.cacheFile)
    dataset = importlib.import_module('datasets.' + opt.dataset)
    print('opt.dataset : ', opt.dataset)
    print('dataset.getInstance(info,opt,split) : ',dataset.getInstance(info, opt, split))
    return dataset.getInstance(info, opt, split)


def exec(opt, cacheFile):
    print("type : ", type(str(opt.data)))
    #opt.data = str(opt.data)
    assert os.path.exists(str(opt.data)), 'Data directory not found: ' + opt.data

    print("=> Generating list of data")


    if opt.testOnly:
        dataFile = str(ref.data_root /  'test1.txt')
    else:
        dataFile = str(ref.data_root /  'train.txt')

    with open(dataFile, 'r') as f:
        dataList = f.read().splitlines()
    dataList = [x[:-4] for x in dataList]

    #print("dataList : ", dataList)
    
    random.shuffle(dataList)
    # set train/val 4800/200

    trainImg = [opt.data / "{}_rgb.png".format(x) for x in dataList]
    trainLine = [opt.data / "{}_line.png".format(x) for x in dataList]

    if opt.testOnly:
        val_list = dataList[:10]
        train_list = dataList[10:100]

        valImg = trainImg[:10]
        valLine = trainLine[:10]
        trainImg = trainImg[10:100]
        trainLine = trainLine[10:100]
    else:
        val_list = dataList[:5000]
        train_list = dataList[5000:]

        valImg = trainImg[:5000]
        valLine = trainLine[:5000]
        trainImg = trainImg[5000:]
        trainLine = trainLine[5000:]

    numTrain = len(trainImg)
    numVal = len(valImg)

    print('#Training images: {}'.format(numTrain))
    print('#Val images: {}'.format(numVal))

    info = {'basedir': opt.data,
            'train': {
                'imagePath'  : trainImg,
                'linePath'   : trainLine
                },
            'val': {
                'imagePath'  : valImg,
                'linePath'   : valLine
                }
            }

    torch.save(info, str(cacheFile))

    return info
