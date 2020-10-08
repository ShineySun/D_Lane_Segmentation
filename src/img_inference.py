# python3 inference.py --netType stackedHGB --GPUs 0 --LR 0.001 --nStack 7 --batchSize 1

import sys
#print(sys.path)
#sys.path.append('/lib/python3.7/site-packages')
import opts
import math
import importlib
from preprocess import *
import _init_paths
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

import cv2
import numpy as np
from PIL import Image

import time

def normalize(ipt, mean, std):
    ipt[:][:][0] = (ipt[:][:][0] - mean[0]) / std[0]
    ipt[:][:][1] = (ipt[:][:][1] - mean[1]) / std[1]
    ipt[:][:][2] = (ipt[:][:][2] - mean[2]) / std[2]
    return ipt

def unNormalize(ipt, mean, std):
    ipt[:][:][0] = (ipt[:][:][0] * std[0]) + mean[0]
    ipt[:][:][1] = (ipt[:][:][1] * std[1]) + mean[1]
    ipt[:][:][2] = (ipt[:][:][2] * std[2]) + mean[2]
    return ipt

def preprocess_img(im):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = np.asarray(im)
    im = normalize(im, mean, std)
    im = np.transpose(im, (2, 0, 1))
    return im

def postprocess_img(im):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = np.transpose(im, (1, 2, 0))
    im = unNormalize(im, mean, std)
    return im

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    #cudnn.benchmark = True

    opt = opts.parse()

    print(("device id: {}".format(torch.cuda.current_device())))
    print("torch.version",torch.__version__)
    print("cuda_version",torch.version.cuda)


    models = importlib.import_module('models.init')
    # print(models)
    criterions = importlib.import_module('criterions.init')
    checkpoints = importlib.import_module('checkpoints')
    Trainer = importlib.import_module('models.' + opt.netType + '-train')

    # if opt.genLine:
    #     if opt.testOnly:
    #         processData('test')
    #     else:
    #         print('Prepare train data')
    #         processData('train')

    # try:
    #     DataLoader = importlib.import_module('models.' + opt.netType + '-dataloader')
    #     print('DataLoader1 : ', DataLoader)
    # except ImportError:
    #     DataLoader = importlib.import_module('datasets.dataloader')
    #     #print('DataLoader2 : ', DataLoader)

    # Data loading
    print('=> Setting up data loader')
    #trainLoader, valLoader = DataLoader.create(opt)
    #print('opt',opt)

    # Load previous checkpoint, if it exists
    print('=> Checking checkpoints')
    checkpoint = checkpoints.load(opt)

    # Create model
    model, optimState = models.setup(opt, checkpoint)
    model.cuda()

    criterion = criterions.setup(opt, checkpoint, model)

    ##################################################################################

    model.eval()

    image = cv2.imread("0000.png")
    image = cv2.resize(image,(640,480))

    cur_time = time.time()

    input_img = image / 255.
    input_img = preprocess_img(input_img)

    # array to tensor
    input_img = torch.from_numpy(input_img).float()

    with torch.no_grad():
        inputData_var = Variable(input_img).unsqueeze(0).cuda()

        # inference
        output = model.forward(inputData_var, None)

        # gpu -> cpu,  tensor -> numpy
        output = output.detach().cpu().numpy()

        output = output[0]

        output = postprocess_img(output)
        output = np.clip(output, 0, 255)
        output = np.uint8(output)

        #output[output > 100] = 255
        #output[output <= 100] = 0

        end_time = time.time()
        sec = end_time - cur_time

        fps = 1/sec

        print("Estimated fps {0} " . format(fps))

        cv2.imshow("input", image)
        cv2.imshow("output", output)

        key = cv2.waitKey(-1) & 0xFF


if __name__ == '__main__':
    main()
