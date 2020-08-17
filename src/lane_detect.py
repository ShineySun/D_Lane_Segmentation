# python3 inference_unet.py --netType Unet --GPUs 0 --batchSize 1

import sys
#print(sys.path)
#sys.path.append('/lib/python3.7/site-packages')
import opts
import math
import importlib
import os

import _init_paths
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

import cv2
import numpy as np
from PIL import Image
from warper import Warper
from slidewindow import SlideWindow

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
    warper = Warper()
    slidewindow = SlideWindow()

    print(("device id: {}".format(torch.cuda.current_device())))
    print("torch.version",torch.__version__)
    print("cuda_version",torch.version.cuda)


    models = importlib.import_module('models.init')
    # print(models)
    criterions = importlib.import_module('criterions.init')
    checkpoints = importlib.import_module('checkpoints')
    Trainer = importlib.import_module('models.' + opt.netType + '-train')

    # Data loading
    print('=> Setting up data loader')

    # Load previous checkpoint, if it exists
    print('=> Checking checkpoints')
    checkpoint = checkpoints.load(opt)

    # Create model
    model, optimState = models.setup(opt, checkpoint)
    model.cuda()

    criterion = criterions.setup(opt, checkpoint, model)

    ##################################################################################

    model.eval()

    cap = cv2.VideoCapture("input_video/test.avi")

    if cap.isOpened():
        print("width : {}, height : {}".format(cap.get(3), cap.get(4)))

    video_width = int(cap.get(3))
    video_height = int(cap.get(4))

    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    # out = cv2.VideoWriter('output_video/TEST_1.avi', fourcc, 20.0, (1280,480),0)

    prev_time = 0

    fps_list = []

    while True:
        ret, frame = cap.read()

        if ret:
            cur_time = time.time()
            frame_new = cv2.resize(frame, (320,180))

            input_img = frame_new / 255.
            input_img = preprocess_img(input_img)

            # array to tensor
            input_img = torch.from_numpy(input_img).float()

            with torch.no_grad():
                inputData_var = Variable(input_img).unsqueeze(0).cuda()

                # inference
                output = model.forward(inputData_var)
                output = torch.sigmoid(output)
                #output = F.softmax(output, dim=1)




                # gpu -> cpu,  tensor -> numpy
                output = output.detach().cpu().numpy()


                #print(np.unique(output))

                output = output[0]

                output = postprocess_img(output)
                output = np.clip(output, 0, 1)
                output *= 255
                output = np.uint8(output)


                output = cv2.resize(output, (640, 360))
                output[output>80] = 255
                output[output<=80] = 0
                # output[output<=40] = 0



                #ret, thr_img = cv2.threshold(output, 50, 255, cv2.THRESH_BINARY)
                warper_img, point_img = warper.warp(output)
                ret, left_start_x, right_start_x, cf_img = slidewindow.w_slidewindow(warper_img)
                #erode_img = cv2.erode(warper_img, np.ones((5,5), np.uint8), iterations=2)
                #canny_img = cv2.Canny(warper_img, 80,120)



                end_time = time.time()
                sec = end_time - cur_time


                # add_img = cv2.hconcat([frame, output])
                # print("add_img.shape : ", add_img.shape)

                fps = 1/sec
                fps_list.append(fps)

                print("Estimated fps {0} " . format(fps))

                # out.write(add_img)

                cv2.imshow("frame",frame)
                cv2.imshow("src", warper_img)
                cv2.imshow("out_img", output)
                cv2.imshow("cf_img", cf_img)
                #cv2.imshow("canny_img", erode_img)
                # cv2.imshow("out_img2", otsu_img_2)

                key = cv2.waitKey(1) & 0xFF
                if key == 27: break
                elif key == ord('p'):
                    cv2.waitKey(-1)



if __name__ == '__main__':
    main()
