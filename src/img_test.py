# python3 video_test.py --netType stackedHGB --GPUs 0 --LR 0.001 --batchSize 1


import sys
print(sys.path)
sys.path.append('/lib/python3.7/site-packages')
import opts
import math
import importlib
#from preprocess import *
import _init_paths
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import numpy as np
import datasets.transforms as t
import matplotlib.pyplot as plt
import os
from util.prob2lines import getLane
from util.postprocess import *


# from pillow import Image

# img_size = (512,256)
#loader = transforms.Compose([transforms.Scale(img_size), transforms.ToTensor()])

def image_loader(frame):
    print("**** image_loader start ****")
    print(type(frame))
    print(frame.shape)
    im_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    print("im_rgb.shape : ", im_rgb.shape)
    #im_rgb = cv2.GaussianBlur(im_rgb, (5,5), 0)
    pil_img = Image.fromarray(im_rgb)
    image = loader(pil_img).float()
    # image = Image.open(image_name)
    # print("type(image) : ", type(image))
    # image = loader(image).float()   # float tensor, double tensor
    # print("type(image) : ", type(image))
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet

    # image1 = Image.open(two)
    # image1 = loader(image1).float()
    # image1 = Variable(image1, requires_grad=True)
    # image1 = image1.unsqueeze(0)

    return image.cuda() #assumes that you're using GPU

def preprocess_img(im):
    #mean = torch.DoubleTensor([0.485, 0.456, 0.406])
    #std = torch.DoubleTensor([0.229, 0.224, 0.225])
    # print("mean data type : ", mean.dtype)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = np.asarray(im)
    im = t.normalize(im, mean, std)
    im = np.transpose(im, (2, 0, 1))
    return im

def normalize(ipt, mean, std):
    ipt[:][:][0] = (ipt[:][:][0] - mean[0]) / std[0]
    ipt[:][:][1] = (ipt[:][:][1] - mean[1]) / std[1]
    ipt[:][:][2] = (ipt[:][:][2] - mean[2]) / std[2]
    return ipt

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


    # Load previous checkpoint, if it exists
    print('=> Checking checkpoints')
    checkpoint = checkpoints.load(opt)

    # Create model
    model, optimState = models.setup(opt, checkpoint)
    model.cuda()

    criterion = criterions.setup(opt, checkpoint, model)

    ##################################################################################

    model.eval()

    #with torch.no_grad():



    file_name = "0000.png"
    input_img = cv2.imread(file_name,cv2.IMREAD_COLOR)

    tensor_img = input_img / 255.

    tensor_img = preprocess_img(input_img)

    tensor_img = torch.from_numpy(tensor_img).float()

    with torch.no_grad():
        inputData_var = Variable(tensor_img).unsqueeze(0).cuda()

        output = model.forward(inputData_var, None)

        embedding = output['embedding']
        binary_seg = output['binary_seg']

        embedding = embedding.detach().cpu().numpy()
        #embedding = np.transpose(embedding[0], (1,2,0))

        binary_seg = binary_seg.detach().cpu().numpy()
        #bin_seg_pred = np.argmax(binary_seg, axis=1)[0]

        print("embedding shape : ", embedding.shape)
        print("binary_seg shape", binary_seg.shape)

        # seg_img = np.zeros_like(input_img)
        # lane_seg_img = embedding_post_process(embedding, bin_seg_pred, 1.5)
        #
        # color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')
        #
        # for i, lane_idx in enumerate(np.unique(lane_seg_img)):
        #     seg_img[lane_seg_img == lane_idx] = color[i]
        #
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # img = cv2.resize(img, (800, 288))
        # #img = cv2.addWeighted(src1=seg_img, alpha=0.8, src2=img, beta=1., gamma=0.)



        for b in range(len(binary_seg)):
            embed_b = embedding[b]
            embed_b = np.transpose(embed_b,(1,2,0))
            print(embed_b)


            print("embed_b.shape : ", embed_b.shape)
            print("embed_b.dtype : ", embed_b.dtype)

            bin_seg_b = binary_seg[b]
            #bin_seg_b_unit = bin_seg_b*255

            print("bin_seg_b.shape", bin_seg_b.shape)
            print("bin_seg_b.dtype", bin_seg_b.dtype)

            #bin_seg_b_unit = np.transpose(bin_seg_b, (1,2,0))
            bin_seg_b_unit = np.transpose(bin_seg_b*255, (1,2,0))
            print(bin_seg_b_unit)
            bin_seg_b_unit = np.clip(bin_seg_b_unit, 0,255)
            bin_seg_b_unit = np.uint8(bin_seg_b_unit)

            print("bin_seg_b_unit.shape", bin_seg_b_unit.shape)
            print("bin_seg_b_unit.dtype", bin_seg_b_unit.dtype)

            bin_seg_b_max = np.argmax(bin_seg_b, axis=0)



            print("bin_seg_b_max.shape", bin_seg_b_max.shape)
            print("embed_b[0].shape", embed_b[:,:,0].shape)

            lane_seg_img = embedding_post_process(embed_b, bin_seg_b_max, 1.5)
            lane_seg_img_add_dim = np.expand_dims(lane_seg_img, axis=0)
            lane_seg_img_add_dim = np.transpose(lane_seg_img_add_dim, (1,2,0))
            lane_seg_img_add_dim = np.array(lane_seg_img_add_dim, dtype=np.uint8)

            embed_b_0 = np.array(embed_b[:,:,1], dtype=np.uint8)


            print("lane_seg_img.shape",lane_seg_img.shape)
            print("lane_seg_img_add_dim.shape",lane_seg_img_add_dim.shape)
            print("lane_seg_img_add_dim.dtype",lane_seg_img_add_dim.dtype)

            lane_coords = getLane.polyfit2coords_tusimple(lane_seg_img, resize_shape=(720,1280), y_px_gap=10, pts=56)

            print("lane_coords : ",lane_coords)

            for i in range(len(lane_coords)):
                lane_coords[i] = sorted(lane_coords[i], key=lambda pair:pair[1])

            cv2.imshow("original", input_img)
            cv2.imshow("embed_b", bin_seg_b_unit)
            cv2.imshow("lane_seg_img_add_dim",lane_seg_img_add_dim)
            cv2.imshow("embedding[0]", embed_b_0)
            #cv2.imshow("bin_seg_b", bin_seg_b)

            #cv2.imshow("dst",img)
            cv2.waitKey(100000)















    #img_h,img_w,_ = input_img.shape
    #print("w : ", img_w, " h : ", img_h)
    #print("input_img.shape : ", input_img.shape)

    #image = image_loader(input_img)
    #print("before :", image.shape)
    # im =  model(src_img)
    # img = im[0].cpu().detach().numpy()
    # print("img.shape : ", img.shape)
    # img = np.transpose(img, (1, 2, 0))
    # print(img.shape, ' ', type(img))
    # img = np.clip(img,0,255)
    # img = np.uint8(img)
    # print("final result : ", img.shape)
    #dst = cv2.resize(img,dsize=(512,256),interpolation=cv2.INTER_LANCZOS4)
    #dst = cv2.threshold(dst,50,255,cv2.THRESH_BINARY)
    #dst = dst*255
    #dst = np.clip(dst,0,255)

    #print(dst.shape)
    # plt.imshow(dst)
    # plt.show()
    #cv2.imshow("src", input_img)

    #cv2.imshow("dst",img)
    #cv2.waitKey(100000)
    #cv2.imwrite('output_img/test12_bin.png',dst)
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])




if __name__ == '__main__':
    main()
