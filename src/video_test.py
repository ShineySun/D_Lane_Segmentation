# python3 video_test.py --netType stackedHGB --GPUs 0 --LR 0.001 --batchSize 1


import sys
print(sys.path)
sys.path.append('/lib/python3.7/site-packages')
import opts
import math
import importlib
from preprocess import *
import _init_paths
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import numpy as np

# from pillow import Image

img_size = (320,320)
loader = transforms.Compose([transforms.Scale(img_size), transforms.ToTensor()])

def image_loader(frame):
    # print("**** image_loader start ****")
    # print(type(frame))
    im_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    im_rgb = cv2.GaussianBlur(im_rgb, (5,5), 0)
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

    try:
        DataLoader = importlib.import_module('models.' + opt.netType + '-dataloader')
        print('DataLoader1 : ', DataLoader)
    except ImportError:
        DataLoader = importlib.import_module('datasets.dataloader')
        #print('DataLoader2 : ', DataLoader)

    # Data loading
    print('=> Setting up data loader')
    trainLoader, valLoader = DataLoader.create(opt)
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

    cap = cv2.VideoCapture("output_video/TEST11.avi")

    if cap.isOpened():
        print("width : {}, height : {}".format(cap.get(3), cap.get(4)))

    video_width = int(cap.get(3))
    print(type(video_width))
    video_height = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output_video/TEST11_ADAM.avi', fourcc, 25.0, (video_width,video_height),0)

    while True:
        ret, frame = cap.read()

        if ret:
            image = image_loader(frame)
            im =  model(image)
            img = im[0][0].cpu().detach().numpy()
            img = np.clip(img,0,255)
            slice1Copy = np.uint8(img)
            print(slice1Copy.shape)
            slice1Copy = np.transpose(slice1Copy,(1,2,0))
            dst = cv2.resize(slice1Copy,dsize=(video_width,video_height),interpolation=cv2.INTER_LANCZOS4)
            print(type(dst))
            #dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

            out.write(dst)

            # add_img = cv2.hconcat([frame, dst])
            #cv2.imshow("original",frame)
            cv2.imshow('video', dst)
            #
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                 break
        else:
            print('error')
            break

    cap.release()
    cv2.destroyAllWindows()

    #PATH_IMAGE = "test.jpeg"
    #image = image_loader(PATH_IMAGE)  # float variable(tensor)  [ 1 : 3 : 320 : 320]
    # image = image[1,3,320,320]
    #print(image[0].size())


    # im : tensor class [1, 1, 320, 320]
    #im =  model(image)
    #img = im[0][0].cpu().detach().numpy()
    #dst = cv2.resize(img, dsize=(800, 448), interpolation=cv2.INTER_LANCZOS4)

    #print(type(img), img.shape)

    #cv2.imshow("img",dst)
    #cv2.waitKey(100000)
    #####################################################################################

    # The trainer handles the training loop and evaluation on validation set
    # trainer = Trainer.createTrainer(model, criterion, opt, optimState)

    # if opt.testOnly:
    #     loss = trainer.test(valLoader, 0)
    #     sys.exit()
    #
    #
    # bestLoss = math.inf
    # startEpoch = max([1, opt.epochNum])
    # #print("opt.epochNum : ", opt.epochNum)
    #
    # if checkpoint != None:
    #     startEpoch = checkpoint['epoch'] + 1
    #     bestLoss = checkpoint['loss']
    #     print('Previous loss: \033[1;36m%1.4f\033[0m' % bestLoss)
    #
    # trainer.LRDecay(startEpoch)
    # # opt.nEpochs + 1
    # for epoch in range(startEpoch, 100):
    #     trainer.scheduler.step()
    #
    #     # trainLoss = trainer.train(trainLoader, epoch)
    #     testLoss = trainer.test(valLoader, epoch)
    #
    #     bestModel = False
    #     if testLoss < bestLoss:
    #         bestModel = True
    #         bestLoss = testLoss
    #         print(' * Best model: \033[1;36m%1.4f\033[0m * ' % testLoss)
    #
    #     checkpoints.save(epoch, trainer.model, criterion, trainer.optimizer, bestModel, testLoss ,opt)
    #
    # print(' * Finished Err: \033[1;36m%1.4f\033[0m * ' % bestLoss)


if __name__ == '__main__':
    main()
