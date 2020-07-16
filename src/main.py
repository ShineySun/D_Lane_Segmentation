import sys
# print(sys.path)
# sys.path.append('/lib/python3.7/site-packages')
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

# python3 main.py --netType stackedHGB --GPUs 0 --LR 0.001 --batchSize 1 --nStack 7 --optim Adam



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
        #print('DataLoader1 : ', DataLoader)
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



    # The trainer handles the training loop and evaluation on validation set
    trainer = Trainer.createTrainer(model, criterion, opt, optimState)

    if opt.testOnly:
        loss = trainer.test(valLoader, 0)
        sys.exit()


    bestLoss = math.inf
    startEpoch = max([1, opt.epochNum])
    #print("opt.epochNum : ", opt.epochNum)

    if checkpoint != None:
        startEpoch = checkpoint['epoch'] + 1
        bestLoss = checkpoint['loss']
        print('Previous loss: \033[1;36m%1.4f\033[0m' % bestLoss)
#     optimizer.step()
    trainer.LRDecay(startEpoch)
    # opt.nEpochs + 1
    for epoch in range(startEpoch, opt.nEpochs + 1):
        trainer.scheduler.step()

        #trainLoss = trainer.train(trainLoader, epoch)
        testLoss = trainer.test(valLoader, epoch)

        break

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
