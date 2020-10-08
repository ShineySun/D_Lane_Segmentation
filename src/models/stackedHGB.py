from .layers.Residual import Residual
import torch
import torch.nn as nn

class Hourglass(nn.Module):
    def __init__(self, n, nModules, nFeats):
        super(Hourglass, self).__init__()
        self.n = n
        self.nModules = nModules
        self.nFeats = nFeats

        _up1_, _low1_, _low2_, _low3_ = [], [], [], []
        for j in range(self.nModules):
            _up1_.append(Residual(self.nFeats, self.nFeats))
        self.low1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        for j in range(self.nModules):
            _low1_.append(Residual(self.nFeats, self.nFeats))

        if self.n > 1:
            self.low2 = Hourglass(n - 1, self.nModules, self.nFeats)
        else:
            for j in range(self.nModules):
                _low2_.append(Residual(self.nFeats, self.nFeats))
            self.low2_ = nn.ModuleList(_low2_)

        for j in range(self.nModules):
            _low3_.append(Residual(self.nFeats, self.nFeats))

        self.up1_ = nn.ModuleList(_up1_)
        self.low1_ = nn.ModuleList(_low1_)
        self.low3_ = nn.ModuleList(_low3_)

        self.up2 = nn.Upsample(scale_factor = 2)

    def forward(self, x):
        up1 = x
        for j in range(self.nModules):
            up1 = self.up1_[j](up1)

        low1 = self.low1(x)
        for j in range(self.nModules):
            low1 = self.low1_[j](low1)

        if self.n > 1:
            low2 = self.low2(low1)
        else:
            low2 = low1
            for j in range(self.nModules):
                low2 = self.low2_[j](low2)

        low3 = low2
        for j in range(self.nModules):
            low3 = self.low3_[j](low3)
        #up2 = self.up2(low3)
        up2 = nn.Upsample(size=(up1.shape[2], up1.shape[3]), mode='bilinear')(low3)

        return up1 + up2

class HourglassNet3D(nn.Module):
    def __init__(self, nStack, nModules, nFeats, nOutChannels):
        super(HourglassNet3D, self).__init__()
        self.nStack = nStack
        self.nModules = nModules
        self.nFeats = nFeats
        self.nOutChannels = nOutChannels
        self.conv1_ = nn.Conv2d(3, 64, bias = True, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.r1 = Residual(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.r4 = Residual(128, 128)
        self.r5 = Residual(128, self.nFeats)

        _hourglass, _Residual, _lin_, _tmpOut, _ll_, _tmpOut_ = [], [], [], [], [], []

        for i in range(self.nStack):
            _hourglass.append(Hourglass(4, self.nModules, self.nFeats))
            for j in range(self.nModules):
                _Residual.append(Residual(self.nFeats, self.nFeats))
            lin = nn.Sequential(nn.Conv2d(self.nFeats, self.nFeats, bias = True, kernel_size = 1, stride = 1),
                                                    nn.BatchNorm2d(self.nFeats), self.relu)
            _lin_.append(lin)
            _tmpOut.append(nn.Conv2d(self.nFeats, self.nOutChannels, bias = True, kernel_size = 1, stride = 1))
            _ll_.append(nn.Conv2d(self.nFeats, self.nFeats, bias = True, kernel_size = 1, stride = 1))
            _tmpOut_.append(nn.Conv2d(self.nOutChannels, self.nFeats, bias = True, kernel_size = 1, stride = 1))

        self.hourglass = nn.ModuleList(_hourglass)
        self.Residual = nn.ModuleList(_Residual)
        self.lin_ = nn.ModuleList(_lin_)
        self.tmpOut = nn.ModuleList(_tmpOut)
        self.ll_ = nn.ModuleList(_ll_)
        self.tmpOut_ = nn.ModuleList(_tmpOut_)

        self.deconv1 = nn.ConvTranspose2d(self.nOutChannels, self.nOutChannels//2, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.nOutChannels//2)
        self.deconv2 = nn.ConvTranspose2d(self.nOutChannels//2, self.nOutChannels//4, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.nOutChannels//4)
        self.conv2 = nn.Conv2d(self.nOutChannels//4, 1, kernel_size=5, stride=1, padding=2, bias=False)

        # self.soft = nn.Softmax(dim=1)
        # self.bn4 = nn.BatchNorm2d(self.nOutChannels//8)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, line = None):
        print("Input Shape : {}".format(x.shape))
        x_1 = self.conv1_(x)
        print("Conv1 Shape : {}".format(x_1.shape))
        x_2 = self.bn1(x_1)
        print("BN1 Shape : {}".format(x_2.shape))
        x_3 = self.relu(x_2)
        print("RELU 1 Shape : {}".format(x_3.shape))
        x_4 = self.r1(x_3)
        print("PRM 1 Shape : {}".format(x_4.shape))
        x_5 = self.maxpool(x_4)
        print("MAXPOOL 1 Shape : {}".format(x_5.shape))
        x_6 = self.r4(x_5)
        print("PRM 2 Shape : {}".format(x_6.shape))
        x_7 = self.r5(x_5)
        print("PRM 3 Shape : {}".format(x_7.shape))

        out = []

        # stacked hourglass module
        for i in range(self.nStack):
            hg = self.hourglass[i](x_7)
            if i == 4:
                hg_1 = hg
            ll = hg
            for j in range(self.nModules):
                ll = self.Residual[i * self.nModules + j](ll)
            ll = self.lin_[i](ll)
            tmpOut = self.tmpOut[i](ll)
            out.append(tmpOut)

            ll_ = self.ll_[i](ll)
            tmpOut_ = self.tmpOut_[i](tmpOut)
            x_7 = x_7 + ll_ + tmpOut_
            print("HG Shape : {}".format(x_7.shape))

        shareFeat = out[-1]
        lineOut0 = self.deconv1(shareFeat)
        print("Deconv 1 Shape : {}".format(lineOut0.shape))
        lineOut1 = self.relu(self.bn2(lineOut0))
        print("BN 2 Shape : {}".format(lineOut1.shape))
        lineOut4 = self.deconv2(lineOut1)
        print("Deconv 2 Shape : {}".format(lineOut4.shape))
        lineOut2 = self.relu(self.bn3(lineOut4))
        print("BN 3 Shape : {}".format(lineOut2.shape))
        lineOut3 = self.conv2(lineOut2)
        print("Conv 2 Shape : {}".format(lineOut3.shape))

        if line is None:
            return lineOut3

        # line_loss = nn.MSELoss()(lineOut3, line)
        # loss = line_loss
        #
        return loss, line_loss, lineOut3
        #return lineOut3, lineOut2


def createModel(opt):
    model = HourglassNet3D(opt.nStack, opt.nModules, opt.nFeats, opt.nOutChannels)

    return model
