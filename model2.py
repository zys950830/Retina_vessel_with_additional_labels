import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import *


def passthrough(x):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

class LUConv(nn.Module):
    def __init__(self, nchan, elu,dilation=1,kernel_size=3,padding=1):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv2d(nchan, nchan, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu,dilation=1,kernel_size=3,padding=1):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu,dilation=dilation,kernel_size=kernel_size,padding=padding))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, inChans, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv2d(inChans, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu1(out)
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False,dilation=1,kernel_size=3,padding=1):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv2d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout2d()
        self.ops = _make_nConv(outChans, nConvs, elu, dilation=dilation,kernel_size=kernel_size,padding=padding)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False,dilation=1,kernel_size=3,padding=1):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose2d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout2d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout2d()
        self.ops = _make_nConv(outChans, nConvs, elu,dilation=dilation,kernel_size=kernel_size,padding=padding)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, outChans=2):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv2d(inChans, outChans, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(outChans)
        self.conv2 = nn.Conv2d(outChans, outChans, kernel_size=1)
        self.relu1 = ELUCons(elu, outChans)
        self.outChans = outChans
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        N,C,H,W = out.data.shape
        out = out.permute(0, 2, 3, 1).contiguous().view(-1, C)
        out = self.softmax(out)
        out = out.view(N,H,W,C).permute(0,3,1,2)

        return out

class side_output(nn.Module):
    def __init__(self,inChans,factor,padding,outpadding, outChans=5):
        super(side_output,self).__init__()
        self.conv0 = nn.Conv2d(inChans,outChans,3,1,1)
        self.transconv1 = nn.ConvTranspose2d(outChans,outChans,2*factor,factor,padding=padding,output_padding=outpadding)

    def forward(self, input):
        out=self.conv0(input)
        out=self.transconv1(out)
        return out

class VNet(nn.Module):
    def __init__(self, args, elu=True):
        super(VNet, self).__init__()
        self.args = args
        self.in_tr = InputTransition(1, 16, elu)
        self.down_tr32 = DownTransition(16, 2, elu,dilation=1,kernel_size=5,padding=2,dropout=True)
        self.down_tr64 = DownTransition(32, 2, elu,dilation=1,kernel_size=5,padding=2,dropout=True)
        self.down_tr128 = DownTransition(64, 2, elu,dilation=1,kernel_size=5,padding=2,dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu,dilation=1,kernel_size=5,padding=2,dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu,dilation=1,kernel_size=5,padding=2,dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu,dilation=1,kernel_size=5,padding=2,dropout=True)
        self.up_tr64 = UpTransition(128, 64, 2, elu,dilation=1,kernel_size=5,padding=2,dropout=True)
        self.up_tr32 = UpTransition(64, 32, 2, elu,dilation=1,kernel_size=5,padding=2,dropout=True)
        self.out_tr = OutputTransition(32, elu, 5)

        #self.side1 = side_output(32,2,1,0)
        self.side2 = side_output(64,4,2,0)
        #self.side3 = side_output(128,8,4,0)
        self.side4 = side_output(256,16,8,0)
        #self.side5 = side_output(256,8,4,0)
        self.side6 = side_output(128,4,2,0)
        #self.side7 = side_output(64,2,1,0)
        #self.side8 = nn.Sequential(nn.Conv2d(32, 1, 3, 1, 1),nn.Conv2d(1, 2, 3, 1, 1))

        self.sig = nn.Sigmoid()

        self.fuseconv = nn.Sequential(
            nn.Conv2d(20,5,5,1,2),
            nn.Softmax2d()
        )

        ## weight initialization xaiver_uniform
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0.1)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        upout256 = self.up_tr256(out256, out128)
        upout128 = self.up_tr128(upout256, out64)
        upout64 = self.up_tr64(upout128, out32)
        upout32 = self.up_tr32(upout64, out16)
        out = self.out_tr(upout32)

        #o1 = self.side1(out32)
        o2 = self.side2(out64)
        #o3 = self.side3(out128)
        o4 = self.side4(out256)
        #o5 = self.side5(upout256)
        o6 = self.side6(upout128)
        #o7 = self.side7(upout64)
        #o8 = self.side8(upout32)

        fuse = self.fuseconv(torch.cat((o2,o4,o6,out),1))

        #o1 = self.sig(o1)
        o2 = self.sig(o2)
        #o3 = self.sig(o3)
        o4 = self.sig(o4)
        #o5 = self.sig(o5)
        o6 = self.sig(o6)
        #o7 = self.sig(o7)
        #o8 = self.sig(o8)

        if self.args.issave:
            save_image(make_grid(x.data.transpose(1, 0)), self.args.output_path+'x.jpg')
            save_image(make_grid(out16.data.transpose(1, 0)), self.args.output_path+'out16.jpg')
            save_image(make_grid(out32.data.transpose(1, 0)), self.args.output_path+'out32.jpg')
            save_image(make_grid(out64.data.transpose(1, 0)), self.args.output_path+'out64.jpg')
            save_image(make_grid(out128.data.transpose(1, 0)), self.args.output_path+'out128.jpg')
            save_image(make_grid(out256.data.transpose(1, 0)), self.args.output_path+'out256.jpg')
            save_image(make_grid(upout256.data.transpose(1, 0)), self.args.output_path+'upout256.jpg')
            save_image(make_grid(upout128.data.transpose(1, 0)), self.args.output_path+'upout128.jpg')
            save_image(make_grid(upout64.data.transpose(1, 0)), self.args.output_path+'upout64.jpg')
            save_image(make_grid(upout32.data.transpose(1, 0)), self.args.output_path+'upout32.jpg')
            save_image(make_grid(out.data.transpose(1, 0)), self.args.output_path+'out.jpg')
            #save_image(make_grid(o1.data.transpose(1, 0)), 'o1.jpg')
            save_image(make_grid(o2.data.transpose(1, 0)), self.args.output_path+'o2.jpg')
            #save_image(make_grid(o3.data.transpose(1, 0)), 'o3.jpg')
            save_image(make_grid(o4.data.transpose(1, 0)), self.args.output_path+'o4.jpg')
            #save_image(make_grid(o5.data.transpose(1, 0)), 'o5.jpg')
            save_image(make_grid(o6.data.transpose(1, 0)), self.args.output_path+'o6.jpg')
            #save_image(make_grid(o7.data.transpose(1, 0)), 'o7.jpg')
            #save_image(make_grid(o8.data.transpose(1, 0)), 'o8.jpg')
            save_image(make_grid(fuse.data.transpose(1, 0)), self.args.output_path+'fuse.jpg')

        return fuse,o2,o4,o6
