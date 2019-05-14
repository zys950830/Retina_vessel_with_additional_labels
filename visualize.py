from dataset import *
from model2 import VNet
import argparse
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
import torch.tensor
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
from torch.autograd import Variable
import shutil
import scipy.io as scio
import torchvision
from torchvision.utils import *
from trans import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='/', help='path to dataset')
parser.add_argument('--dataset', default='DRIVE', help='dataset')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--isresume', default=False)
parser.add_argument('--output_path', default='n_exp_d1/', type=str)

parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--iscuda'  , default=True, help='enables cuda')
parser.add_argument('--useBN', default=True, help='enalbes batch normalization')
parser.add_argument('--lossfile', default='loss.mat', type=str)
parser.add_argument('--output_name', default='checkpoint.tar', type=str, help='output checkpoint filename')
parser.add_argument('--issave', default=True, type=bool)
args = parser.parse_args()

############## dataset processing
trans1,trans2 = get_transforms()

dataset = DATASET(args.dataroot,args.dataset,transform1=trans1,transform2=trans2)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, num_workers=args.workers, shuffle=True)

############## create model
model = VNet(args)
if args.iscuda:
  model.cuda()
  cudnn.benchmark = True

############## resume
if os.path.isfile(args.output_path+args.output_name):
    print("=> loading checkpoint '{}'".format(args.output_name))
    if args.iscuda == False:
      checkpoint = torch.load(args.output_path+args.output_name, map_location={'cuda:0':'cpu'})
    else:
      checkpoint = torch.load(args.output_path+args.output_name, map_location={'cuda:0': 'cuda:0'})
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])

    print("=> loaded checkpoint (epoch {})" .format(checkpoint['epoch']) )
else:
    print("=> no checkpoint found at '{}'".format(args.output_path+args.output_name))

model.eval()
train_loader.batch_size=1

def showImg(img, fName=''):
  img = img[0,0,:,:]
  img = Image.fromarray(np.uint8(img*255), mode='L')
  if fName:
    img.save(args.output_path+fName+'.png')
  else:
    img.show()

for i, (x,y) in enumerate(train_loader):
    if i >= 1:
        break
    y_pred, o2, o4, o6 = model(Variable(x.cuda()))

    y_pred = y_pred.cpu().data.numpy()
    y_pred = np.argmax(y_pred, axis=1)[:, np.newaxis, :, :]

    o2=o2.cpu().data.numpy()
    o2=np.argmax(o2,axis=1)[:, np.newaxis, :, :]
    o4=o4.cpu().data.numpy()
    o4=np.argmax(o4,axis=1)[:, np.newaxis, :, :]
    o6=o6.cpu().data.numpy()
    o6=np.argmax(o6,axis=1)[:, np.newaxis, :, :]
    y=np.argmax(y.numpy(),1)[:, np.newaxis, :, :]

    showImg(x.numpy(),  fName='ori_' + str(i))
    showImg(y_pred/4.0, fName='pred_' + str(i))
    showImg(y/4.0, fName='gt_' + str(i))
    showImg(o2/4.0, fName='o2_' + str(i))
    showImg(o4/4.0, fName='o4_' + str(i))
    showImg(o6/4.0, fName='o6_' + str(i))