from dataset import *
from model2 import VNet
import argparse
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.tensor
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import scipy.io as scio
from sklearn.utils import class_weight
from trans import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='/', help='path to dataset')
parser.add_argument('--dataset', default='DRIVE', help='dataset')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_epoch', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--start_epoch', type=int, default=0, help='number of epoch to start')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate, default=0.0002')
parser.add_argument('--isresume', default=True)
parser.add_argument('--output_path', default='n_exp_d1/', type=str)

parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--trainlossfile', default='trainloss.mat', type=str)
parser.add_argument('--evallossfile', default='evalloss.mat', type=str)
parser.add_argument('--output_name', default='checkpoint.tar', type=str, help='output checkpoint filename')
parser.add_argument('--issave', default=False, type=bool)
args = parser.parse_args()
if not os.path.exists(args.output_path):
  os.mkdir(args.output_path)

############## dataset processing
trans1,trans2 = get_transforms()

dataset = DATASET(args.dataroot,args.dataset,transform1=trans1,transform2=trans2)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, num_workers=args.workers, shuffle=True)
dataset1 = DATASETeval(args.dataset,dataset.eval_set_path,transform1=trans1,transform2=trans2)
eval_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batchSize, num_workers=args.workers, shuffle=True)
############## create model
model = VNet(args)

model.cuda()
cudnn.benchmark = True

trainlosses = []
evallosses = []

############## resume
if args.isresume:
    if os.path.isfile(args.output_path+args.output_name):
        print("=> loading checkpoint '{}'".format(args.output_path+args.output_name))
        checkpoint = torch.load(args.output_path+args.output_name, map_location={'cuda:0': 'cuda:0'})
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        trainlosses=checkpoint['trainlosses']
        evallosses=checkpoint['evallosses']
        print("=> loaded checkpoint (epoch {})" .format(checkpoint['epoch']) )
    else:
        print("=> no checkpoint found at '{}'".format(args.output_path+args.output_name))

def save_checkpoint(state, filename=args.output_path+args.output_name):
    torch.save(state, filename)

############## training
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,momentum=0.9,weight_decay=0.0005)
#optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=100,gamma=0.1)

def dice_loss(input, target,weight=None):
    smooth = 0.1
    num = input * target
    num = torch.sum(num)
    den1 = torch.sum(input)
    den2 = torch.sum(target)

    dice = 2 * (num + smooth) / (den1 + den2 + smooth)
    dice_total = 1 - dice

    return dice_total


def ce_loss(input, target,weight=None):
    input = input.permute(0,2,3,1).contiguous().view(-1,5)
    target = target.permute(0,2,3,1).contiguous().view(-1,5)
    _, target = torch.max(target, dim=1)

    input = torch.log(input)

    ce_loss_=F.nll_loss(input, target, weight=weight)

    return ce_loss_


class_weight=torch.FloatTensor([1,1,3,1,5]).cuda()
class_weight2=torch.FloatTensor([1,1,3,1,5]).cuda()

def train(epoch):
    model.train()

    losses1=0
    losses2=0
    for i, (x, y) in enumerate(train_loader):
        x, y_true = Variable(x), Variable(y)
        x = x.cuda()
        y_true = y_true.cuda()

        y_pred, o2, o4, o6 = model(x)

        loss10 = ce_loss(y_pred, y_true, class_weight)
        loss12 = ce_loss(o2, y_true, class_weight)
        loss14 = ce_loss(o4, y_true, class_weight2)
        loss16 = ce_loss(o6, y_true, class_weight)
        loss1 = loss10 + 0.3 * (loss12 + loss14 + loss16)

        loss2 = dice_loss(y_pred, y_true, class_weight)

        loss = loss1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses1 = losses1 + loss1.item()
        losses2 = losses2 + loss2.item()
    avrloss1 = losses1 / len(train_loader)
    avrloss2 = losses2 / len(train_loader)
    print('train: epoch: {}, ce: {}, dice: {}'.format(args.start_epoch+epoch+1,avrloss1,avrloss2))

    trainlosses.append(avrloss2)
    save_checkpoint({
        'epoch': epoch + 1 +args.start_epoch,
        'state_dict': model.state_dict(),
        'trainlosses': trainlosses,
        'evallosses':evallosses
    })
    scio.savemat(args.output_path+args.trainlossfile,{'trainloss':np.asarray(trainlosses)})

def validate(epoch):
    model.eval()

    losses1=0
    losses2=0
    for i, (x, y) in enumerate(eval_loader):
        x, y_true = Variable(x), Variable(y)
        x = x.cuda()
        y_true = y_true.cuda()

        y_pred, o2, o4, o6 = model(x)

        loss10 = ce_loss(y_pred, y_true, class_weight)
        loss12 = ce_loss(o2, y_true, class_weight)
        loss14 = ce_loss(o4, y_true, class_weight)
        loss16 = ce_loss(o6, y_true, class_weight)
        loss1 = loss10 + 0.3 * (loss12 + loss14 + loss16)

        loss2 = dice_loss(y_pred, y_true, class_weight)

        losses1 = losses1 + loss1.item()
        losses2 = losses2 + loss2.item()
    avrloss1 = losses1 / len(eval_loader)
    avrloss2 = losses2 / len(eval_loader)
    print('eval: epoch: {}, ce: {}, dice: {}'.format(args.start_epoch+epoch+1,avrloss1,avrloss2))

    evallosses.append(avrloss2)
    save_checkpoint({
        'epoch': epoch + 1 +args.start_epoch,
        'state_dict': model.state_dict(),
        'trainlosses': trainlosses,
        'evallosses':evallosses
    })
    scio.savemat(args.output_path+args.evallossfile,{'evalloss':np.asarray(evallosses)})


for epoch in range(args.num_epoch):
    train(epoch)
    validate(epoch)
    scheduler.step()



