from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from dataset import *
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from model2 import VNet
import argparse
from torch.autograd import Variable
import time
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='/', help='path to dataset')
parser.add_argument('--dataset', default='DRIVE', help='dataset')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--isresume', default=True)
parser.add_argument('--output_path', default='n_exp_d1/', type=str)
parser.add_argument('--performance', default='train_on_d_test_on_d.txt', type=str)

parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--iscuda'  , default=True, help='enables cuda')
parser.add_argument('--useBN', default=True, help='enalbes batch normalization')
parser.add_argument('--lossfile', default='loss.mat', type=str)
parser.add_argument('--output_name', default='checkpoint.tar', type=str, help='output checkpoint filename')
parser.add_argument('--issave', default=False, type=bool)
args = parser.parse_args()

############## dataset processing
dataset = DATASETtest(args.dataroot,args.dataset,stride=data_dic[args.dataset]['stride'])
test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,num_workers=args.workers, shuffle=False)
model = VNet(args)
if args.iscuda:
  model.cuda()
  cudnn.benchmark = True

if os.path.isfile(args.output_path+args.output_name):
    print("=> loading checkpoint '{}'".format(args.output_path+args.output_name))
    checkpoint = torch.load(args.output_path+args.output_name, map_location={'cuda:0': 'cuda:0'})
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']) )
else:
    print("=> no checkpoint found at '{}'".format(args.output_path+args.output_name))

model.eval()

gts = dataset.get_gts_data()
oris = dataset.get_ori_data()

masks=dataset.get_masks_data()
p1 = np.zeros((data_dic[args.dataset]['test_num'],1, data_dic[args.dataset]['paddingh'],data_dic[args.dataset]['w']))
p2 = np.zeros((data_dic[args.dataset]['test_num'],1, data_dic[args.dataset]['h']+data_dic[args.dataset]['paddingh'],data_dic[args.dataset]['paddingw']))
masks = np.concatenate((np.concatenate((masks, p1), 2), p2), 3)

def reconstruct(img):
    img1 = img[3, :, :] + img[4, :, :]
    return img1

pred_patches=[]
start=time.clock()
for i,x in enumerate(test_loader):
    y_pred,o2,o4,o6 = model(Variable(x.cuda()))
    data=y_pred.data.cpu().numpy()
    pred_patches.append(data)
pred_patches=np.concatenate(pred_patches)
pred_imgs = recompone_overlap(pred_patches, gts.shape[2],gts.shape[3], data_dic[args.dataset]['stride'])
elapsed=(time.clock()-start)
print("time used: "+str(elapsed/data_dic[args.dataset]['test_num']))
kill_border(pred_imgs, masks)

for i in range(20):
    pred=reconstruct(pred_imgs[i])
    img=np.uint8(pred*255)
    img = Image.fromarray(img,mode='L')
    img.save(args.output_path+'pred_2c_'+str(i)+'.png')
    pred=pred>0.5
    gt = np.argmax(gts[i], axis=0)
    gt=gt>2
    TP=np.logical_and(gt,pred).astype(np.uint8)
    TN=np.logical_and(np.logical_not(gt),np.logical_not(pred)).astype(np.uint8)
    FP=np.logical_and(np.logical_not(gt),pred).astype(np.uint8)
    FN=np.logical_and(gt,np.logical_not(pred)).astype(np.uint8)
    imgr=TP+FP
    imgg=TP
    imgb=TP+FN
    img=np.concatenate([imgr[:,:,np.newaxis],imgg[:,:,np.newaxis],imgb[:,:,np.newaxis]],axis=2)
    img=np.uint8(img*255)
    img = Image.fromarray(img,mode='RGB')
    img.save(args.output_path+'confusion_'+str(i)+'.png')
    ori=oris[i,0]
    ori=np.uint8(ori*255)
    ori = Image.fromarray(ori,mode='L')
    ori.save(args.output_path+'ori_img_'+str(i)+'.png')
    pred=np.argmax(pred_imgs[i],axis=0)
    img=np.uint8(pred*60)
    img = Image.fromarray(img,mode='L')
    img.save(args.output_path+'pred_mc_'+str(i)+'.png')
"""
for i in range(20):
    pred=reconstruct(pred_imgs[i])
    pred=pred>0.5
    pred=np.uint8(pred*255)
    img = Image.fromarray(pred,mode='L')
    img.save(args.output_path+'pred_'+str(i+1)+'.png')
"""
#====== Evaluate the results

#predictions only inside the FOV
index=np.argmax(gts,axis=1)[:,np.newaxis,:,:]
gts=index>2
pred_imgs0=[]
for i in range(pred_imgs.shape[0]):
    pred0=reconstruct(pred_imgs[i])
    pred_imgs0.append(pred0[np.newaxis,:,:])
pred_imgs=np.asarray(pred_imgs0)

y_scores, y_true = pred_only_FOV(pred_imgs,gts, masks)  #returns data only inside the FOV

#Area under the ROC curve
fpr, tpr, thresholds = roc_curve((y_true), y_scores)
AUC_ROC = roc_auc_score(y_true, y_scores)
# test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
print("\nArea under the ROC curve: " +str(AUC_ROC))
roc_curve =plt.figure()
plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")
plt.savefig(args.output_path+"ROC.png")

#Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
AUC_prec_rec = np.trapz(precision,recall)
print("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
prec_rec_curve = plt.figure()
plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
plt.title('Precision - Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.savefig(args.output_path+"Precision_recall.png")

#Confusion matrix
threshold_confusion = 0.5
print("\nConfusion matrix:  Costum threshold (for positive) of " +str(threshold_confusion))
y_pred = np.empty((y_scores.shape[0]))
for i in range(y_scores.shape[0]):
    if y_scores[i]>=threshold_confusion:
        y_pred[i]=1
    else:
        y_pred[i]=0
confusion = confusion_matrix(y_true, y_pred)
print(confusion)
accuracy = 0
if float(np.sum(confusion))!=0:
    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
print("Global Accuracy: " +str(accuracy))
specificity = 0
if float(confusion[0,0]+confusion[0,1])!=0:
    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
print("Specificity: " +str(specificity))
sensitivity = 0
if float(confusion[1,1]+confusion[1,0])!=0:
    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
print("Sensitivity: " +str(sensitivity))
precision = 0
if float(confusion[1,1]+confusion[0,1])!=0:
    precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
print("Precision: " +str(precision))

#Jaccard similarity index
jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
print("\nJaccard similarity score: " +str(jaccard_index))

#F1 score
F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
print("\nF1 score (F-measure): " +str(F1_score))

#Kappa
kappa = cohen_kappa_score(y_true, y_pred, labels=None, sample_weight=None)
print("\nkappa score: " +str(kappa))

#Save the results
file_perf = open(args.output_path+args.performance, 'w')
file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
                + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
                + "\nJaccard similarity score: " +str(jaccard_index)
                + "\nF1 score (F-measure): " +str(F1_score)
                +"\n\nConfusion matrix:"
                +str(confusion)
                +"\nACCURACY: " +str(accuracy)
                +"\nSENSITIVITY: " +str(sensitivity)
                +"\nSPECIFICITY: " +str(specificity)
                +"\nPRECISION: " +str(precision)
                +"\nKappa: " +str(kappa)
                )
file_perf.close()
