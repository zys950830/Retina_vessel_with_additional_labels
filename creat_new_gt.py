import glob
import os
from scipy.ndimage import imread
import numpy
import pickle
from PIL import Image

gtdir = '1st_manual'
r1 = 3
r2 = 4


for fGT in glob.glob(os.path.join(gtdir,'*_manual1.gif.composite.png')):
    img = imread(fGT)
    H,W,C = img.shape

    new_gt = numpy.zeros((H,W,5))
    for i in range(H):
        for j in range(W):
            if img[i,j,0]==0:
                new_gt[i,j] = [1,0,0,0,0]
    for i in range(H):
        for j in range(W):
            if img[i,j,0]>0 and img[i,j,1]>0:
                new_gt[i,j] = [0,0,0,0,1]
                for a in range(-3,4):
                    for b in range(-3,4):
                        if a ** 2 + b ** 2 < r1**2 and img[i+a,j+b,0]==0:
                            new_gt[i+a,j+b] = [0,0,1,0,0]
            if img[i,j,0]>0 and img[i,j,1]==0:
                new_gt[i,j] = [0,0,0,1,0]
                for a in range(-3,4):
                    for b in range(-3,4):
                        if a**2+b**2<r2**2 and img[i+a,j+b,0]==0:
                            new_gt[i+a,j+b] = [0,1,0,0,0]

    fName = os.path.basename(fGT)

    img = 60*new_gt[:,:,1]+120*new_gt[:,:,2]+180*new_gt[:,:,3]+240*new_gt[:,:,4]
    img = numpy.uint8(img)
    img = Image.fromarray(img, mode='L')
    img.save(os.path.join(gtdir,fName[0:2] + '.png'))

    new_gt = numpy.asarray(new_gt.transpose(2,0,1))
    f = open(os.path.join(gtdir,fName[0:2]+'_new_gt'+'.pkl'),'w')
    obj = {'new_gt':new_gt}
    pickle.dump(obj,f)

