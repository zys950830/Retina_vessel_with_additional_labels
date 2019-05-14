import torch.utils.data as data
import torch
import random
from scipy.ndimage import imread
import os
import os.path
import glob
from pre_processing import my_PreProc
import numpy as np
import pickle
random.seed(10)

data_dic = {
    'DRIVE': {'train_img_path':'training/images','train_gt_path':'training/1st_manual',
             'test_img_path':'test/images','test_gt_path':'test/1st_manual','mask_path':'/../mask/',
             'train_img_suf':'*_training.tif','test_img_suf':'*_test.tif','gt_suf':'_manual1.gif','mask_suf':'_test_mask.gif',
             'h':584,'w':565,'paddingh':2,'paddingw':1,
             'test_num':20,'stride':10},
    'DRIVE3': {'train_img_path': 'training/images', 'train_gt_path': 'training/1st_manual',
              'test_img_path': 'test/images', 'test_gt_path': 'test/1st_manual', 'mask_path': '/../mask/',
              'train_img_suf': '*_training.tif', 'test_img_suf': '*_test.tif', 'gt_suf': '_manual1.gif',
              'mask_suf': '_test_mask.gif',
              'h': 584, 'w': 565, 'paddingh': 2, 'paddingw': 1,
              'test_num': 2, 'stride': 10},
    'STARE': {'train_img_path': 'training/images', 'train_gt_path': 'training/labels',
             'test_img_path': 'test/images', 'test_gt_path': 'test/labels', 'mask_path': '',
              'train_img_suf': '*.ppm', 'test_img_suf': '*.ppm', 'gt_suf': '.ah.ppm', 'mask_suf': '',
              'h': 605, 'w': 700, 'paddingh': 1, 'paddingw': 6,
              'test_num': 5, 'stride': 10},
    'CHASEDB1': {'train_img_path': 'training/images', 'train_gt_path': 'training/1st_manual',
                'test_img_path': 'testing/images', 'test_gt_path': 'testing/mannul1', 'mask_path': '',
                 'train_img_suf': '*.jpg', 'test_img_suf': '*.jpg', 'gt_suf': '_1stHO.png', 'mask_suf': '',
                 'h': 960, 'w': 999, 'paddingh': 6, 'paddingw': 7,
                 'test_num': 7, 'stride': 10},
    'HRF': {'train_img_path': 'training/images', 'train_gt_path': 'training/labels',
            'test_img_path': 'testing/images', 'test_gt_path': 'testing/labels', 'mask_path': '/../masks/',
            'train_img_suf':'*.*','test_img_suf':'*.*','gt_suf':'.tif','mask_suf':'_mask.tif',
            'h':2336,'w':3504,'paddingh':40,'paddingw':32,
            'test_num':11,'stride':40},
}

def get_img_id(filename,dataset):
    if dataset == 'DRIVE':
        return filename[0:2]
    if dataset == 'STARE':
        return filename[0:6]
    if dataset == 'CHASEDB1':
        return filename[0:9]
    if dataset == 'HRF':
        return filename.split('.')[0]
    assert 0

class DATASET(data.Dataset):
    def __init__(self, root, name, transform1=None,transform2=None):
        self.img_patches=[]
        self.gt_patches=[]
        self.num_patches_per_img=1000
        self.patch_size=96
        self.transform1 = transform1
        self.transform2 = transform2
        self.train_set_path,self.eval_set_path = make_train_dataset(root, name)
        self.img_patches,self.gt_patches=extract_random(name,self.train_set_path,self.patch_size,self.num_patches_per_img)

    def __getitem__(self, idx):
        img = self.img_patches[idx]
        gt = self.gt_patches[idx]

        img,gt = trans_img(img,gt,self.transform1,self.transform2)

        img=torch.from_numpy(np.transpose(img,(2,0,1))/255.0).float()
        gt=torch.from_numpy(np.transpose(gt,(2,0,1))/255.0).float()

        return img, gt

    def __len__(self):
        return len(self.img_patches)

class DATASETtest(data.Dataset):
    def __init__(self, root, name, stride):
        self.name = name
        self.img_patches=[]
        self.gt_patches = []
        self.images = []
        self.gts = []
        self.patch_size=96
        self.stride=data_dic[name]['stride']
        self.test_set_path = make_test_dataset(root, name)
        self.img_patches,self.gt_patches,self.images,self.gts=extract_ordered_overlap(name,self.test_set_path,self.patch_size,self.stride)

    def __getitem__(self, idx):
        img = self.img_patches[idx]
        img=torch.from_numpy(img/255.0).float()
        return img

    def __len__(self):
        return self.img_patches.shape[0]

    def get_masks_data(self):
        images = []
        h=data_dic[self.name]['h']
        w=data_dic[self.name]['w']
        for (fimg, _) in self.test_set_path:
            if self.name == 'DRIVE' or self.name == 'DRIVE3' or self.name == 'HRF':
                fname = os.path.basename(fimg)
                fpath = os.path.dirname(fimg)
                fmask = fpath + data_dic[self.name]['mask_path'] + get_img_id(fname,self.name) + data_dic[self.name]['mask_suf']
                img = imread(fmask)
                img = img / 255.0
                if len(img.shape) == 3:
                    if img.shape[0] < 10:
                        img = img[0, :, :]
                    if img.shape[2] < 10:
                        img = img[:, :, 0]
                img = img[np.newaxis, :, :]
            if self.name == 'STARE':
                img = np.zeros([1,h,w])
                ori = imread(fimg)
                for x in range(h):
                    for y in range(w):
                        if ori[x,y,0] > 35:
                            img[0,x,y] = 1
            if self.name == 'CHASEDB1':
                img = np.zeros([1,h,w])
                ori = imread(fimg)
                for x in range(h):
                    for y in range(w):
                        if ori[x,y,0] > 20:
                            img[0,x,y] = 1
            images.append(img)

        return np.asarray(images)

    def get_gts_data(self):
        return self.gts

    def get_ori_data(self):
        return self.images/255.0

class DATASETeval(data.Dataset):
    def __init__(self, name, eval_set_path, transform1=None,transform2=None):
        self.img_patches=[]
        self.gt_patches=[]
        self.num_patches_per_img=1000
        self.patch_size=96
        self.eval_set_path = eval_set_path
        self.transform1 = transform1
        self.transform2 = transform2
        self.img_patches,self.gt_patches=extract_random(name,self.eval_set_path,self.patch_size,self.num_patches_per_img)

    def __getitem__(self, idx):
        img = self.img_patches[idx]
        gt = self.gt_patches[idx]

        img,gt = trans_img(img,gt,self.transform1,self.transform2)

        img=torch.from_numpy(np.transpose(img,(2,0,1))/255.0).float()
        gt=torch.from_numpy(np.transpose(gt,(2,0,1))/255.0).float()

        return img, gt

    def __len__(self):
        return len(self.img_patches)

def trans_img(img, gt, trans1, trans2):
    img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
    gt = np.transpose(gt, (1, 2, 0)).astype(np.uint8)

    both = np.concatenate((img, gt), axis=2)
    both = trans1.augment_image(both)
    both = np.split(both, [1, 6], axis=2)
    gt = both[1]
    img = trans2.augment_image(both[0])

    return img, gt

def make_train_dataset(root,name):
    dataset = []

    imdir = os.path.join(root, name, data_dic[name]['train_img_path'])
    gtdir = os.path.join(root, name, data_dic[name]['train_gt_path'])
    for fName in glob.glob(os.path.join(imdir, data_dic[name]['train_img_suf'])):
        fGT = os.path.basename(fName)
        fGT = get_img_id(fGT,name) + data_dic[name]['gt_suf']
        dataset.append( [os.path.join(imdir, fName), os.path.join(gtdir, fGT)] )

    eval_set_path = random.sample(dataset,1)
    for item in eval_set_path:
        dataset.remove(item)
    return dataset,eval_set_path

def make_test_dataset(root,name):
    dataset = []

    imdir = os.path.join(root, name, data_dic[name]['test_img_path'])
    gtdir = os.path.join(root, name, data_dic[name]['test_gt_path'])
    for fName in glob.glob(os.path.join(imdir, data_dic[name]['test_img_suf'])):
        fGT = os.path.basename(fName)
        fGT = get_img_id(fGT,name) + data_dic[name]['gt_suf']
        dataset.append( [os.path.join(imdir, fName), os.path.join(gtdir, fGT)] )
    return dataset

def extract_ordered_overlap(name,set_path, patch_size,stride):
    images=[]
    gts=[]
    for img_path, gt_path in set_path:
        img = imread(img_path).transpose((2,0,1)).astype(np.float32)
        p1=np.zeros((3,data_dic[name]['paddingh'],data_dic[name]['w']))
        p2 = np.zeros((3, data_dic[name]['paddingh']+data_dic[name]['h'],data_dic[name]['paddingw']))
        img = np.concatenate((np.concatenate((img,p1),1),p2),2)
        images.append(img)

        gt = pickle.load(open(gt_path, 'r'))['new_gt']
        p1=np.zeros((5,data_dic[name]['paddingh'],data_dic[name]['w']))
        p1[0, :, :] = 1
        p2 = np.zeros((5, data_dic[name]['paddingh']+data_dic[name]['h'],data_dic[name]['paddingw']))
        p2[0, :, :] = 1
        gt = np.concatenate((np.concatenate((gt,p1),1),p2),2)
        gts.append(gt)

    images = my_PreProc(np.asarray(images))
    gts = np.asarray(gts)

    img_h=img.shape[1]
    img_w=img.shape[2]
    N_patches_img = ((img_h - patch_size) // stride + 1) * ((img_w - patch_size) // stride + 1)
    N_patches_tot = N_patches_img * len(set_path)
    patches = np.empty((N_patches_tot,images.shape[1],patch_size,patch_size))
    patches_masks = np.empty((N_patches_tot, gt.shape[0], patch_size, patch_size))
    iter_tot = 0
    for i in range(len(set_path)):
        for h in range((img_h-patch_size)//stride+1):
            for w in range((img_w-patch_size)//stride+1):
                patch = images[i,:,h*stride:(h*stride)+patch_size,w*stride:(w*stride)+patch_size]
                patch_mask = gts[i,:,h*stride:(h*stride)+patch_size,w*stride:(w*stride)+patch_size]
                patches[iter_tot]=patch
                patches_masks[iter_tot]=patch_mask
                iter_tot +=1
    assert (iter_tot==N_patches_tot)

    return patches,patches_masks,images,gts

def recompone_overlap(preds, img_h,img_w, stride):
    assert (len(preds.shape)==4)
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h-patch_h)//stride+1
    N_patches_w = (img_w-patch_w)//stride+1
    N_patches_img = N_patches_h * N_patches_w
    assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
    full_prob = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))
    full_sum = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))
    k = 0
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride+1):
            for w in range((img_w-patch_w)//stride+1):
                full_prob[i,:,h*stride:(h*stride)+patch_h,w*stride:(w*stride)+patch_w]+=preds[k]
                full_sum[i,:,h*stride:(h*stride)+patch_h,w*stride:(w*stride)+patch_w]+=1
                k+=1

    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0)
    final_avg = full_prob/full_sum

    assert(np.max(final_avg)<=1.0)
    assert(np.min(final_avg)>=0.0)
    return final_avg

def extract_random(name,set_path, patch_size,num_patch_per_img):
    images=[]
    gts=[]
    for img_path, gt_path in set_path:
        img = imread(img_path).transpose((2,0,1)).astype(np.float32)
        p1=np.zeros((3,data_dic[name]['paddingh'],data_dic[name]['w']))
        p2 = np.zeros((3, data_dic[name]['paddingh']+data_dic[name]['h'],data_dic[name]['paddingw']))
        img = np.concatenate((np.concatenate((img,p1),1),p2),2)
        images.append(img)

        gt = pickle.load(open(gt_path, 'r'))['new_gt']
        p1=np.zeros((5,data_dic[name]['paddingh'],data_dic[name]['w']))
        p1[0, :, :] = 1
        p2 = np.zeros((5, data_dic[name]['paddingh']+data_dic[name]['h'],data_dic[name]['paddingw']))
        p2[0, :, :] = 1
        gt = np.concatenate((np.concatenate((gt,p1),1),p2),2)
        gts.append(gt)

    images = my_PreProc(np.asarray(images))
    gts=np.asarray(gts)*255.0

    patches = np.empty((len(set_path)*num_patch_per_img,images.shape[1],patch_size,patch_size))
    patches_masks = np.empty((len(set_path)*num_patch_per_img,gt.shape[0],patch_size,patch_size))
    img_h=img.shape[1]
    img_w=img.shape[2]

    iter_tot = 0
    for i in range(len(set_path)):
        k=0
        while k <num_patch_per_img:
            x_center = random.randint(0+int(patch_size/2),img_w-int(patch_size/2))
            y_center = random.randint(0+int(patch_size/2),img_h-int(patch_size/2))
            patch = images[i,:,y_center-int(patch_size/2):y_center+int(patch_size/2),x_center-int(patch_size/2):x_center+int(patch_size/2)]
            patch_mask = gts[i,:,y_center-int(patch_size/2):y_center+int(patch_size/2),x_center-int(patch_size/2):x_center+int(patch_size/2)]

            patches[iter_tot]=patch
            patches_masks[iter_tot]=patch_mask
            iter_tot +=1
            k+=1

    return patches, patches_masks

def pred_only_FOV(data_imgs,data_masks,original_imgs_border_masks):
    assert (len(data_imgs.shape)==4 and len(data_masks.shape)==4)
    assert (data_imgs.shape[0]==data_masks.shape[0])
    assert (data_imgs.shape[2]==data_masks.shape[2])
    assert (data_imgs.shape[3]==data_masks.shape[3])
    height = data_imgs.shape[2]
    width = data_imgs.shape[3]
    new_pred_imgs = []
    new_pred_masks = []
    for i in range(data_imgs.shape[0]):
        for x in range(width):
            for y in range(height):
                if inside_FOV(i,x,y,original_imgs_border_masks)==True:
                    new_pred_imgs.append(data_imgs[i,:,y,x])
                    new_pred_masks.append(data_masks[i,:,y,x])
    new_pred_imgs = np.asarray(new_pred_imgs)
    new_pred_masks = np.asarray(new_pred_masks)
    return new_pred_imgs, new_pred_masks

def inside_FOV(i, x, y, masks):
    assert (len(masks.shape)==4)
    assert (masks.shape[1]==1)
    if (x >= masks.shape[3] or y >= masks.shape[2]):
        return False
    if (masks[i,0,y,x]>0):
        return True
    else:
        return False


def kill_border(data, original_imgs_border_masks):
    assert (len(data.shape)==4)
    height = data.shape[2]
    width = data.shape[3]
    for i in range(data.shape[0]):
        for x in range(width):
            for y in range(height):
                if inside_FOV(i,x,y,original_imgs_border_masks)==False:
                    data[i,:,y,x]=0.0