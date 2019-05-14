import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

def get_transforms():
    sometimes = lambda aug: iaa.Sometimes(0.2,aug)

    seq1 = iaa.Sequential(
        [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            sometimes(iaa.Affine(
                scale={"x":(0.8,1.2),"y":(0.8,1.2)},
                translate_percent={"x":(-0.2,0.2),"y":(-0.2,0.2)},
                rotate=(-30,30),
                shear=(-10,10),
                mode='constant',
                cval=(0,255),
            )),
            sometimes(iaa.PiecewiseAffine(scale=(0.01,0.05),
                                          nb_cols=8,
                                          nb_rows=8,
                                          mode='constant',
                                          cval=(0,255),)),
        ],
    )

    seq2 = iaa.Sequential(
        [
            iaa.SomeOf((0,1),[
                sometimes(iaa.MultiplyElementwise((0.8, 1.2))),
                sometimes(iaa.AddElementwise((-20,20))),
                sometimes(iaa.ContrastNormalization((0.8, 1.2))),
            ]),

            iaa.SomeOf((0,1),[
                iaa.OneOf([
                    iaa.GaussianBlur((0,2.0)),
                    iaa.AverageBlur(k=2),
                    iaa.MedianBlur(k=3),
                ]),
                iaa.AdditiveGaussianNoise(0,10),
                iaa.SaltAndPepper(0.01),
                iaa.ReplaceElementwise(0.05,(0,255))
            ]),
        ],
    )

    return seq1,seq2
