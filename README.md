# [Deep Supervision with Additional Labels for Retinal Vessel Segmentation Task](https://link.springer.com/chapter/10.1007/978-3-030-00934-2_10) [1]

## main contributions

1. Introducing a deep supervision mechanism into U-net, which helps networkslearn a better semantically representation;
2. Separating thin vessels from thick vessels during the progress of training;
3.  Applying an edge-aware mechanism by labelling the boundary region to extraclasses, making the network focus on the boundaries of vessels and thereforeget finer edges in the segmentation resul

In our project, AUC score on DRIVE can be improved to 97.99%.

## How to use

In our project, we generate new ground truth firstly. Using the ImageJ script give by this [link](http://www.cs.put.poznan.pl/kkrawiec/?n=Site.2015IEEETMI), we seperate thick vessels from thin vessels. Then we label the boundary region by creat_new_gt.py. We offer the generated new ground truth here. Each image is saved in a .pkl file, in the data format of 5 * H * W.

For training, please run main.py; for testing, please run eval.py.

Please note modify the file path to dataset.

## Reference

We use some source code from [vnet.pytorch](https://github.com/mattmacy/vnet.pytorch) and [retina-unet](https://github.com/orobix/retina-unet). Thanks a lot for their help. 

This page is maintained by [ZHANG Yishuo](ys.zhang@connect.ust.hk)

[1] Zhang, Yishuo, and Albert CS Chung. "Deep supervision with additional labels for retinal vessel segmentation task." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2018.
