import os
import numpy as np
import time
import sys

from RiskModel_Densenet import ChexnetTrainer


#--------------------------------------------------------------------------------   

def runTrain():
    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'
    RSENET101='RESNET-101'
    RSENET50='r50'
    WIDERSENET50='WIDE-RESNET-50'
    EFFICIENTNETB4 = 'EFFICIENT-NET_B4'
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    # For compatibility with the existing code structure, we'll use the same directory for all splits
    # The dataset class will handle splitting internally
    pathImgTrain = cifar100_dir  # Directory containing train pickle file
    pathImgVal = cifar100_dir   # Directory containing train pickle file (will split internally)
    pathImgTest = cifar100_dir  # Directory containing test pickle file

    # ---- Neural network parameters: type of the network, is it pre-trained
    # ---- on imagenet, number of classes
    nnArchitecture = RSENET50  # Using ResNet50 as in the CLIP code
    nnIsTrained = True
    nnClassCount = 100  # CIFAR-100 has 100 classes

    # ---- Training settings: batch size, maximum number of epochs
    trBatchSize = 32  # Increased batch size for CIFAR (was 8 for medical images)
    trMaxEpoch = 350  # More epochs as in CLIP training

    # ---- Parameters related to image transforms: size of the down-scaled image, cropped image
    # Note: CIFAR images are 32x32, but we'll resize to medical-style dimensions
    imgtransResize = 256  # Resize from 32 to 256
    imgtransCrop = 224    # Then crop to 224

    pathModel = 'm-' + timestampLaunch + '.pth.tar'

    print('=== Training NN architecture = ', nnArchitecture, '===')
    print('=== Dataset: CIFAR-100 ===')
    print('=== Number of classes: 100 ===')
    print('=== Training directory: ', pathImgTrain, '===')
    
    ChexnetTrainer.train(
        pathImgTrain, 
        pathImgVal, 
        pathImgTest, 
        nnArchitecture, 
        nnIsTrained, 
        nnClassCount, 
        trBatchSize,
        trMaxEpoch, 
        imgtransResize, 
        imgtransCrop, 
        timestampLaunch, 
        10,  # val_num
        'Office31',  # store_name - changed from 'BRACS_ROI'
        '/Office31_Amazon_deepseek/saved_checkpoint.pth',  # Optional: pretrained model
        None,  # start_epoch
        False  # resume
    )

    #print('=== Testing the trained model ===')
    #print(pathModel)
    #ChexnetTrainer.test(pathImgTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize,
    #                     imgtransCrop, timestampLaunch)


# --------------------------------------------------------------------------------

def runFinetune():
    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    # ---- Path to the directory with images
    pathImgTrain = '/home/ssd0/lfy/datasets/get_distribution_hosp/val'
    pathImgVal = '/home/ssd0/lfy/datasets/get_distribution_hosp/test'
    pathImgTest = '/home/ssd0/lfy/datasets/get_distribution_hosp/test'

    # ---- Neural network parameters: type of the network, is it pre-trained
    # ---- on imagenet, number of classes
    checkpoint = '/home/ssd0/lfy/gml_project/test/chexnet1/m-19042022-110254.pth.tar'
    nnArchitecture = RSENET101
    nnIsTrained = True
    nnClassCount = 1  # 14

    # ---- Training settings: batch size, maximum number of epochs
    trBatchSize = 16
    trMaxEpoch = 100

    # ---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 256
    imgtransCrop = 224

    pathModel = 'finetune-' + timestampLaunch + '.pth.tar'

    print('=== Training NN architecture = ', nnArchitecture, '===')
    ChexnetTrainer.finetune(pathImgTrain, pathImgVal, pathImgTest, nnArchitecture, nnIsTrained, nnClassCount,
                            trBatchSize,
                            trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, checkpoint)

    print('=== Testing the trained model ===')
    ChexnetTrainer.test(pathImgTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize,
                        imgtransCrop, timestampLaunch)


# --------------------------------------------------------------------------------

def runTest():
    pathImgTest = '/home/4t/SG/Bracs_ROI/test'
    nnArchitecture = RSENET101
    nnIsTrained = True
    nnClassCount = 7  # 14
    trBatchSize = 1
    imgtransResize = 256
    imgtransCrop = 224

    pathModel = '/home/ssd0/SG/sheeraz/result_archive/Bracs_Resnet101/max_acc_66.35.pth'

    timestampLaunch = ''

    ChexnetTrainer.test(pathImgTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize,
                        imgtransCrop, timestampLaunch)


# --------------------------------------------------------------------------------

if __name__ == '__main__':
 runTrain()  # 在 chest_xray 的 trainval 上预训练
 # runFinetune()  # 在 hosp_val 上微调模型
 #  runTest()  # 没什么用，改好设置后可用来算模型准确率


