# AECT-GAN: Reconstructing CT from biplane X-rays using Auto-encoding Generative Adversarial Network

## Introduction
-----

This paper proposes a new 3D self-driven generative adversarial network model base on X2CT-GAN that can reconstruct 3D CT from two orthogonal X-rays, with significant improvements in the enhancement of 3D image boundaries, depth texture information, and contours.





## Requirements
----
1. pytorch>=0.4 versions had been tested 
2. python3.6 was tested
3. python dependencies: please see the requirements.txt file
4. CUDA8.0 and cudnn7.0 had been tested

## Installation
----
- Install Python 3.6.0
- pip install -r requirements.txt
- Install PyTorch 0.41 or above
- Make sure CUDA and CUDNN are installed
- Download data from X2CT-GAN(X2CT-GAN: Reconstructing CT from Biplanar X-Rays with Generative Adversarial Networks) 
  https://share.weiyun.com/5xRVfvP (from https://github.com/kylekma/X2CT)
- Download our model's modle_dic and Pre-processed IXI data from google cloud (https://drive.google.com/drive/folders/1VH9f1Wk0DvQt_8YwxsnMOYGYY3e8PXmA?usp=sharing)
- Download the source code and put the data file in the right location according to the code structure below
## Structure
----
```
   
AECT-GAN/:
|--model_dic/: folders include trained modesl from us
   |    |--Sig_AECT-GAN/: single view X-Ray to CT model
   |    |--Mul_AECT-GAN/: Biplanar X-Rays to CT model
|--experiment/: experiment configuration folder
   |    |--multiView2500/: multiview experiment configuration file
   |    |--singleView2500/: singleview experiment configuration file
   |
|--lib/:folder includes all the dependency source codes
   |    |--config/: folder includes the config file
   |    |--dataset/: folder includes the source code to process the data
   |    |--model/: folder includes the network definitions and loss definitions
   |    |--utils/: utility functions
   |

   |--test.py: test script that demonstrates the inference workflow and outputs the metric results
   |--train.py: training script that trains models
   |--visual.py: same working mechanism as test.py but visualizing the output instead of calculating the statistics 
   |--requirements.txt: python dependency libraries
  

```

## Demo
----

### Input Arguments
+ --ymlpath: path to the configuration file of the experiemnt
+ --gpu: specific which gpu device is used for testing, multiple devices use "," to separate, e.g. --gpu 0,1,2,3
+ --dataroot: path to the test data
+ --dataset: flag indicating data is for training, validation or testing purpose
+ --tag: name of the experiment that includes the trained model
+ --data: input dataset prefix for saving and loading purposes, e.g. LIDC256 
+ --dataset_class: input data format, e.g. single view X-Rays or multiview X-Rays, see lib/dataset/factory.py for the complete list of supported data input format
+ --model_class: flag indicating the selected model, see lib/model/factory.py for the complete list of supported models
+ --datasetfile: the file list used for testing
+ --resultdir: output path of the algorithm
+ --check_point: the selected training iteration to load the correct checkpoint of the model
+ --how_many: how many test samples will be run for visualization (useful for visual mode only)
+ --valid_datasetfile: the file list used for validation

### Test our Models

Please use the following example settings to test our model. 
 
1. **Single-view Input Parameters for Test Script：**  
python3 test.py --ymlpath=./experiment/singleview2500/d2_singleview2500.yml --gpu=0 --dataroot=./data/LIDC-HDF5-256 --dataset=test --tag=d2_singleview2500 --data=LIDC256 --dataset_class=align_ct_xray_std --model_class=SingleView-AECT-GAN --datasetfile=/data/chengsq/datah5/LIDC90G/LIDC-HDF5-256/test.txt --resultdir=/data/chengsq/AECT-GAN/model_dic/Sig_AECT-GAN --check_point=90 --how_many=3   
2. **Multi-view Input Parameters for Test Script：**  
python3 test.py --ymlpath=./experiment/multiview2500/d2_multiview2500.yml --gpu=0 --dataroot=./data/LIDC-HDF5-256 --dataset=test --tag=d2_multiview2500 --data=LIDC256 --dataset_class=align_ct_xray_views_std --model_class=MultiView-AECT-GAN --datasetfile=/data/chengsq/datah5/LIDC90G/LIDC-HDF5-256/test.txt --resultdir=/data/chengsq/AECT-GAN/model_dic/MultiView-AECT-GAN --check_point=90 --how_many=3

### Train from Scratch
Please use the following example settings to train your model. 

1. **Single-view Input Parameters for Training Script：**  
    nohup python train.py --ymlpath=.AN/experiment/singleview2500/d2_singleview2500.yml --gpu=0 --dataroot=./data/chengsq/datah5/LIDC90G/LIDC-HDF5-256 --dataset=train --tag=d2_singleview2500 --data=/data/chengsq/AECT-GAN/model_dic/Sig_AECT-GAN --dataset_class=align_ct_xray_std --model_class=SingleView-AECT-GAN --datasetfile=/data/chengsq/datah5/LIDC90G/LIDC-HDF5-256/train.txt --valid_datasetfile=/data/chengsq/datah5/LIDC90G/LIDC-HDF5-256/test.txt --valid_dataset=test &
2. **Multi-view Input Parameters for Training Script：**  
    nohup python train.py --ymlpath=./experiment/multiview2500/d2_multiview2500.yml --gpu=0 --dataroot=./data/chengsq/datah5/LIDC90G/LIDC-HDF5-256 --dataset=train --tag=d2_multiview2500 --data=/data/chengsq/AECT-GAN/model_dic/MultiView-AECT-GAN --dataset_class=align_ct_xray_views_std --model_class=MultiView-AECT-GAN --datasetfile=/data/chengsq/datah5/LIDC90G/LIDC-HDF5-256/train.txt --valid_datasetfile=/data/chengsq/datah5/LIDC90G/LIDC-HDF5-256/test.txt --valid_dataset=test &


## Acknowledgement
----
We thank the public <a href="https://github.com/kylekma/X2CT">code </a> and paper X2CT-GAN: Reconstructing CT from Biplanar X-Rays with Generative Adversarial Networks Paper that is used to build our algorithm. 

Ying, Xingde, et al. "X2CT-GAN: reconstructing CT from biplanar X-rays with generative adversarial networks." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.

