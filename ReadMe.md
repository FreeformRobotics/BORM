# BORM: Bayesian Object Relation Model for Indoor Scene Recognition  

The repository is the Pytorch implementation of IROS2021 paper: 

"BORM: Bayesian Object Relation Model for Indoor Scene Recognition"

## Environment

1. The code has been implemented and tested on Ubuntu 16.04, python 3.6.5, PyTorch 1.1.0 (tested on NVIDIA Titan Xp with CUDA 9.0.176)
3. Clone the repository as:
```
git@github.com:hszhoushen/borm.git
```

## Dataset 

1. Places365_7classes: The train/test splits can be found in /data/cenj/places365_train.
2. Places365_14classes: The train/test splits can be found in /data/cenj/places365_train_2.
3. SUNRGBD_7classes: The train/test splits can be found in /data/cenj/SUNRGBD_val. It's used to inference only to test the generaliztion of model.

Update the locations accordingly in the config file.

## Prerequisite

Before training different models, some .json files which contains the object information of your dataset should be obtained. Based on these .json files we don't have to detect object of image when training which will be certainly helpful to speed up training. There are two ways to get .json file including YOLOv3 and Scene parsing model which provide 80 classes and 150 classes dectection result respectively.

To get 80 classes .json file, please run `/yolov3/data_analysis.py`. Please specify your dataset and name of .json file in this python file.

To get 150 classes .json file, please enter `/150obj` and run

`python3 train.py --gpus GPUS --cfg config/ade20k-resnet50dilated-ppm_deepsup.yaml`

## Training

For the Places365-7 dataset, please execute the following commands

```
CUDA_VISIBLE_DEVICES=0 python train_cdopm_resnet50.py  --dataset Places365-7 --num_classes 7 --om_type cdopm_resnet18
```

For the Places365-14 dataset, please execute the following commands

```
CUDA_VISIBLE_DEVICES=0 python train_cdopm_resnet50.py  --dataset Places365-14 --num_classes 14 --om_type cdopm_resnet50 --batch-size 64 
```

## Evaluation

For the Places365-7 dataset, please execute the following commands

```
CUDA_VISIBLE_DEVICES=0 python test_cdopm_resnet50.py  --dataset Places365-7 --num_classes 7 --om_type cdopm_resnet18
```

For the Places365-14 dataset, please execute the following commands

```
CUDA_VISIBLE_DEVICES=0 python test_cdopm_resnet50.py  --dataset Places365-14 --num_classes 14 --om_type cdopm_resnet50 --batch-size 64 
```

For the SUNRGBD dataset, please execute the following commands 

```
CUDA_VISIBLE_DEVICES=0 python test_cdopm_resnet50.py --om_type cdopm_resnet18 --dataset sun --num_classes 7
```

## Pre-trained model

Pretrained model is uploaded to the google driver.

## Reference

If you find the paper or code or pre-trained models useful, please cite the following papers:

```
@InProceedings{Zhou21borm,
  author     = {Liguang Zhou and Cen Jun and Xingchao Wang and Zhenglong Sun and Tin Lun Lam and Yangsheng Xu},
  title      = {BORM: Bayesian Object Relation Model for Indoor Scene Recognition},
  booktitle  = {IEEE/RSJ International Conference on Intelligent Robots and Systems},
  year       = {2021},
  organization={IEEE}
}
```



```
@InProceedings{Miao2021ots,
  author    = {Bo Miao and Liguang Zhou and Ajmal Mian and Tin Lun Lam and Yangsheng Xu},
  title     = {Object-to-Scene: Learning to Transfer Object Knowledge to Indoor Scene Recognition},
  booktitle = {2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2021},
  organization={IEEE}
}
```



 