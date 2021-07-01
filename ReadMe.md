# BORM: Bayesian Object Relation Model for Indoor Scene Recognition  

The repository is the implementation of paper:

Liguang Zhou, Jun Cen, Xingchao Wang, Zhenglong Sun, Tin Lun Lam, Yangsheng Xu. “**BORM: Bayesian Object Relation Model for Indoor Scene Recognition  **,” Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS),  Prague, Czech Republic (Virtual),  September 27 - October 1, 2021

## Environment

1. The code has been implemented and tested on Ubuntu 16.04, python 3.6.5, PyTorch 1.1.0 (tested on NVIDIA Titan Xp with CUDA 9.0.176)
3. Clone the repository as:
```
git@github.com:hszhoushen/borm.git
```

## Data

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

For 7 classes, please enter `/RAIL_7classes`. For 14 classes, please enter `/RAIL_14classes`.

For the OM-80 model, please run
```
    python train_80obj.py
```

For the OPM-80 model, please run
```
    python train_80obj_joint.py
```

For the OM-150 model, please run

```
    python train_150obj.py
```

For the OPM-150 model, please run

```
    python train_150obj_joint.py
```

For the COPM-80 model, please run

```
    python train_combined_80obj_joint.py
```

For the COM-150 model, please run

```
    python train_combined_150obj.py
```

For the COPM-150 model, please run

```
    python train_combined_150obj_joint.py
```

For the DOPM-80 model, please enter `/data_analysis_80obj` and run

```
    python train_80obj_joint_dis.py
```

For the DOPM-150 model, please enter `/data_analysis_150obj` and run

```
    python train_150obj_joint_dis.py
```

## Evaluation on image datasets

For 7 classes, please enter `/RAIL_7classes`. For 14 classes, please enter `/RAIL_14classes`.

For the OM-80 model, please run

```
    python test_80obj.py
```

For the OPM-80 model, please run

```
    python test_80obj_joint.py
```

For the OM-150 model, please run

```
    python test_150obj.py
```

For the OPM-150 model, please run

```
    python test_150obj_joint.py
```

For the COPM-80 model, please run

```
    python test_combined_80obj_joint.py
```

For the COM-150 model, please run

```
    python test_combined_150obj.py
```

For the COPM-150 model, please run

```
    python test_combined_150obj_joint.py
```

For the DOPM-80 model, please enter `/data_analysis_80obj` and run

```
    python test_80obj_joint_dis.py
```

For the DOPM-150 model, please enter `/data_analysis_150obj` and run

```
    python test_150obj_joint_dis.py
```

To inference SUNRGBD dataset, just change the dataset route in corresponding python file.