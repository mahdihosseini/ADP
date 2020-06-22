- [Introduction](#introduction)
  * [ADP](#adp)
  * [Differences from CVPR Code](#differences-from-cvpr-code)
- [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installing](#installing)
- [Quick Start](#quick-start)
- [License](#license)
- [Citing](#citing)
- [Acknowledgments](#acknowledgments)

## Introduction
This repository contains the Keras implementation of Convolutional Neural Network (CNN) training on the Atlas of Digital Pathology (ADP)

### ADP

Patch-level Annotated Database of Digital Pathology images for Histological Tissue Type (HTT) Classification, as presented by Hosseini *et al.*'s CVPR 2019 paper [Atlas of Digital Pathology: A Generalized Hierarchical Histological Tissue Type-Annotated Database for Deep Learning](http://openaccess.thecvf.com/content_CVPR_2019/html/Hosseini_Atlas_of_Digital_Pathology_A_Generalized_Hierarchical_Histological_Tissue_Type-Annotated_CVPR_2019_paper.html).

### Differences from CVPR Code
* HTTs with no training examples (i.e. N.G.A, N.G.O, N.G.E, N.G.R, N.G.T) are removed, to prevent infinite class weights
* (Optional) Log-inv-freq used instead of Inv-freq as class weights
* More models are included for training, including a tailored specific model "HistoNet" where the architecure is simplified for the purpose of training for ADP
* Additional option for Colour Agumentation is provided using either HSV or YCbCr method

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. You will download a copy of the ADP database and train/test all CNN architectures with any of the five taxonomic configurations.

### Prerequisites

Mandatory
* `python` (checked on 3.5)
* `pandas` (checked on 0.23.4)
* `keras` (checked on 2.2.4)
* `numpy` (checked on 1.16.2)
* `tensorflow` (checked on 1.13.1)

Optional
* `matplotlib` (checked on 3.0.2)

### Installing

```
cd folder/to/clone-into
git clone https://github.com/mahdihosseini/ADP.git
```

## Quick Start

First, download the separate ADP database to a local directory. To access the database, please refer to [ADP Website](http://www.dsp.utoronto.ca/projects/ADP/)

Then, open `demo_train.py`, then edit the value of `DATASET_DIR` to the location of that local directory.

Edit the `settings.csv` file as necessary. This file allows you to specify the following
* Model type/Variant: defines the CNN architecture for training. Available options are: VGG/Default, ResNet/(resnet_18 or resnet_34, or resnet_50), Xception/Xception_V1, MobileNet/V1, Inception/Default, HistoNet/Series-1.0
* Level: level of training using an of five taxonomic configurations. Available options are: L1, L2, L2+, L3, L3+
* Dataset type: the ground-truth labels set to: ADP-Release1-Flat
* Micron Resolution: resolution of images for training/validation set to: 1 (The ADP is released with 1 micron resolution)
* Downsampling Method: images are downsampled from 0.25 micron resolution to 1 micron using bicubic and set to: bicubic
* CLR: cyclical learning rate. Avaialble options are: TRUE/FALSE
* Colour Augmentation: using colour augmentation for training. Available options are: TRUE/FALSE (by default we set this to YCbCr augmentation. You can substitute HSV in learner.py line 211 to 'custom_augmentation_ycbcr' for ImageDataGenerator)
* Epoch: maximum numbe of epochs

Next, run the demo script to train
```
cd folder/to/clone-into
python demo_train.py
```

## License

This project is protected under the EULA form you will sign during the registration from [ADP Website](http://www.dsp.utoronto.ca/projects/ADP/) to access the database. Use of this repository code or any biproduct (model weights, pre-trained architecture, etc) is strictly prohibited for any commercial use under the EULA agreement. 

## Citing ##
```text
@inproceedings{hosseini2019atlas,
  title={Atlas of digital pathology: A generalized hierarchical histological tissue type-annotated database for deep learning},
  author={Hosseini, Mahdi S and Chan, Lyndon and Tse, Gabriel and Tang, Michael and Deng, Jun and Norouzi, Sajad and Rowsell, Corwyn and Plataniotis, Konstantinos N and Damaskinos, Savvas},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={11747--11756},
  year={2019}
}
```

## Acknowledgments

* @[geifmany](https://github.com/geifmany/cifar-vgg) for his implementation of VGG16 for CIFAR, which we adapted for our VGG16 implementation
