# PyTorch-From-Zero-to-All

Welcome to Season 2 of Deep Learning for All, Made by Everyone.

## Getting Started

You can start learning with slides and videos at the link below.

* PowerPoint Slide: https://pan.baidu.com/s/1b8-Y-WzHF1131LR8a2oH3w : Password: rfr2
* Custom Data: https://pan.baidu.com/s/1s8OM9EIBOzO95MgQ1dHqJw : Password: g6f9

### Install Requirements

```bash
pip install -r Requirements.txt
```

Install PyTorch from website: [https://pytorch.org/](https://pytorch.org/)


---

## PyTorch

PyTorch From Zero to All

All codes are written based on PyTorch 1.0.0.

## Contributions/Comments

We always welcome your participation. Leave comments or pull requests

We always welcome your comments and pull requests

## Contents

### PART 1: Machine Learning & PyTorch Basic

* Lesson-01_Tensor_Manipulation.ipynb
* Lesson-02_Linear_Regression.ipynb
* Lesson-03_Minimizing_Cost.ipynb
* Lesson-04_1_Multivariable_Linear_Regression.ipynb
* Lesson-04_2_Load_Data.ipynb
* Lesson-05_Logistic_Classification.ipynb
* Lesson-06_1_Softmax_Classification.ipynb
* Lesson-06_2_Fancy_Softmax_Classification.ipynb
* Lesson-07_1_Tips.ipynb
* Lesson-07_2_Mnist_Introduction.ipynb

### PART 2: Neural Network

* Lesson-08_1_Xor.ipynb
* Lesson-08_2_Xor_Nn.ipynb
* Lesson-08_3_Xor_Nn_Wide_Deep.ipynb
* Lesson-08_4_Mnist_Back_Prop.ipynb
* Lesson-09_1_Mnist_Softmax.ipynb
* Lesson-09_2_Mnist_Nn.ipynb
* Lesson-09_3_Mnist_Nn_Xavier.ipynb
* Lesson-09_4_Mnist_Nn_Deep.ipynb
* Lesson-09_5_Mnist_Nn_Dropout.ipynb
* Lesson-09_6_Mnist_Batchnorm.ipynb
* Lesson-09_7_Mnist_Nn_Selu(Wip).ipynb

### PART 3: Convolutional Neural Network

* Lesson-10_1_Mnist_Cnn.ipynb
* Lesson-10_2_Mnist_Deep_Cnn.ipynb
* Lesson-10_3_1_Visdom-Example.ipynb
* Lesson-10_3_2_Mnist-Cnn With Visdom.ipynb
* Lesson-10_4_1_Imagefolder_1.ipynb
* Lesson-10_4_2_Imagefolder_2.ipynb
* Lesson-10_5_1_Advance-Cnn(Vgg).ipynb
* Lesson-10_5_2_Aadvance-Cnn(Vgg_Cifar10).ipynb
* Lesson-10_6_1_Advance-Cnn(Resnet).ipynb
* Lesson-10_6_2_Advance-Cnn(Resnet_Cifar10).ipynb

### PART 4: Recurrent Neural Network

* Lesson-11_1_Rnn_Basics.ipynb
* Lesson-11_2_1_Charseq.ipynb
* Lesson-11_2_2_Hihello.ipynb
* Lesson-11_3_Longseq.ipynb
* Lesson-11_4_Timeseries.ipynb
* Lesson-11_5_Seq2Seq.ipynb
* Lesson-11_6_Packedsequence.ipynb

### Running PyTorch moodel on multiple GPUs
This repo provides test codes for running PyTorch model using multiple GPUs. 

You can find the environment setup for mutiple GPUs on this [repo](https://github.com/JiahongChen/Set-up-deep-learning-frameworks-with-GPU-on-Google-Cloud-Platform).

## How to make your code run on multiple GPUs
You only need to warp your model using ```torch.nn.DataParallel``` function:
```
model = nn.DataParallel(model)
```
You may check codes [here](https://github.com/JiahongChen/multiGPU/blob/master/testMultiGPU.py) to test your multiple GPU environment. These codes are mainly from this [tutorial](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html).


Sample codes to run *deep learning* model are provided [in this folder](https://github.com/JiahongChen/multiGPU/tree/master/MCD_multi_GPU), which replicates the paper [Maximum Classifier Discrepancy for Unsupervised Domain Adaptation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.pdf).

## Error: 'DataParallel' object has no attribute 'xxx'
Instead of using model.xxx, access the model attributes by model.module.xxx.

[ref: https://discuss.pytorch.org/t/how-to-reach-model-attributes-wrapped-by-nn-dataparallel/1373]


### Summary

* Pytorch Neural Programming 01.ipynb
* Pytorch Neural Programming 02.ipynb
* Pytorch Neural Programming 03.ipynb
* CSDN Blog (Chinese Version): https://blog.csdn.net/weixin_48367136
