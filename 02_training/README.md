## Training a CV model on Amazon SageMaker

Training models is easy on Amazon SageMaker. You simply specify the location of your data in Amazon S3, define the type and quantity number of ML instances you need, and start training a model with just a few lines of code. Amazon SageMaker sets up a distributed compute cluster, performs the training, outputs the result to Amazon S3, and tears down the cluster when complete. 

---
## Introduction
This module is focused on SageMaker training for Computer Vision models. You will go through examples of Bring Your Own(BYO) Script training, hyperparameter tuning, and experiment tracking. In the future modules, you will see how experiment tracking can be automated through SageMaker Pipeline's native integration.

At the end of this lab, you should develop hands on experience 1) training custom CV models on Amazon SageMaker, 2) Build Automatic Model Tuning Jobs, and 3) Organize your ML experiments.

### Keras
The model used for the tensorflow\cv_hpo_keras_pipe notebook is a simple deep CNN that is based on the [Keras examples](https://www.tensorflow.org/tutorials/images/cnn). 

** Note: This Notebook was tested on Data Science Kernel in SageMaker Studio**

### Pytorch
The model used for the pytorch\cv_hpo_pytorch_pipe notebook is a simple deep CNN that is based on the [Pytorch cnn example](https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-python-sdk/pytorch_cnn_cifar10/source/cifar10.py). 

** Note: This Notebook was tested on 'Python3 (Pytorch 1.8, Python 3.6)' kernel in SageMaker Studio**

---
## Prerequisites

To get started, download the provided Jupyter notebook and associated files to you SageMaker Studio Environment. To run the notebook, you can simply execute each cell in order. To understand what's happening, you'll need:

- Access to the SageMaker default S3 bucket. All the files related to this lab will be stored under the "cv_keras_cifar10" prefix of the bucket.
- Familiarity with Python and numpy
- Basic familiarity with AWS S3.
- Basic understanding of AWS Sagemaker.
- Basic familiarity with AWS Command Line Interface (CLI) -- ideally, you should have it set up with credentials to access the AWS account you're running this notebook from.
- SageMaker Studio is preferred for the full UI integration

---

## Dataset

The dataset we are using is from [Caltech Birds (CUB 200 2011)](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset contains 11,788 images across 200 bird species. Each species comes with around 60 images, with a typical size of about 350 pixels by 500 pixels. Bounding boxes are provided, as are annotations of bird parts. A recommended train/test split is given, but image size data is not.

![Bird Dataset](statics/birds.png)

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) is one of the most popular machine learning datasets. It consists of 60,000 32x32 images belonging to 10 different classes (6,000 images per class). Here are the classes in the dataset, as well as 10 random images from each.

![cifar10](statics/CIFAR-10.png)



