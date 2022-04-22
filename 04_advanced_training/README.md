## Advance Training on Amazon SageMaker
Machine Learning (ML) practitioners commonly face performance and scalibilty challenges when training Computer Vision (CV) models. Your model and dataset size and complexity grows quickly. While model size and complexity can result in better accuracy, there are limits to the underlying service which, at some point, the model cannot fit into a single CPU/GPU or introduce significant bottlenecks during training. 

This is when we need to leverage advance training techniques like distributed training, debugging and monitoring to help us over come these challenges.

---

## Introduction

This module covers advance training topics like debugging and distributed training. you will get exposure to SageMaker services like SageMaker debugger and Amazon SageMaker's distributed library. [SageMaker debugger](https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html) allows you to attach a debug process to your training job. This helps you monitor your training at a much granualar time interval and automatically profiling the instance to help you identify performance bottlenecks.

While [Amazon SageMaker's distributed library](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training.html) helps you train deep learning models faster and cheaper. The [data parallel](https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel.html) feature in this library is a distributed data parallel training framework for PyTorch, TensorFlow, and MXNet. This module provides 2 examples demonstrateing how to use the SageMaker distributed data library to train a TensorFlow and PyTorch model using the [Caltech Birds (CUB 200 2011)](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and MNIST dataset.

** Note: This Notebook was tested on Data Science Kernel in SageMaker Studio**

---
## Prerequisites

To get started, download the provided Jupyter notebook and associated files to you SageMaker Studio Environment. To run the notebook, you can simply execute each cell in order. To understand what's happening, you'll need:

- Access to the SageMaker default S3 bucket. All the files related to this lab will be stored under the "cv_keras_cifar10" prefix of the bucket.
- Access to 2 p3.16xlarge GPU instances.  SageMaker distributed training library requirement, so you may need to request a service limit adjustment in your account.
- Familiarity with distributed training concept
- Familiarity with training on SageMaker
- Basic familiarity with AWS S3.
- Basic familiarity with AWS Command Line Interface (CLI) -- ideally, you should have it set up with credentials to access the AWS account you're running this notebook from.
- SageMaker Studio is preferred for the full UI integration

---

## Dataset
For the Tensorflow example, The dataset we are using is from [Caltech Birds (CUB 200 2011)](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset contains 11,788 images across 200 bird species (the original technical report can be found here). Each species comes with around 60 images, with a typical size of about 350 pixels by 500 pixels. Bounding boxes are provided, as are annotations of bird parts. A recommended train/test split is given, but image size data is not. If you plan to complete the entire workshop, please keep the file to avoid re-download and re-process the data.

For pytorch, we will use fashion mnist dataset

---

## Additional Resource:

1. [SageMaker distributed data parallel PyTorch API Specification](https://sagemaker.readthedocs.io/en/stable/api/training/smd_data_parallel_pytorch.html)
1. [Getting started with SageMaker distributed data parallel](https://sagemaker.readthedocs.io/en/stable/api/training/smd_data_parallel.html)
1. [PyTorch in SageMaker](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html)