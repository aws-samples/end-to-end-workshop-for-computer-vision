# Model Deployment with Sagemaker

---

# Introduction

Once your models have been built, trained, and evaluated such that you are satisfied with their performance, you would likely want to deploy them to get predictions. This lab focuses on real time predictions in three ways
* [Real time inference deployments](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html) to cloud endpoints with compute you have control over
* [Serverless inference deployments](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html) to cloud endpoints with compute provisioned and manged by Sagemaker
* [Edge deployments](https://docs.aws.amazon.com/sagemaker/latest/dg/edge.html) to devices using [AWS IoT Greengrass for ML Inferneces](https://aws.amazon.com/greengrass/ml/)

---
# Prerequisites

Download the notebook corresponding to the desired deployment type you'd want to do into your environemnt. Run it by executing each cell in order.

To understand what's happening, you'll need:  

- Access to the SageMaker default S3 bucket.
- Familiarity with Python and numpy
- Basic familiarity with AWS S3.
- Basic understanding of AWS Sagemaker.
- SageMaker Studio is preferred for the full UI integration

---

### Dataset & Inputs
The dataset we are using is from [Caltech Birds (CUB 200 2011)](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). The main components needed for running the deployment notebooks are:  

- S3 path for test image data
- S3 path for test data annotation file
- S3 path for the bird classification model

If you don't have model artifacts from previous modules, there is an `optional-prepare-data-and-model` notebook you can run to generate the necessary components. In the first cell, be sure to change the S3 prefix to the appropriate value according to the deployment type you want to do.  
The outputs from a single run of this notebook, or kept ones from previous modules can be used for any of the deployments by simply updating the S3 location in the first cell of the deployment notebook you run.



---
# Review Outputs

At the end of the deployment notbook, you will have used sample images from the test dataset to get predictions from the endpoints.