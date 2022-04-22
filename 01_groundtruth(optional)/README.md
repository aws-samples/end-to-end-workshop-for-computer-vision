## Data Labeling w/ SageMaker GroundTruth

SageMaker Ground Truth (SMGT) is fully managed data labeling service in which you can launch a labeling job with just a few clicks in the console or use a single AWS SDK API call. It provides 30+ labeling workflows for computer vision and NLP use cases, and also allows you to tap into different workforce options.

![SMGT](https://docs.aws.amazon.com/sagemaker/latest/dg/images/image-classification-example.png)

This module is optional. But if you want to get some labeling practices, here are all the [SMGT Examples](https://github.com/aws/amazon-sagemaker-examples/tree/main/ground_truth_labeling_jobs).  

For Computer Vision (CV) specific examples, This [Image Classification](https://github.com/aws/amazon-sagemaker-examples/blob/master/ground_truth_labeling_jobs/from_unlabeled_data_to_deployed_machine_learning_model_ground_truth_demo_image_classification/from_unlabeled_data_to_deployed_machine_learning_model_ground_truth_demo_image_classification.ipynb) and [Object Detection](https://github.com/aws/amazon-sagemaker-examples/blob/2c2cd35c8ed389e638fe5c912e24cd00d0874874/ground_truth_labeling_jobs/ground_truth_object_detection_tutorial/object_detection_tutorial.ipynb) tutorials are good practices.

## Prerequisites

To run this notebook, you simply download the provided Jupyter notebook and execute each cell one-by-one. To understand what's happening, you'll need:

- An S3 bucket you can write to -- please provide its name in the following cell. The bucket must be in the same region as this SageMaker Notebook instance. You can also change the EXP_NAME to any valid S3 prefix. All the files related to this experiment will be stored in that prefix of your bucket.
- The S3 bucket that you use for this demo must have a CORS policy attached. To learn more about this requirement, and how to attach a CORS policy to an S3 bucket, see CORS Permission Requirement.
- Familiarity with Python and numpy.
- Basic familiarity with AWS S3,
- Basic understanding of AWS Sagemaker,
- Basic familiarity with AWS Command Line Interface (CLI) -- set it up with credentials to access the AWS account you're running this notebook from. This should work out-of-the-box on SageMaker Jupyter Notebook instances.