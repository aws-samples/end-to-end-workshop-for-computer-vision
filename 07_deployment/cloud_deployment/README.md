# Cloud Deployment

---

# Overview

Use the notebook corresponding to the type of deployment you want to do:
* `real-time-inference.ipynb` for deploying to compute you specify
* `serverless-inference-deployment.ipynb` for deploying to automatically provisioned compute

In either scenario, the notebook will take you through the following flow:
* Configure constants (S3 locations) so your notebook can find the artifacts needed for deployment and testing
* initialize the TensorFlowModel and deploy to an endpoint
* Get predictions from the deployed model using sample images from the test dataset.
    * the `cv_utils` module creates wrappers around the needed steps for this