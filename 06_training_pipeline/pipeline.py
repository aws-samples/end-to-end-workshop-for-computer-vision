import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)

from sagemaker.workflow.steps import (
    CacheConfig, 
    ProcessingStep, 
    TrainingStep
)

from sagemaker.sklearn.processing import SKLearnProcessor

from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    FrameworkProcessor,
)

from sagemaker.inputs import TrainingInput
from sagemaker.tensorflow import TensorFlow

from sagemaker.workflow.properties import PropertyFile

from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.workflow.pipeline import Pipeline

import uuid

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.
    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name = "cv-week4-model-group",
    pipeline_name="cv-week4-pipeline",  
    base_job_prefix="cv-week4",  # Choose any name
):
    """Gets a SageMaker ML Pipeline instance
    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
        
    # Define parameters for pipeline execution
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", default_value=1
    )

    processing_instance_type = "ml.m5.xlarge"
#     processing_instance_type = ParameterString(
#         name="ProcessingInstanceType", default_value="ml.m5.xlarge"
#     )
    training_instance_count = 1
#     training_instance_count = ParameterInteger(
#         name="TrainingInstanceCount", default_value=1
#     )
    training_instance_type = "ml.c5.4xlarge"
#     training_instance_type = ParameterString(
#         name="TrainingInstanceType", default_value="ml.c5.4xlarge"
#     )

    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval"  # ModelApprovalStatus can be set to a default of "Approved" if you don't want manual approval.
    )

    input_data = ParameterString(
        name="InputDataUrl",
        default_value='s3://sagemaker-us-east-1-909708043314/cv-week4/full/data'
    )

    input_annotation = ParameterString(
        name="AnnotationFileName",
        default_value="classes.txt"
    )

    # This is a large dataset, we are only going to train a subset of the classes
    class_selection = ParameterString(
        name="ClassSelection",
        default_value="13, 17, 35, 36, 47, 68, 73, 87"
    )
    
    ## By enabling cache, if you run this pipeline again, without changing the input 
    ## parameters it will skip the training part and reuse the previous trained model
    cache_config = CacheConfig(enable_caching=True, expire_after="30d")
    
    # Processing step for split raw data into train/valid/test sets ===================
    sklearn_processor = SKLearnProcessor(base_job_name = f"{base_job_prefix}-preprocess",  # choose any name
                                        framework_version='0.20.0',
                                        role=role,
                                        instance_type=processing_instance_type,
                                        instance_count=processing_instance_count)

    output_s3_uri = f's3://{default_bucket}/{base_job_prefix}/outputs/{uuid.uuid4()}'

    step_process = ProcessingStep(
        name=f"{base_job_prefix}PreProcess",  # choose any name
        processor=sklearn_processor,
        code=os.path.join(BASE_DIR, "preprocess.py"),
        job_arguments=["--classes", class_selection,
                    "--input-data", input_annotation],
        inputs=[ProcessingInput(source=input_data, 
                destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(output_name='train_data', 
                             source="/opt/ml/processing/output/train", 
                             destination = output_s3_uri +'/train'),
            ProcessingOutput(output_name='val_data',
                             source="/opt/ml/processing/output/validation", 
                             destination = output_s3_uri +'/validation'),
            ProcessingOutput(output_name='test_data',
                             source="/opt/ml/processing/output/test", 
                             destination = output_s3_uri +'/test'),
            ProcessingOutput(output_name='manifest',
                             source="/opt/ml/processing/output/manifest", 
                             destination = output_s3_uri +'/manifest'),
        ],
        cache_config=cache_config
        )
    
    # Training step begins here =============================
    
    TF_FRAMEWORK_VERSION = '2.4.1'

    hyperparameters = {'initial_epochs':     5,
                       'batch_size':         8,
                       'fine_tuning_epochs': 20, 
                       'dropout':            0.4,
                       'data_dir':           '/opt/ml/input/data'}

    metric_definitions = [{'Name': 'loss',      'Regex': 'loss: ([0-9\\.]+)'},
                      {'Name': 'acc',       'Regex': 'accuracy: ([0-9\\.]+)'},
                      {'Name': 'val_loss',  'Regex': 'val_loss: ([0-9\\.]+)'},
                      {'Name': 'val_acc',   'Regex': 'val_accuracy: ([0-9\\.]+)'}]

    if training_instance_count > 1:
        distribution = {'parameter_server': {'enabled': True}}
        DISTRIBUTION_MODE = 'ShardedByS3Key'
    else:
        distribution = {'parameter_server': {'enabled': False}}
        DISTRIBUTION_MODE = 'FullyReplicated'

    train_in = TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
                             distribution=DISTRIBUTION_MODE)
    test_in  = TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri,
                             distribution=DISTRIBUTION_MODE)
    val_in   = TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["val_data"].S3Output.S3Uri,
                             distribution=DISTRIBUTION_MODE)

    inputs = {'train':train_in, 'test': test_in, 'validation': val_in}
    
    # Executing spot vs on demand training in parallel
    training_steps = dict()
    eval_steps = dict()
    condition_steps = dict()


    training_options = ['Spot', 'OnDemand']

    for t in training_options:
        tags = dict()
        tags['Key'] = 'TrainingType'
        tags['Value'] = t
            # Training step for generating model artifacts
        model_path = f"s3://{default_bucket}/{base_job_prefix}/output/models"
        checkpoint_s3_uri = f"s3://{default_bucket}/{base_job_prefix}/outputcheckpoints"

        if t.lower() == 'spot':
            
            estimator = TensorFlow(entry_point='train-mobilenet.py',
                                   source_dir='code',
                                   output_path=model_path,
                                   instance_type=training_instance_type,
                                   instance_count=training_instance_count,
                                   distribution=distribution,
                                   hyperparameters=hyperparameters,
                                   metric_definitions=metric_definitions,
                                   role=role,
                                   use_spot_instances=True,
                                   max_run=60*60*10,
                                   max_wait=60*60*12, # Seconds to wait for spot instances to become available
                                   checkpoint_s3_uri=checkpoint_s3_uri,
                                   framework_version=TF_FRAMEWORK_VERSION, 
                                   py_version='py37',
                                   base_job_name=base_job_prefix,
                                   script_mode=True,
                                   tags=[tags])
        else:
            estimator = TensorFlow(entry_point='train-mobilenet.py',
                           source_dir='code',
                           output_path=model_path,
                           instance_type=training_instance_type,
                           instance_count=training_instance_count,
                           distribution=distribution,
                           hyperparameters=hyperparameters,
                           metric_definitions=metric_definitions,
                           role=role,
                           framework_version=TF_FRAMEWORK_VERSION, 
                           py_version='py37',
                           base_job_name=base_job_prefix,
                           script_mode=True,
                           tags=[tags])

        step_train = TrainingStep(
            name=f"BirdClassification{t}Train",
            estimator=estimator,
            inputs=inputs,
            cache_config=cache_config
        )

        training_steps[t] = step_train
        
        # Evaluate Trained Models =======================
        pipeline_session = PipelineSession()

        script_eval = FrameworkProcessor(
            estimator_cls=TensorFlow,
            framework_version=TF_FRAMEWORK_VERSION,
            base_job_name = f"{base_job_prefix}-evaluation",
            command=['python3'],
            py_version="py37",
            role=role,
            instance_count=processing_instance_count,
            instance_type=processing_instance_type,
            sagemaker_session = pipeline_session)
        
        step_args = script_eval.run(
            code=os.path.join(BASE_DIR, "evaluation.py"),
            arguments=["--model-file", "model.tar.gz"],
            inputs=[ProcessingInput(source=step_process.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri, 
                                    destination="/opt/ml/processing/input/test"),
                    ProcessingInput(source=step_process.properties.ProcessingOutputConfig.Outputs["manifest"].S3Output.S3Uri, 
                                    destination="/opt/ml/processing/input/manifest"),
                    ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts, 
                                    destination="/opt/ml/processing/model"),
                   ],
            outputs=[
                ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
            ]
        )

        evaluation_report = PropertyFile(
            name=f"Evaluation{t}Report",
            output_name="evaluation",
            path="evaluation.json",
        )

        step_eval = ProcessingStep(
            name=f"BirdClassification{t}Eval",
            step_args = step_args,
            property_files=[evaluation_report],
            cache_config=cache_config
        )

        eval_steps[t] = step_eval
        
        # Register Models ===========================
        # Create ModelMetrics object using the evaluation report from the evaluation step
        # A ModelMetrics object contains metrics captured from a model.
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri="{}/evaluation.json".format(
                    step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"][
                        "S3Uri"
                    ]
                ),
                content_type="application/json",
            )
        )

        # Crete a RegisterModel step, which registers the model with Sagemaker Model Registry.
        step_register = RegisterModel(
            name=f"Register{t}Model",
            estimator=estimator,
            model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.t2.medium", "ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name=model_package_group_name,
            approval_status=model_approval_status,
            model_metrics=model_metrics,
        )
        
        # Condition Step to only register model if a certain level of model accuracy is reached

        cond_gte = ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step=step_eval,
                property_file=evaluation_report,
                json_path="multiclass_classification_metrics.accuracy.value",
            ),
            right=0.7,
        )

        # Create a Sagemaker Pipelines ConditionStep, using the condition above.
        # Enter the steps to perform if the condition returns True / False.
        step_cond = ConditionStep(
            name=f"BirdClassification{t}Condition",
            conditions=[cond_gte],
            if_steps=[step_register],
            else_steps=[],
        )

        condition_steps[t] = step_cond

    # Create a Sagemaker Pipeline.
    # Each parameter for the pipeline must be set as a parameter explicitly when the pipeline is created.

    # build the steps =============================
    steps = [step_process]
    for t in training_steps:
        steps.append(training_steps[t])

    for e in eval_steps:
        steps.append(eval_steps[e])

    for c in condition_steps:
        steps.append(condition_steps[c])

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
#             processing_instance_type,
            processing_instance_count,
#             training_instance_count,
#             training_instance_type,
            model_approval_status,
            input_data,
            input_annotation,
            class_selection
        ],
        steps=steps,
        sagemaker_session=sagemaker_session,
    )
    return pipeline