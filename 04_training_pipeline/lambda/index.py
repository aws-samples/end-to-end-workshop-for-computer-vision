import json
import boto3
import os
import pprint as pp
import uuid

s3 = boto3.client('s3')
sagemaker = boto3.client("sagemaker")

pipeline_name = os.getenv('PipelineName')
role = os.getenv('RoleArn')

def lambda_handler(event, context):
    if len(event['Records']) >0:
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']
        response = s3.get_object(Bucket=bucket, Key=key)
        
        definition_data = response["Body"].read().decode('utf-8')
        
        try:
            response = sagemaker.list_pipelines(
                    PipelineNamePrefix=pipeline_name
                )
            
            pipelines = response['PipelineSummaries']
        except Exception as e:
            return e
        
        create = False
        
        if len(pipelines)>0:
            for p in pipelines:
                if p['PipelineName'] == pipeline_name:
                    create = False
                else:
                    create = True
        # Update the pipeline
        try:
            if create:
                response = sagemaker.create_pipeline(
                        PipelineName=pipeline_name,
                        PipelineDisplayName=pipeline_name,
                        PipelineDefinition=definition_data,
                        PipelineDescription=f'pipeline from {key}',
                        ClientRequestToken=str(uuid.uuid4()),
                        RoleArn=role
                    )
                print(f'created new pipeline: {pipeline_name}')
                
            else:
                response = sagemaker.update_pipeline(
                        PipelineName=pipeline_name,
                        PipelineDisplayName=pipeline_name,
                        PipelineDefinition=definition_data,
                        PipelineDescription=f'pipeline from {key}',
                        RoleArn=role
                    )
                    
                print(f'updated pipeline: {pipeline_name}')
        except Exception as e:
            return e
        
        # Execute the pipeline
        try:
            
            response = sagemaker.start_pipeline_execution(
                PipelineName=pipeline_name,
                PipelineExecutionDisplayName=pipeline_name,
                PipelineExecutionDescription=f'pipeline from {key}',
            )
        
        except Exception as e:
            print(f'Failed starting SageMaker pipeline because: {e}')
    
        return {
            'statusCode': 200,
            'body': json.dumps(f"Launched SageMaker pipeline: {pipeline_name}")
        }
        
    return "Complete Pipeline Execution...."