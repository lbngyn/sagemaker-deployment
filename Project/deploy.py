import os
import json
import argparse
import boto3
import sagemaker
import pandas as pd
from io import StringIO
from sagemaker.pytorch import PyTorchModel
from sagemaker.predictor import Predictor
from sagemaker.serializers import StringSerializer
from sagemaker.estimator import Estimator

from dotenv import load_dotenv

load_dotenv()
IAM_ROLE_NAME = os.environ['IAM_ROLE_NAME']

REGION = os.environ['AWS_REGION']
session = sagemaker.Session(boto_session=boto3.session.Session(region_name=REGION))
BUCKET_NAME = session.default_bucket()
PREFIX = os.environ['PREFIX']
ACCOUNT_ID = session.boto_session.client(
    'sts').get_caller_identity()['Account']

class StringPredictor(Predictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super().__init__(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=StringSerializer(content_type='text/plain')
        )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Deploy SageMaker model")
    parser.add_argument("--instance-type", type=str, default="ml.m5.large", 
                        help="SageMaker instance type for deployment")
    parser.add_argument("--instance-count", type=int, default=1, 
                        help="Number of instances to deploy")
    return parser.parse_args()

def get_latest_training_job_name_from_s3(bucket, key):
    """Get latest training job name from S3 CSV file."""
    s3 = boto3.client('s3')
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(data))
        latest_record = df.iloc[-1]  # Lấy dòng cuối cùng
        
        # Giả sử có cột 'training_job_name' hoặc extract từ model_data URL
        if 'training_job_name' in df.columns:
            training_job_name = latest_record['training_job_name']
        else:
            # Extract training job name from model_data URL
            model_data_url = latest_record['model_data_s3_uri']
            # model_data format: s3://bucket/prefix/training-job-name/output/model.tar.gz
            training_job_name = model_data_url.split('/')[-3]
        
        print(f"Latest training job name: {training_job_name}")
        return training_job_name
    except Exception as e:
        print(f"Error loading model_data.csv from S3: {e}")
        raise

def create_valid_endpoint_name(training_job_name):
    """Create valid endpoint name from training job name."""
    # SageMaker endpoint names must be 1-63 characters, alphanumeric and hyphens only
    endpoint_name = f"{training_job_name}-endpoint2"
    
    # Remove invalid characters and truncate if necessary
    import re
    endpoint_name = re.sub(r'[^a-zA-Z0-9-]', '-', endpoint_name)
    endpoint_name = endpoint_name[:63]  # Max 63 characters
    
    return endpoint_name

def main():
    args = parse_args()

    # Load training job name from S3 model_data.csv
    model_data_key = f"{PREFIX}/model_data.csv"
    training_job_name = get_latest_training_job_name_from_s3(BUCKET_NAME, model_data_key)

    try:
        # Attach to existing training job
        attached_estimator = Estimator.attach(training_job_name)
        print(f"Successfully attached to training job: {training_job_name}")
        
        # Create valid endpoint name
        endpoint_name = create_valid_endpoint_name(training_job_name)
        
        print(f"Deploying model to endpoint: {endpoint_name}")
        predictor = attached_estimator.deploy(
            initial_instance_count=args.instance_count,
            instance_type=args.instance_type,
            endpoint_name=endpoint_name,
            wait=True,
            enable_network_isolation=False  # <== Rất quan trọng để log hoạt động
        )
        
        # Save endpoint information
        endpoint_info = {
            'endpoint_name': predictor.endpoint_name,
            'instance_type': args.instance_type,
            'instance_count': args.instance_count,
            'training_job_name': training_job_name
        }
        
        print(f"Model deployed successfully")
        print(json.dumps(endpoint_info, indent=2))
        
        # Test the endpoint with a sample review
        test_review = 'The simplest pleasures in life are the best, and this film is one of them.'
        print(f"Testing endpoint with sample review: '{test_review}'")
        
        try:
            result = predictor.predict(test_review)
            print(f"Prediction result: {result}")
        except Exception as e:
            print(f"Error during prediction: {e}")
            print("Note: The endpoint might still be starting up. Try again in a few minutes.")
        
        return predictor.endpoint_name
        
    except Exception as e:
        print(f"Error during deployment: {e}")
        raise

if __name__ == "__main__":
    main()