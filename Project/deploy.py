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

def get_latest_model_data_from_s3(bucket, key):
    s3 = boto3.client('s3')
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(data))
        latest_record = df.iloc[-1]  # Lấy dòng cuối cùng
        model_data_url = latest_record['model_data']  # Giả sử cột tên model_data
        print(f"Latest model_data from CSV: {model_data_url}")
        return model_data_url
    except Exception as e:
        print(f"Error loading model_data.csv from S3: {e}")
        raise

def main():
    args = parse_args()

    # Load model_data path from S3 model_data.csv (lấy phần tử cuối cùng)
    model_data_key = f"{PREFIX}/model_data.csv"  # đường dẫn file trên S3
    model_data = get_latest_model_data_from_s3(BUCKET_NAME, model_data_key)

    # Create a PyTorch model
    model = PyTorchModel(
        model_data=model_data,
        role=IAM_ROLE_NAME,
        # framework_version="1.13.1",
        # py_version="py39",
        # entry_point='predict.py',
        # source_dir='serve',
        image_uri=f'{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/my-app:latest',
        predictor_cls=StringPredictor
    )
    
    # Deploy the model
    print(f"Deploying model...")
    predictor = model.deploy(
        initial_instance_count=args.instance_count,
        instance_type=args.instance_type,
    )
    
    # Save endpoint information
    endpoint_info = {
        'endpoint_name': predictor.endpoint_name,
        'instance_type': args.instance_type,
        'instance_count': args.instance_count,
    }
    
    print(f"Model deployed successfully")
    print(endpoint_info)
    
    # Test the endpoint with a sample review
    test_review = 'The simplest pleasures in life are the best, and this film is one of them.'
    print(f"Testing endpoint with sample review: '{test_review}'")
    result = predictor.predict(test_review)
    print(f"Prediction result: {result}")
    return predictor.endpoint_name

if __name__ == "__main__":
    main()