import os
import json
import argparse
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.predictor import Predictor
from sagemaker.serializers import StringSerializer


from dotenv import load_dotenv

load_dotenv()
IAM_ROLE_NAME = os.environ['IAM_ROLE_NAME']

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

def main():
    args = parse_args()

    # Get model_data path in s3
    with open('model_data.txt') as f:
        model_data = f.read()
    
    # Create a PyTorch model
    model = PyTorchModel(
        model_data=model_data,
        role= IAM_ROLE_NAME,
        framework_version="1.13.1",
        py_version="py39",
        entry_point='predict.py',
        source_dir='serve',
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
