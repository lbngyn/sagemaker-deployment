import os


import pandas as pd
import sagemaker
import argparse
import json
import boto3
from sagemaker.pytorch import PyTorch
from prepare_data import IMDbDataPreparation
from dotenv import load_dotenv

load_dotenv()
REGION = os.environ['AWS_REGION']
session = sagemaker.Session(boto_session=boto3.session.Session(region_name=REGION))
BUCKET_NAME = session.default_bucket()
PREFIX = os.environ['PREFIX']
# Replace with your IAM role arn that has enough access (e.g. SageMakerFullAccess)
IAM_ROLE_NAME = os.environ['IAM_ROLE_NAME']
ACCOUNT_ID = session.boto_session.client(
    'sts').get_caller_identity()['Account']


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SageMaker training job")
    parser.add_argument("--training-instance-type", type=str, default="ml.t3.large", 
                        help="SageMaker training instance type")
    parser.add_argument("--instance-count", type=int, default=1, 
                        help="Number of instances for training")
    parser.add_argument("--epochs", type=int, default=1, 
                        help="Number of training epochs")
    parser.add_argument("--hidden-dim", type=int, default=100, 
                        help="LSTM hidden dimension")
    parser.add_argument("--embedding-dim", type=int, default=32, 
                        help="Word embedding dimension")
    parser.add_argument("--vocab-size", type=int, default=5000, 
                        help="Vocabulary size")
    parser.add_argument("--output-path", type=str, default="./training_output", 
                        help="Path to save training results")
    return parser.parse_args()

def upload_data_to_s3(data_dir, bucket, prefix):
    """Upload data to S3 bucket."""
    print(f"Uploading data from {data_dir} to s3://{bucket}/{prefix}")
    sagemaker_session = sagemaker.Session()
    input_data = sagemaker_session.upload_data(
        path=data_dir,
        bucket=bucket,
        key_prefix=prefix
    )
    print(f"Data uploaded to: {input_data}")
    return input_data

def save_job_info(job_info, output_path):
    """Save job information to a JSON file."""
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'job_info.json'), 'w') as f:
        json.dump(job_info, f, indent=2)
    print(f"Job info saved to {os.path.join(output_path, 'job_info.json')}")

def main():
    args = parse_args()
    
    # Set data paths
    base_dir = './data'

    # data_prep = IMDbDataPreparation(base_dir=base_dir)
    # data, data_dir = data_prep.prepare_data()
    # print(data['train_X_len'])
    # print(data['test_X_len'])
    # # Upload data to S3
    # data_dir = './data/processed'
    # input_data = upload_data_to_s3(data_dir, BUCKET_NAME, PREFIX)
    input_data = f's3://{BUCKET_NAME}/{PREFIX}'
    
    # Create and run PyTorch estimator
    estimator = PyTorch(
        entry_point="train.py",
        source_dir="train",
        role=IAM_ROLE_NAME,
        framework_version="1.13.1",
        py_version="py39",
        instance_count=args.instance_count,
        instance_type=args.training_instance_type,
        output_path= 's3://{}/{}/model/'.format(BUCKET_NAME, PREFIX), 
        requirements_file="requirements.txt",  # 👈 ADD THIS LINE
        hyperparameters={
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "embedding_dim": args.embedding_dim,
            "vocab_size": args.vocab_size,
            "bucket_name": BUCKET_NAME, 
            "prefix": PREFIX
        }
    )
    
    print("Starting model training...")
    estimator.fit({'training': input_data})
    with open("model_data.txt", "w") as f:
        f.write(f"{estimator.model_data}")

    training_job_name = estimator.latest_training_job.name
    hyperparameters_dictionary = estimator.hyperparameters()
    print("Training_job_name:", training_job_name)
    print("hyperparameters_dictionary:", hyperparameters_dictionary)
    report = pd.read_csv(f's3://{BUCKET_NAME}/{PREFIX}/reports.csv')
    message = (f"## Training Job Submission Report\n\n"
            f"Training Job name: '{training_job_name}'\n\n"
                "Model Artifacts Location:\n\n"
            f"'s3://{BUCKET_NAME}/{PREFIX}/output/{training_job_name}/output/model.tar.gz'\n\n"
            f"Model hyperparameters: {hyperparameters_dictionary}\n\n"
                "See the Logs in a few minute at: "
            f"[CloudWatch](https://{REGION}.console.aws.amazon.com/cloudwatch/home?region={REGION}#logStream:group=/aws/sagemaker/TrainingJobs;prefix={training_job_name})\n\n"
                "If you merge this pull request the resulting endpoint will be avaible this URL:\n\n"
            f"'https://runtime.sagemaker.{REGION}.amazonaws.com/endpoints/{training_job_name}/invocations'\n\n"
            f"## Training Job Performance Report\n\n"
            f"{report.to_markdown(index=False)}\n\n"
            )
    print(message)

if __name__ == "__main__":
    main()
