import sagemaker
from sagemaker.pytorch import PyTorch
from torchvision.datasets import MNIST
import os

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = sagemaker_session.default_bucket()

print(f"Using Bucket: {bucket}")


data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

print("Downloading data locally...")
MNIST(root=data_dir, download=True)

print("Uploading data to S3...")
input_data = sagemaker_session.upload_data(path=data_dir, bucket=bucket, key_prefix='data/mnist')
print(f"Data uploaded to: {input_data}")


estimator = PyTorch(
    entry_point='train.py',
    source_dir='code',
    role=role,
    framework_version='2.0',
    py_version='py310',
    instance_count=1,
    instance_type='ml.g4dn.xlarge',
    hyperparameters={
        'epochs': 3,
        'batch-size': 64,
        'lr': 0.01
    }
)

print("Starting Training Job...")
estimator.fit({'train': input_data})

print(f"Training finished. Model artifacts are at: {estimator.model_data}")