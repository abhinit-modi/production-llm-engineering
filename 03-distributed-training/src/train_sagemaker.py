from sagemaker.huggingface import HuggingFace
import os

# SageMaker HuggingFace Estimator configuration for cloud training
# Uses p3.2xlarge GPU instance with HuggingFace PyTorch container

huggingface_estimator = HuggingFace(
    source_dir=".",
    entry_point="src/training_application.py",
    instance_type="ml.p3.2xlarge",
    instance_count=1,
    role=os.environ.get('SAGEMAKER_EXECUTION_ROLE', 'arn:aws:iam::YOUR_ACCOUNT_ID:role/service-role/YOUR_SAGEMAKER_EXECUTION_ROLE'),
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04",
    py_version="pyv39",
    dependencies=["requirements.txt", ".env"],
    hyperparameters={
        'training_type': "accelerated"  # one of ["basic", "accelerated"]
    }

)
huggingface_estimator.fit()
