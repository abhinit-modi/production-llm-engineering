from sagemaker.huggingface import HuggingFace
import os

role = os.environ.get('SAGEMAKER_EXECUTION_ROLE', 'arn:aws:iam::YOUR_ACCOUNT_ID:role/service-role/YOUR_SAGEMAKER_EXECUTION_ROLE')

huggingface_estimator = HuggingFace(
        entry_point='training_application.py',
        source_dir='.',
        instance_type='ml.p3.2xlarge',
        instance_count=1,
        role=role,
        dependencies=['./requirements.txt'],
        image_uri='763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04',
        py_version='py310',
         environment={
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
        }
)

huggingface_estimator.fit()