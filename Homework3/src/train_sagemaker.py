from sagemaker.huggingface import HuggingFace

# TODO: Provide all the necessary information to the HuggingFace estimator to get it to run on Sagemaker:
# - Specify the source_dir. This is where the file that is running the training application is located.
# - Specify the entry_point. This is the file where is running the training application.
# - Specify the instance_type. Make sure you have requested enough quotas to use a specific machine (https://console.aws.amazon.com/servicequotas/home). Personally, I was using ml.p3.2xlarge, but there are cheaper machines. Make sure to check the cost of those (https://aws.amazon.com/sagemaker/pricing/)!!!
# - Specify the role. Check HW2, to generate a role.
# - Specify the image_uri. You can find the different Huggingface training containers 
# here: https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-training-containers. 
# Personally, I tend to use the following because it works: 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04
# - Specify the dependencies. You will most likely need two dependency files, 
# requirements.txt to install the additional packages not available in the docker image 
# and that your code may need, and the .env file where you can store the environment variables. 
# You can use the load_dotenv function to load the environment variables. 
# You will need to have your Huggingface and your Weight and Bias tokens.

huggingface_estimator = HuggingFace(
    source_dir=".",
    entry_point="src/training_application.py",
    instance_type="ml.p3.2xlarge",
    instance_count=1,
    role="arn:aws:iam::951719175793:role/service-role/AmazonSageMaker-ExecutionRole-20250620T120866",
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04",
    py_version="pyv39",
    dependencies=["requirements.txt", ".env"],
    hyperparameters={
        'training_type': "accelerated"  # one of ["basic", "accelerated"]
    }

)
huggingface_estimator.fit()
