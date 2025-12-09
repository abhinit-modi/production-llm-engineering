FROM python:3.12
# Create a new user named 'user' with user ID 1000 and create their home directory
RUN useradd -m -u 1000 user
# Switch to the newly created user
USER user
# Add the user's local bin directory to the PATH
ENV PATH="/home/user/.local/bin:$PATH"
# Set the working directory in the container to /app
WORKDIR /app
# Copy the requirements.txt file from the host to the container
# The --chown=user ensures the copied file is owned by our 'user'
COPY --chown=user ./requirements.txt requirements.txt
# Install the Python dependencies listed in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
# Copy the rest of the application code from the host to the container
# Again, ensure the copied files are owned by 'user'
COPY --chown=user . /app
# Specify the command to run when the container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]