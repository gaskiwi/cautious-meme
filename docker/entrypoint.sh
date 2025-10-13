#!/bin/bash
set -e

# AWS Spot Fleet training entrypoint script

# Download checkpoint from S3 if specified
if [ ! -z "$S3_CHECKPOINT_PATH" ]; then
    echo "Downloading checkpoint from S3: $S3_CHECKPOINT_PATH"
    aws s3 cp "$S3_CHECKPOINT_PATH" /app/checkpoint.zip
    unzip /app/checkpoint.zip -d /app/models/
fi

# Run training
python3 train.py --config configs/training_config.yaml ${TRAINING_ARGS}

# Upload results to S3
if [ ! -z "$S3_OUTPUT_PATH" ]; then
    echo "Uploading results to S3: $S3_OUTPUT_PATH"
    aws s3 sync /app/models/ "$S3_OUTPUT_PATH/models/"
    aws s3 sync /app/logs/ "$S3_OUTPUT_PATH/logs/"
fi

echo "Training completed!"
