#!/bin/bash
# Script to build and push Docker image to ECR

set -e

# Configuration
STACK_NAME="rl-robotics-training"
REGION="us-east-1"
IMAGE_TAG="latest"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --stack-name)
      STACK_NAME="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Get ECR repository URI from CloudFormation stack
echo "Getting ECR repository URI from stack: $STACK_NAME"
ECR_URI=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`ECRRepositoryUri`].OutputValue' \
  --output text)

if [ -z "$ECR_URI" ]; then
  echo "Error: Could not get ECR repository URI from stack"
  exit 1
fi

echo "ECR Repository: $ECR_URI"

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region "$REGION" | \
  docker login --username AWS --password-stdin "$ECR_URI"

# Build Docker image
echo "Building Docker image..."
docker build -t rl-robotics:$IMAGE_TAG .

# Tag image for ECR
echo "Tagging image..."
docker tag rl-robotics:$IMAGE_TAG "$ECR_URI:$IMAGE_TAG"

# Push to ECR
echo "Pushing image to ECR..."
docker push "$ECR_URI:$IMAGE_TAG"

echo ""
echo "Successfully pushed image: $ECR_URI:$IMAGE_TAG"
echo ""
