#!/bin/bash
# AWS deployment script for RL robotics training infrastructure

set -e

# Configuration
STACK_NAME="rl-robotics-training"
TEMPLATE_FILE="cloudformation/spot-fleet-training.yaml"
REGION="us-east-1"

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
    --vpc-id)
      VPC_ID="$2"
      shift 2
      ;;
    --subnet-ids)
      SUBNET_IDS="$2"
      shift 2
      ;;
    --key-name)
      KEY_NAME="$2"
      shift 2
      ;;
    --fleet-size)
      FLEET_SIZE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate required parameters
if [ -z "$VPC_ID" ] || [ -z "$SUBNET_IDS" ] || [ -z "$KEY_NAME" ]; then
  echo "Error: Missing required parameters"
  echo "Usage: $0 --vpc-id VPC_ID --subnet-ids SUBNET1,SUBNET2 --key-name KEY_NAME [--fleet-size SIZE]"
  exit 1
fi

echo "Deploying CloudFormation stack: $STACK_NAME"
echo "Region: $REGION"
echo "VPC: $VPC_ID"
echo "Subnets: $SUBNET_IDS"

# Create/Update CloudFormation stack
aws cloudformation deploy \
  --stack-name "$STACK_NAME" \
  --template-file "$TEMPLATE_FILE" \
  --region "$REGION" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    VpcId="$VPC_ID" \
    SubnetIds="$SUBNET_IDS" \
    KeyName="$KEY_NAME" \
    ${FLEET_SIZE:+FleetSize="$FLEET_SIZE"}

# Get stack outputs
echo ""
echo "Stack deployment complete!"
echo ""
echo "Outputs:"
aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
  --output table

# Get ECR repository URI
ECR_URI=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`ECRRepositoryUri`].OutputValue' \
  --output text)

echo ""
echo "Next steps:"
echo "1. Build and push Docker image:"
echo "   ./aws/push-image.sh --region $REGION"
echo ""
echo "2. Start training by creating spot fleet instances"
echo ""
