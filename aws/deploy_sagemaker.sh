#!/bin/bash
# Deploy AWS SageMaker infrastructure for RL robotics training
# This script deploys the SageMaker-optimized CloudFormation stack

set -e

# Configuration
STACK_NAME="rl-robotics-sagemaker"
TEMPLATE_FILE="cloudformation/sagemaker-pipeline.yaml"
REGION="us-east-1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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
    --project-name)
      PROJECT_NAME="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 --vpc-id VPC_ID --subnet-ids SUBNET1,SUBNET2 [OPTIONS]"
      echo ""
      echo "Required arguments:"
      echo "  --vpc-id VPC_ID          VPC ID for SageMaker training jobs"
      echo "  --subnet-ids SUBNETS     Comma-separated list of subnet IDs"
      echo ""
      echo "Optional arguments:"
      echo "  --stack-name NAME        CloudFormation stack name (default: rl-robotics-sagemaker)"
      echo "  --region REGION          AWS region (default: us-east-1)"
      echo "  --project-name NAME      Project name for resources (default: rl-robotics)"
      echo "  --help                   Show this help message"
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      exit 1
      ;;
  esac
done

# Validate required parameters
if [ -z "$VPC_ID" ] || [ -z "$SUBNET_IDS" ]; then
  echo -e "${RED}Error: Missing required parameters${NC}"
  echo "Usage: $0 --vpc-id VPC_ID --subnet-ids SUBNET1,SUBNET2 [--project-name NAME]"
  echo "Run with --help for more information"
  exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deploying SageMaker Infrastructure${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Stack Name: $STACK_NAME"
echo "Region: $REGION"
echo "VPC: $VPC_ID"
echo "Subnets: $SUBNET_IDS"
echo ""

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
  echo -e "${RED}Error: AWS CLI not configured${NC}"
  echo "Please configure AWS CLI with: aws configure"
  exit 1
fi

# Navigate to aws directory
cd "$(dirname "$0")"

# Validate CloudFormation template
echo -e "${YELLOW}Validating CloudFormation template...${NC}"
aws cloudformation validate-template \
  --template-body file://"$TEMPLATE_FILE" \
  --region "$REGION" \
  > /dev/null

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✓ Template validation successful${NC}"
else
  echo -e "${RED}✗ Template validation failed${NC}"
  exit 1
fi

# Deploy CloudFormation stack
echo ""
echo -e "${YELLOW}Deploying CloudFormation stack...${NC}"
aws cloudformation deploy \
  --stack-name "$STACK_NAME" \
  --template-file "$TEMPLATE_FILE" \
  --region "$REGION" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    VpcId="$VPC_ID" \
    SubnetIds="$SUBNET_IDS" \
    ${PROJECT_NAME:+ProjectName="$PROJECT_NAME"}

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✓ Stack deployment successful${NC}"
else
  echo -e "${RED}✗ Stack deployment failed${NC}"
  exit 1
fi

# Get stack outputs
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Stack Outputs${NC}"
echo -e "${GREEN}========================================${NC}"
aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
  --output table

# Extract important outputs
S3_BUCKET=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`SageMakerBucketName`].OutputValue' \
  --output text)

SAGEMAKER_ROLE=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`SageMakerExecutionRoleArn`].OutputValue' \
  --output text)

ECR_URI=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`TrainingRepositoryUri`].OutputValue' \
  --output text)

SECURITY_GROUP=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`SageMakerSecurityGroupId`].OutputValue' \
  --output text)

# Save outputs to file
OUTPUT_FILE="sagemaker_outputs.sh"
cat > "$OUTPUT_FILE" << EOF
# SageMaker Infrastructure Outputs
# Generated on: $(date)
# Stack: $STACK_NAME
# Region: $REGION

export SAGEMAKER_S3_BUCKET="$S3_BUCKET"
export SAGEMAKER_ROLE_ARN="$SAGEMAKER_ROLE"
export SAGEMAKER_ECR_URI="$ECR_URI"
export SAGEMAKER_SECURITY_GROUP="$SECURITY_GROUP"
export SAGEMAKER_SUBNET_IDS="$SUBNET_IDS"
export AWS_REGION="$REGION"
EOF

echo ""
echo -e "${GREEN}✓ Output variables saved to: $OUTPUT_FILE${NC}"
echo -e "${YELLOW}  Source this file with: source $OUTPUT_FILE${NC}"

# Display next steps
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Next Steps${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "1. Build and push Docker image to ECR:"
echo "   ./push-image.sh --region $REGION --repository $ECR_URI"
echo ""
echo "2. Update pipeline configuration with AWS resources:"
echo "   Edit configs/pipeline_config_aws.yaml:"
echo "   - s3_bucket: $S3_BUCKET"
echo "   - sagemaker.role_arn: $SAGEMAKER_ROLE"
echo "   - sagemaker.container_image_uri: $ECR_URI:latest"
echo "   - sagemaker.vpc_config.security_group_ids: [$SECURITY_GROUP]"
echo "   - sagemaker.vpc_config.subnets: [${SUBNET_IDS//,/, }]"
echo ""
echo "3. Run training pipeline on SageMaker:"
echo "   python3 -m src.training.sagemaker_runner \\"
echo "     --config configs/pipeline_config_aws.yaml \\"
echo "     --role-arn $SAGEMAKER_ROLE"
echo ""
echo -e "${GREEN}========================================${NC}"
