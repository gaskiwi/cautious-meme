#!/usr/bin/env python3
"""
Unified entry point for running training pipelines.

This script provides a simple interface to run training pipelines
either locally or on AWS SageMaker with warm pools.
"""

import argparse
import sys
from pathlib import Path


def run_local_pipeline(args):
    """Run training pipeline locally."""
    from src.training.pipeline import TrainingPipeline
    
    print("="*80)
    print("Running Training Pipeline Locally")
    print("="*80)
    
    pipeline = TrainingPipeline(
        pipeline_config_path=args.config,
        output_dir=args.output_dir,
        use_s3=args.use_s3,
        s3_bucket=args.s3_bucket
    )
    
    pipeline.run(
        start_from=args.start_from,
        resume=args.resume
    )


def run_sagemaker_pipeline(args):
    """Run training pipeline on AWS SageMaker."""
    from src.training.sagemaker_runner import SageMakerTrainingRunner
    
    if not args.role_arn:
        print("Error: --role-arn required for SageMaker execution")
        print("Get the role ARN from CloudFormation outputs:")
        print("  aws cloudformation describe-stacks --stack-name rl-robotics-sagemaker \\")
        print("    --query 'Stacks[0].Outputs[?OutputKey==`SageMakerExecutionRoleArn`].OutputValue'")
        sys.exit(1)
    
    print("="*80)
    print("Running Training Pipeline on AWS SageMaker")
    print("="*80)
    
    runner = SageMakerTrainingRunner(
        pipeline_config_path=args.config,
        role_arn=args.role_arn,
        output_dir=args.output_dir
    )
    
    runner.run_pipeline()


def main():
    parser = argparse.ArgumentParser(
        description="Run RL robotics training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run locally
  python3 run_pipeline.py --config configs/pipeline_config.yaml
  
  # Run on AWS SageMaker
  python3 run_pipeline.py --config configs/pipeline_config_aws.yaml \\
    --platform sagemaker --role-arn arn:aws:iam::123456789012:role/SageMakerRole
  
  # Resume interrupted pipeline
  python3 run_pipeline.py --config configs/pipeline_config.yaml --resume
  
  # Start from specific session
  python3 run_pipeline.py --config configs/pipeline_config.yaml \\
    --start-from session2_crush_resistance
  
  # Use S3 for checkpoint storage (local execution)
  python3 run_pipeline.py --config configs/pipeline_config.yaml \\
    --use-s3 --s3-bucket my-bucket-name
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to pipeline configuration file'
    )
    
    parser.add_argument(
        '--platform',
        type=str,
        default='local',
        choices=['local', 'sagemaker'],
        help='Execution platform (default: local)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./pipeline_output',
        help='Directory for pipeline outputs (default: ./pipeline_output)'
    )
    
    # Local execution options
    local_group = parser.add_argument_group('local execution options')
    local_group.add_argument(
        '--start-from',
        type=str,
        default=None,
        help='Session name to start from (skips earlier sessions)'
    )
    
    local_group.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last interrupted session'
    )
    
    local_group.add_argument(
        '--use-s3',
        action='store_true',
        help='Use S3 for checkpoint storage'
    )
    
    local_group.add_argument(
        '--s3-bucket',
        type=str,
        default=None,
        help='S3 bucket name for checkpoints'
    )
    
    # SageMaker execution options
    sagemaker_group = parser.add_argument_group('sagemaker execution options')
    sagemaker_group.add_argument(
        '--role-arn',
        type=str,
        default=None,
        help='IAM role ARN for SageMaker execution (required for SageMaker)'
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Run on appropriate platform
    if args.platform == 'local':
        run_local_pipeline(args)
    elif args.platform == 'sagemaker':
        run_sagemaker_pipeline(args)


if __name__ == '__main__':
    main()
