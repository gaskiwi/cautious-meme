"""AWS SageMaker integration for distributed training with warm pools."""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


class SageMakerTrainingRunner:
    """
    Manages training execution on AWS SageMaker with warm pool support.
    
    Features:
    - Managed warm pools for fast session transitions
    - Spot instance support for cost optimization
    - Automatic checkpoint management via S3
    - CloudWatch metrics integration
    - VPC and security configuration
    """
    
    def __init__(
        self,
        pipeline_config_path: str,
        role_arn: str,
        output_dir: str = "./sagemaker_output"
    ):
        """
        Initialize SageMaker runner.
        
        Args:
            pipeline_config_path: Path to pipeline configuration
            role_arn: IAM role ARN for SageMaker execution
            output_dir: Local directory for outputs
        """
        self.pipeline_config = load_config(pipeline_config_path)
        self.role_arn = role_arn
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = setup_logger(
            'sagemaker_runner',
            log_file=str(self.output_dir / 'sagemaker.log')
        )
        
        # Initialize AWS clients
        self._setup_aws_clients()
        
        # Get AWS config
        self.aws_config = self.pipeline_config.get('aws', {})
        self.sagemaker_config = self.aws_config.get('sagemaker', {})
        self.s3_bucket = self.aws_config.get('s3_bucket')
        self.region = self.aws_config.get('region', 'us-east-1')
        
        if not self.s3_bucket:
            raise ValueError("S3 bucket must be specified in pipeline config")
        
        self.logger.info("SageMaker runner initialized")
        self.logger.info(f"Region: {self.region}")
        self.logger.info(f"S3 Bucket: {self.s3_bucket}")
    
    def _setup_aws_clients(self):
        """Setup AWS SDK clients."""
        try:
            import boto3
            self.sagemaker_client = boto3.client('sagemaker')
            self.s3_client = boto3.client('s3')
            self.cloudwatch_client = boto3.client('cloudwatch')
            self.logger.info("AWS clients initialized")
        except ImportError:
            self.logger.error("boto3 not installed. Install with: pip install boto3")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize AWS clients: {e}")
            raise
    
    def _create_training_job_name(self, session_name: str) -> str:
        """Generate unique training job name."""
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        return f"rl-robotics-{session_name}-{timestamp}"
    
    def _prepare_hyperparameters(
        self,
        session: Dict[str, Any],
        checkpoint_s3_uri: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Prepare hyperparameters for SageMaker training job.
        
        Args:
            session: Session configuration
            checkpoint_s3_uri: S3 URI of checkpoint to resume from
            
        Returns:
            Hyperparameters dict (all values must be strings)
        """
        hyperparameters = {
            'config': session['config'],
            'session-name': session['name'],
        }
        
        if checkpoint_s3_uri:
            hyperparameters['checkpoint-s3-uri'] = checkpoint_s3_uri
        
        if session.get('timesteps'):
            hyperparameters['timesteps'] = str(session['timesteps'])
        
        # Add any additional overrides
        if 'overrides' in session:
            for key, value in session['overrides'].items():
                if isinstance(value, dict):
                    hyperparameters[f'override-{key}'] = json.dumps(value)
                else:
                    hyperparameters[f'override-{key}'] = str(value)
        
        return hyperparameters
    
    def _get_training_job_config(
        self,
        job_name: str,
        session: Dict[str, Any],
        checkpoint_s3_uri: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build SageMaker training job configuration.
        
        Args:
            job_name: Training job name
            session: Session configuration
            checkpoint_s3_uri: S3 URI of checkpoint
            
        Returns:
            Training job configuration dict
        """
        # Get session-specific AWS overrides
        session_aws = session.get('aws_overrides', {}).get('sagemaker', {})
        
        # Merge with global SageMaker config
        instance_type = session_aws.get('instance_type', 
                                        self.sagemaker_config.get('instance_type', 'ml.g4dn.xlarge'))
        instance_count = session_aws.get('instance_count',
                                        self.sagemaker_config.get('instance_count', 1))
        max_run = session_aws.get('max_run_seconds',
                                  self.sagemaker_config.get('max_run_seconds', 86400))
        
        # Base configuration
        config = {
            'TrainingJobName': job_name,
            'RoleArn': self.role_arn,
            'AlgorithmSpecification': {
                'TrainingImage': self.sagemaker_config.get('container_image_uri'),
                'TrainingInputMode': 'File',
                'EnableSageMakerMetricsTimeSeries': True
            },
            'ResourceConfig': {
                'InstanceType': instance_type,
                'InstanceCount': instance_count,
                'VolumeSizeInGB': 50
            },
            'StoppingCondition': {
                'MaxRuntimeInSeconds': max_run
            },
            'HyperParameters': self._prepare_hyperparameters(session, checkpoint_s3_uri),
            'OutputDataConfig': {
                'S3OutputPath': f's3://{self.s3_bucket}/output/'
            }
        }
        
        # Add checkpoint configuration
        if checkpoint_s3_uri:
            config['CheckpointConfig'] = {
                'S3Uri': f's3://{self.s3_bucket}/checkpoints/{session["name"]}/',
                'LocalPath': '/opt/ml/checkpoints'
            }
        
        # Add spot instance configuration
        if self.sagemaker_config.get('use_spot_instances', False):
            config['EnableManagedSpotTraining'] = True
            max_wait = self.sagemaker_config.get('max_wait_seconds', 3600)
            config['StoppingCondition']['MaxWaitTimeInSeconds'] = max_run + max_wait
        
        # Add VPC configuration if specified
        vpc_config = self.sagemaker_config.get('vpc_config', {})
        if vpc_config.get('subnets') and vpc_config.get('security_group_ids'):
            config['VpcConfig'] = {
                'SecurityGroupIds': vpc_config['security_group_ids'],
                'Subnets': vpc_config['subnets']
            }
        
        # Add warm pool configuration
        warm_pool = self.sagemaker_config.get('warm_pool', {})
        if warm_pool.get('enabled', False):
            config['ResourceConfig']['KeepAlivePeriodInSeconds'] = \
                warm_pool.get('retention_period_seconds', 3600)
        
        # Add metric definitions for CloudWatch
        monitoring = self.pipeline_config.get('monitoring', {})
        sagemaker_metrics = monitoring.get('sagemaker_metrics', [])
        if sagemaker_metrics:
            config['AlgorithmSpecification']['MetricDefinitions'] = sagemaker_metrics
        
        # Add tags
        config['Tags'] = [
            {'Key': 'Project', 'Value': 'RL-Robotics'},
            {'Key': 'Session', 'Value': session['name']},
            {'Key': 'Pipeline', 'Value': self.pipeline_config.get('name', 'default')}
        ]
        
        return config
    
    def run_training_job(
        self,
        session: Dict[str, Any],
        checkpoint_s3_uri: Optional[str] = None,
        wait: bool = True
    ) -> Dict[str, Any]:
        """
        Launch SageMaker training job for a session.
        
        Args:
            session: Session configuration
            checkpoint_s3_uri: S3 URI of checkpoint to resume from
            wait: Whether to wait for job completion
            
        Returns:
            Training job details
        """
        job_name = self._create_training_job_name(session['name'])
        
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Launching SageMaker Training Job: {job_name}")
        self.logger.info(f"Session: {session['name']}")
        self.logger.info(f"{'='*80}")
        
        # Build job configuration
        job_config = self._get_training_job_config(job_name, session, checkpoint_s3_uri)
        
        # Launch training job
        try:
            response = self.sagemaker_client.create_training_job(**job_config)
            
            self.logger.info(f"Training job launched: {response['TrainingJobArn']}")
            
            if wait:
                return self._wait_for_job_completion(job_name)
            else:
                return {
                    'job_name': job_name,
                    'job_arn': response['TrainingJobArn'],
                    'status': 'InProgress'
                }
                
        except Exception as e:
            self.logger.error(f"Failed to launch training job: {e}", exc_info=True)
            raise
    
    def _wait_for_job_completion(self, job_name: str) -> Dict[str, Any]:
        """
        Wait for training job to complete.
        
        Args:
            job_name: Training job name
            
        Returns:
            Final job status and metrics
        """
        self.logger.info(f"Waiting for training job completion: {job_name}")
        
        start_time = time.time()
        last_status = None
        
        while True:
            try:
                response = self.sagemaker_client.describe_training_job(
                    TrainingJobName=job_name
                )
                
                status = response['TrainingJobStatus']
                
                # Log status changes
                if status != last_status:
                    self.logger.info(f"Job status: {status}")
                    last_status = status
                
                # Check if completed
                if status in ['Completed', 'Failed', 'Stopped']:
                    duration = time.time() - start_time
                    
                    if status == 'Completed':
                        self.logger.info(f"Training job completed successfully!")
                        self.logger.info(f"Duration: {duration:.2f} seconds")
                        
                        # Get final metrics
                        metrics = response.get('FinalMetricDataList', [])
                        if metrics:
                            self.logger.info("Final Metrics:")
                            for metric in metrics:
                                self.logger.info(f"  {metric['MetricName']}: {metric['Value']}")
                        
                        return {
                            'job_name': job_name,
                            'status': status,
                            'duration_seconds': duration,
                            'metrics': metrics,
                            'model_artifacts': response.get('ModelArtifacts', {})
                        }
                    else:
                        failure_reason = response.get('FailureReason', 'Unknown')
                        self.logger.error(f"Training job {status.lower()}: {failure_reason}")
                        
                        return {
                            'job_name': job_name,
                            'status': status,
                            'duration_seconds': duration,
                            'failure_reason': failure_reason
                        }
                
                # Wait before next check
                time.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error checking job status: {e}")
                time.sleep(30)
    
    def run_pipeline(self):
        """Execute complete training pipeline on SageMaker."""
        self.logger.info("="*80)
        self.logger.info("SAGEMAKER TRAINING PIPELINE STARTED")
        self.logger.info("="*80)
        
        sessions = self.pipeline_config['sessions']
        results = []
        previous_checkpoint_uri = None
        
        for session in sessions:
            session_name = session['name']
            
            # Determine checkpoint
            checkpoint_uri = None
            if session.get('use_previous_checkpoint', False) and previous_checkpoint_uri:
                checkpoint_uri = previous_checkpoint_uri
                self.logger.info(f"Using checkpoint: {checkpoint_uri}")
            
            # Run training job
            try:
                result = self.run_training_job(
                    session,
                    checkpoint_s3_uri=checkpoint_uri,
                    wait=True
                )
                
                results.append(result)
                
                # Update checkpoint for next session
                if result['status'] == 'Completed':
                    model_artifacts = result.get('model_artifacts', {})
                    if model_artifacts:
                        previous_checkpoint_uri = model_artifacts.get('S3ModelArtifacts')
                else:
                    # Stop on failure unless configured to continue
                    if not self.pipeline_config.get('continue_on_error', False):
                        self.logger.error("Pipeline stopped due to job failure")
                        break
                        
            except Exception as e:
                self.logger.error(f"Session '{session_name}' failed: {e}", exc_info=True)
                if not self.pipeline_config.get('continue_on_error', False):
                    break
        
        # Generate report
        self._generate_report(results)
        
        self.logger.info("="*80)
        self.logger.info("SAGEMAKER TRAINING PIPELINE COMPLETED")
        self.logger.info("="*80)
    
    def _generate_report(self, results: List[Dict[str, Any]]):
        """Generate pipeline execution report."""
        report = {
            'pipeline': self.pipeline_config.get('name', 'Unnamed Pipeline'),
            'started_at': datetime.now().isoformat(),
            'platform': 'AWS SageMaker',
            'sessions': results,
            'summary': {
                'total_sessions': len(results),
                'completed': sum(1 for r in results if r['status'] == 'Completed'),
                'failed': sum(1 for r in results if r['status'] != 'Completed'),
                'total_duration': sum(r.get('duration_seconds', 0) for r in results)
            }
        }
        
        report_path = self.output_dir / 'sagemaker_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Report saved to: {report_path}")
        
        # Upload to S3
        try:
            s3_key = f"reports/sagemaker_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.s3_client.upload_file(
                str(report_path),
                self.s3_bucket,
                s3_key
            )
            self.logger.info(f"Report uploaded to S3: s3://{self.s3_bucket}/{s3_key}")
        except Exception as e:
            self.logger.error(f"Failed to upload report to S3: {e}")


def main():
    """CLI entry point for SageMaker pipeline execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Execute training pipeline on AWS SageMaker with warm pools"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to pipeline configuration file'
    )
    parser.add_argument(
        '--role-arn',
        type=str,
        required=True,
        help='IAM role ARN for SageMaker execution'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./sagemaker_output',
        help='Local directory for outputs'
    )
    
    args = parser.parse_args()
    
    # Create and run SageMaker runner
    runner = SageMakerTrainingRunner(
        pipeline_config_path=args.config,
        role_arn=args.role_arn,
        output_dir=args.output_dir
    )
    
    runner.run_pipeline()


if __name__ == '__main__':
    main()
