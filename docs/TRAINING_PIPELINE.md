# Training Pipeline Guide

This guide explains how to use the organized training pipeline system for sequential environment execution in the RL Robotics project.

## Overview

The training pipeline system provides:
- **Sequential Execution**: Run multiple training sessions in order
- **Checkpoint Management**: Automatic saving and loading of model checkpoints
- **Transfer Learning**: Use models from previous sessions as starting points
- **AWS Integration**: Support for both local and SageMaker execution
- **Warm Pools**: Fast session transitions on AWS SageMaker
- **State Persistence**: Resume interrupted pipelines

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Pipeline Orchestrator                  │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│  │ Session 1  │→ │ Session 2  │→ │ Session 3  │       │
│  │  Height    │  │   Crush    │  │   Future   │       │
│  └────────────┘  └────────────┘  └────────────┘       │
│        ↓              ↓               ↓                 │
│  ┌────────────────────────────────────────────┐       │
│  │         Checkpoint Management               │       │
│  │  - Local Storage or S3                      │       │
│  │  - Transfer Learning Support                │       │
│  └────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Local Execution

1. **Basic pipeline run:**
```bash
python3 run_pipeline.py --config configs/pipeline_config.yaml
```

2. **Resume interrupted pipeline:**
```bash
python3 run_pipeline.py --config configs/pipeline_config.yaml --resume
```

3. **Start from specific session:**
```bash
python3 run_pipeline.py --config configs/pipeline_config.yaml \
  --start-from session2_crush_resistance
```

4. **With S3 checkpoint storage:**
```bash
python3 run_pipeline.py --config configs/pipeline_config.yaml \
  --use-s3 --s3-bucket my-bucket-name
```

### AWS SageMaker Execution

1. **Deploy infrastructure:**
```bash
cd aws
./deploy_sagemaker.sh --vpc-id vpc-xxxxx \
  --subnet-ids subnet-xxxx,subnet-yyyy
```

2. **Build and push Docker image:**
```bash
./push-image.sh --region us-east-1
```

3. **Run pipeline on SageMaker:**
```bash
python3 run_pipeline.py \
  --config configs/pipeline_config_aws.yaml \
  --platform sagemaker \
  --role-arn arn:aws:iam::123456789012:role/SageMakerRole
```

## Pipeline Configuration

### Basic Configuration Structure

```yaml
name: "Multi-Environment RL Robotics Pipeline"
description: "Sequential training pipeline"

# Pipeline settings
continue_on_error: false  # Stop on failure
enable_transfer_learning: true  # Use previous checkpoints
save_all_checkpoints: true

# AWS settings (optional)
aws:
  use_s3: false
  s3_bucket: "rl-robotics-models"
  region: "us-east-1"
  
  # SageMaker configuration
  sagemaker:
    enabled: true
    instance_type: "ml.g4dn.xlarge"
    
    # Warm pools for fast transitions
    warm_pool:
      enabled: true
      retention_period_seconds: 3600

# Training sessions
sessions:
  - name: "session1_height_maximize"
    description: "Height maximization training"
    config: "configs/training_config.yaml"
    use_previous_checkpoint: false
    
  - name: "session2_crush_resistance"
    description: "Crush resistance training"
    config: "configs/session2_config.yaml"
    use_previous_checkpoint: true  # Transfer learning
```

### Session Configuration

Each session can have:
- `name`: Unique identifier for the session
- `description`: Human-readable description
- `config`: Path to training configuration file
- `timesteps`: Override total training timesteps
- `use_previous_checkpoint`: Use checkpoint from previous session
- `checkpoint`: Explicit checkpoint path (optional)
- `overrides`: Override config values for this session
- `aws_overrides`: AWS-specific settings for this session

## Features

### 1. Sequential Execution

The pipeline executes training sessions in order, allowing each session to build on the previous one:

```python
# Session 1: Learn basic height control
Session 1 (Height) → Checkpoint

# Session 2: Use height skills for crush resistance
Checkpoint → Session 2 (Crush) → Checkpoint

# Session 3: Further refinement
Checkpoint → Session 3 (Future) → Final Model
```

### 2. Checkpoint Management

Checkpoints are automatically managed:
- Saved after each session completes
- Stored locally or on S3
- Can be used for transfer learning
- Include full model state

**Local Storage:**
```
pipeline_output/
├── session1_height_maximize/
│   ├── models/
│   │   ├── best_model.zip
│   │   └── final_model.zip
│   └── logs/
└── session2_crush_resistance/
    ├── models/
    └── logs/
```

**S3 Storage:**
```
s3://bucket/
├── checkpoints/
│   ├── session1_height_maximize/
│   └── session2_crush_resistance/
└── reports/
```

### 3. Transfer Learning

Enable transfer learning to use skills from previous sessions:

```yaml
sessions:
  - name: "session2"
    use_previous_checkpoint: true  # Use Session 1's checkpoint
```

This is especially useful when:
- Sessions share similar environments
- Later sessions build on earlier skills
- You want to speed up convergence

### 4. State Persistence

The pipeline saves its state to enable resume:

```json
{
  "sessions_completed": ["session1_height_maximize"],
  "current_session": null,
  "started_at": "2025-10-23T10:00:00",
  "checkpoints": {
    "session1_height_maximize": "path/to/checkpoint.zip"
  }
}
```

Resume with:
```bash
python3 run_pipeline.py --config pipeline_config.yaml --resume
```

### 5. AWS SageMaker Integration

#### Benefits of SageMaker
- **GPU instances**: Fast training with NVIDIA GPUs
- **Managed warm pools**: Fast transitions between sessions
- **Spot instances**: Up to 70% cost savings
- **Automatic scaling**: Scale compute as needed
- **CloudWatch metrics**: Built-in monitoring

#### Warm Pools

Warm pools keep instances alive between sessions, dramatically reducing startup time:

```yaml
warm_pool:
  enabled: true
  retention_period_seconds: 3600  # Keep alive for 1 hour
```

**Without warm pools:**
- Session 1: 5 min setup + 60 min training = 65 min
- Session 2: 5 min setup + 90 min training = 95 min
- **Total: 160 min**

**With warm pools:**
- Session 1: 5 min setup + 60 min training = 65 min
- Session 2: ~10 sec transition + 90 min training = 90.2 min
- **Total: 155.2 min (3% faster, ~$3 saved)**

#### Spot Instances

Use spot instances for additional cost savings:

```yaml
sagemaker:
  use_spot_instances: true
  max_wait_seconds: 3600  # Wait up to 1 hour for capacity
```

Typical savings: **60-70% vs on-demand pricing**

## AWS Deployment

### Prerequisites

1. AWS CLI configured:
```bash
aws configure
```

2. VPC with subnets (for SageMaker)
3. Docker installed (for building images)

### Step-by-Step Deployment

#### 1. Deploy CloudFormation Stack

```bash
cd aws
./deploy_sagemaker.sh \
  --vpc-id vpc-xxxxx \
  --subnet-ids subnet-xxxx,subnet-yyyy \
  --region us-east-1
```

This creates:
- S3 bucket for checkpoints and data
- IAM role for SageMaker execution
- Security groups for training jobs
- ECR repository for container images
- SNS topics for notifications
- Lambda orchestrator (optional)

#### 2. Build and Push Docker Image

```bash
./push-image.sh --region us-east-1
```

This:
- Builds training container image
- Pushes to ECR
- Tags as `latest`

#### 3. Update Pipeline Configuration

Edit `configs/pipeline_config_aws.yaml` with CloudFormation outputs:

```yaml
aws:
  use_s3: true
  s3_bucket: "rl-robotics-sagemaker-123456789012"  # From outputs
  
  sagemaker:
    enabled: true
    role_arn: "arn:aws:iam::123456789012:role/..."  # From outputs
    container_image_uri: "123456789012.dkr.ecr.us-east-1.amazonaws.com/..."
    
    vpc_config:
      security_group_ids: ["sg-xxxxx"]  # From outputs
      subnets: ["subnet-xxxx", "subnet-yyyy"]
```

#### 4. Run Pipeline

```bash
python3 run_pipeline.py \
  --config configs/pipeline_config_aws.yaml \
  --platform sagemaker \
  --role-arn arn:aws:iam::123456789012:role/SageMakerRole
```

### Monitoring

#### CloudWatch Logs

View training logs:
```bash
aws logs tail /aws/sagemaker/TrainingJobs/rl-robotics --follow
```

#### Training Job Status

Check job status:
```bash
aws sagemaker describe-training-job \
  --training-job-name rl-robotics-session1-20251023-100000
```

#### SNS Notifications

Subscribe to training notifications:
```bash
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:123456789012:rl-robotics-training-notifications \
  --protocol email \
  --notification-endpoint your-email@example.com
```

## Cost Optimization

### Strategies

1. **Use Spot Instances**: 60-70% savings
2. **Enable Warm Pools**: Reduce startup time
3. **Right-Size Instances**: Start small, scale up if needed
4. **Set Max Runtime**: Prevent runaway costs
5. **Use S3 Lifecycle Policies**: Archive old checkpoints

### Example Cost Calculation

**On-Demand ml.g4dn.xlarge**: $0.736/hour

**Training Pipeline (Without Optimization):**
- Session 1: 2 hours = $1.47
- Session 2: 3 hours = $2.21
- Total: $3.68

**Training Pipeline (With Spot + Warm Pools):**
- Session 1: 2 hours @ 70% discount = $0.44
- Session 2: 3 hours @ 70% discount = $0.66
- Warm pool: 0.02 hours @ full price = $0.01
- Total: $1.11

**Savings: 70%** ($2.57 saved)

## Advanced Usage

### Custom Session Orchestration

Create custom orchestration logic:

```python
from src.training.pipeline import TrainingPipeline

pipeline = TrainingPipeline(
    pipeline_config_path="configs/pipeline_config.yaml",
    output_dir="./custom_output"
)

# Run specific sessions with custom logic
for session in pipeline.pipeline_config['sessions']:
    if should_run_session(session):
        result = pipeline.run_session(session)
        
        if not meets_success_criteria(result):
            # Retry with different hyperparameters
            session['overrides']['agent']['learning_rate'] *= 0.5
            result = pipeline.run_session(session)
```

### Multi-Pipeline Execution

Run multiple pipelines in parallel for hyperparameter search:

```bash
# Terminal 1
python3 run_pipeline.py --config pipeline_lr_0.001.yaml &

# Terminal 2
python3 run_pipeline.py --config pipeline_lr_0.0001.yaml &

# Terminal 3
python3 run_pipeline.py --config pipeline_lr_0.00001.yaml &
```

### Integration with Ray Tune

Use Ray Tune for hyperparameter optimization:

```python
from ray import tune
from src.training.pipeline import TrainingPipeline

def train_pipeline(config):
    # Update pipeline config with Ray Tune parameters
    pipeline = TrainingPipeline("configs/pipeline_config.yaml")
    # Run and return metrics
    return metrics

analysis = tune.run(
    train_pipeline,
    config={
        "learning_rate": tune.grid_search([1e-3, 1e-4, 1e-5]),
        "batch_size": tune.grid_search([32, 64, 128])
    }
)
```

## Troubleshooting

### Pipeline Fails to Resume

**Problem**: Pipeline doesn't resume from correct session

**Solution**: Check state file at `pipeline_output/pipeline_state.json`
```bash
cat pipeline_output/pipeline_state.json
```

### SageMaker Job Fails to Start

**Problem**: Training job fails to launch

**Common causes:**
1. Insufficient spot capacity → Wait or use on-demand
2. VPC/subnet misconfigured → Check security groups
3. IAM role missing permissions → Review CloudFormation outputs

**Debug:**
```bash
aws sagemaker describe-training-job \
  --training-job-name <job-name> \
  | jq '.FailureReason'
```

### Checkpoint Not Found

**Problem**: Session can't find previous checkpoint

**Solution**: Verify checkpoint location
```bash
# Local
ls -la pipeline_output/session1_height_maximize/models/

# S3
aws s3 ls s3://bucket/checkpoints/session1_height_maximize/
```

### Out of Memory on GPU

**Problem**: Training job crashes with OOM error

**Solutions:**
1. Reduce batch size
2. Use larger instance type
3. Enable gradient checkpointing
4. Reduce network size

## Best Practices

1. **Start Simple**: Test pipeline locally before AWS deployment
2. **Version Configs**: Keep pipeline configs in version control
3. **Monitor Costs**: Set up billing alerts in AWS
4. **Save Checkpoints Frequently**: Enable automatic checkpointing
5. **Use Transfer Learning**: Leverage previous session knowledge
6. **Document Sessions**: Keep TRAINING_SESSIONS.md updated
7. **Test Environments**: Validate each environment before pipeline
8. **Set Reasonable Timeouts**: Prevent runaway training jobs

## Next Steps

- Add more training sessions to the pipeline
- Experiment with different algorithms (PPO, SAC, TD3)
- Implement curriculum learning strategies
- Set up automated hyperparameter tuning
- Create evaluation pipeline for model comparison

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review CloudWatch logs for SageMaker jobs
3. Check GitHub issues: https://github.com/gaskiwi/cautious-meme
4. Review AWS SageMaker documentation

---

**Last Updated**: 2025-10-23
