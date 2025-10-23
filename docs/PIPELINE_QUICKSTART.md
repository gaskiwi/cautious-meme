# Training Pipeline Quick Start

Get started with the organized training pipeline in 5 minutes.

## What is the Training Pipeline?

The training pipeline allows you to run multiple training sessions sequentially, where each session:
- Trains on a different environment or objective
- Can use checkpoints from previous sessions (transfer learning)
- Automatically manages state and checkpoints
- Works locally or on AWS SageMaker

## Prerequisites

- Python 3.8+
- PyBullet, Stable-Baselines3, Gymnasium
- (Optional) AWS account for cloud training

## Installation

```bash
# Clone repository
git clone https://github.com/gaskiwi/cautious-meme.git
cd cautious-meme

# Install dependencies
make install
# or
pip install -r requirements.txt
```

## Your First Pipeline Run

### 1. Test Individual Environments

Before running the pipeline, test that environments work:

```bash
# Test height maximization environment
python3 examples/test_height_env.py

# Test crush resistance environment
python3 examples/test_crush_env.py
```

### 2. Run Local Pipeline

Run the complete pipeline locally:

```bash
make pipeline
# or
python3 run_pipeline.py --config configs/pipeline_config.yaml
```

This will:
1. Train Session 1 (Height Maximization) for 1M timesteps
2. Save checkpoint
3. Train Session 2 (Crush Resistance) for 2M timesteps using Session 1's checkpoint
4. Generate final report

**Expected time**: 2-4 hours on a modern laptop (depending on CPU)

### 3. Monitor Progress

Watch logs in real-time:

```bash
tail -f pipeline_output/pipeline.log
```

Or view TensorBoard:

```bash
tensorboard --logdir pipeline_output/
```

### 4. Check Results

After completion, review results:

```bash
# View pipeline report
cat pipeline_output/pipeline_report.json

# Evaluate final model
python3 evaluate.py pipeline_output/session2_crush_resistance/models/best_model.zip --render
```

## Pipeline Output Structure

```
pipeline_output/
├── pipeline.log                    # Main pipeline log
├── pipeline_state.json             # Pipeline state (for resume)
├── pipeline_report.json            # Final report
├── session1_height_maximize/
│   ├── models/
│   │   ├── best_model.zip         # Best checkpoint
│   │   └── final_model.zip        # Final checkpoint
│   ├── logs/
│   │   └── training.log
│   └── runs/                       # TensorBoard logs
└── session2_crush_resistance/
    ├── models/
    ├── logs/
    └── runs/
```

## Common Operations

### Resume Interrupted Pipeline

If training is interrupted, resume from where it left off:

```bash
make pipeline-resume
# or
python3 run_pipeline.py --config configs/pipeline_config.yaml --resume
```

### Start from Specific Session

Skip earlier sessions and start from a specific one:

```bash
python3 run_pipeline.py --config configs/pipeline_config.yaml \
  --start-from session2_crush_resistance
```

### Use S3 for Checkpoints

Store checkpoints on S3 instead of locally:

```bash
make pipeline-s3 S3_BUCKET=my-bucket-name
# or
python3 run_pipeline.py --config configs/pipeline_config.yaml \
  --use-s3 --s3-bucket my-bucket-name
```

## Customizing the Pipeline

### Edit Pipeline Configuration

Open `configs/pipeline_config.yaml`:

```yaml
sessions:
  - name: "session1_height_maximize"
    config: "configs/training_config.yaml"
    
    # Customize session settings
    overrides:
      training:
        total_timesteps: 500000  # Reduce for faster testing
      environment:
        num_type_b_robots: 1     # Use fewer robots
```

### Modify Training Parameters

Edit individual session configs:

```bash
# Session 1 config
vim configs/training_config.yaml

# Session 2 config
vim configs/session2_config.yaml
```

Key parameters to adjust:
- `total_timesteps`: How long to train
- `learning_rate`: Learning rate
- `batch_size`: Batch size
- `n_steps`: Steps per update

## Running on AWS SageMaker

### 1. Deploy Infrastructure

```bash
make aws-deploy-sm VPC_ID=vpc-xxxxx SUBNET_IDS=subnet-xxxx,subnet-yyyy
```

This creates:
- S3 bucket for storage
- SageMaker execution role
- ECR repository
- Security groups
- CloudWatch logging

### 2. Build and Push Container

```bash
make aws-push
```

### 3. Update AWS Configuration

Edit `configs/pipeline_config_aws.yaml` with outputs from step 1:

```yaml
aws:
  s3_bucket: "rl-robotics-sagemaker-123456789012"  # From CloudFormation
  
  sagemaker:
    role_arn: "arn:aws:iam::123456789012:role/..."  # From CloudFormation
    container_image_uri: "123456789012.dkr.ecr.us-east-1.amazonaws.com/..."
```

### 4. Run Pipeline on SageMaker

```bash
make aws-pipeline ROLE_ARN=arn:aws:iam::123456789012:role/SageMakerRole
```

**Benefits of SageMaker:**
- GPU acceleration (10-20x faster)
- Managed infrastructure
- Warm pools for fast session transitions
- Spot instances for 70% cost savings
- Automatic checkpointing to S3

## Troubleshooting

### Pipeline Won't Start

**Check Python version:**
```bash
python3 --version  # Should be 3.8+
```

**Verify dependencies:**
```bash
python3 -c "import gymnasium; import stable_baselines3; import pybullet"
```

### Out of Memory

**Reduce batch size** in config:
```yaml
agent:
  batch_size: 32  # Reduce from 64
```

**Or reduce number of robots:**
```yaml
environment:
  num_type_b_robots: 1  # Reduce from 2
```

### Pipeline Stuck

**Check if training is progressing:**
```bash
tail -f pipeline_output/session1_height_maximize/logs/training.log
```

**Monitor system resources:**
```bash
htop  # or top
```

### Can't Find Checkpoint

**Verify checkpoint exists:**
```bash
ls -la pipeline_output/session1_height_maximize/models/
```

**Check pipeline state:**
```bash
cat pipeline_output/pipeline_state.json
```

## Next Steps

- **Read full documentation**: [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md)
- **Learn about environments**: [TRAINING_SESSIONS.md](../TRAINING_SESSIONS.md)
- **Understand architecture**: [ARCHITECTURE.md](../ARCHITECTURE.md)
- **Deploy to AWS**: [AWS Deployment Guide](TRAINING_PIPELINE.md#aws-deployment)

## Quick Reference

### Local Commands
```bash
make pipeline              # Run complete pipeline
make pipeline-resume       # Resume interrupted pipeline
make pipeline-s3          # Run with S3 storage
```

### AWS Commands
```bash
make aws-deploy-sm        # Deploy SageMaker infrastructure
make aws-push             # Push Docker image to ECR
make aws-pipeline         # Run pipeline on SageMaker
```

### Evaluation
```bash
make evaluate MODEL=pipeline_output/session2_crush_resistance/models/best_model.zip
```

## FAQ

**Q: How long does the pipeline take?**
A: Locally: 2-4 hours. On AWS with GPU: 20-40 minutes.

**Q: Can I run just one session?**
A: Yes, use `python3 train.py --config configs/training_config.yaml`

**Q: How much does AWS cost?**
A: With spot instances: ~$1-2 for complete pipeline (ml.g4dn.xlarge)

**Q: Can I add my own environments?**
A: Yes! See [Adding Environments Guide](TRAINING_PIPELINE.md#custom-session-orchestration)

**Q: Does it support multi-GPU training?**
A: Currently single GPU, but SageMaker supports distributed training

**Q: Can I visualize training?**
A: Yes, use TensorBoard: `tensorboard --logdir pipeline_output/`

## Support

- **Documentation**: [docs/](.)
- **Issues**: https://github.com/gaskiwi/cautious-meme/issues
- **Examples**: [examples/](../examples/)

---

Ready to train? Run `make pipeline` and watch your agents learn!
